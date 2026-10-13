use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

use anyhow::{Context, Result};
use smart_compress_codecs::cancellation_requested;

use super::{BatchSummary, Optimizer, count_failures, inputs::input_identity, unix_time_ms};
use crate::{
    atomic_file::atomic_write,
    config::{CliConfig, SearchMode},
    report::{
        ImageOutcome, ImageReport, METRIC_COLOR_SPACE, REPORT_SCHEMA_VERSION, ReportOutput,
        RunManifest, RunStatus,
    },
};

/// How often the report of a run with `--report-out` is rewritten at most.
const CHECKPOINT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(3);

impl Optimizer {
    /// The report of a run that is going on or has ended.
    pub(super) fn report_output(
        &self,
        reports: Vec<ImageReport>,
        failures: usize,
        manifest: RunManifest,
    ) -> ReportOutput {
        ReportOutput {
            schema_version: REPORT_SCHEMA_VERSION,
            metric_color_space: METRIC_COLOR_SPACE.to_string(),
            tools: Some(self.backend.tools().clone()),
            gpu: Some((*self.gpu).clone()),
            stage_totals: ReportOutput::calculate_stage_totals(&reports),
            warnings: self.run_warnings(),
            manifest: Some(manifest),
            failures,
            reports,
        }
    }

    /// Processes the inputs on file workers, one thread each. They hold no
    /// shared resource while they wait, so they never block the candidates
    /// they are waiting for.
    pub(super) fn process_files(&self, inputs: &[PathBuf]) -> Vec<ImageReport> {
        let files = self.file_workers(inputs.len());
        let next = AtomicUsize::new(0);
        let slots = inputs.iter().map(|_| Mutex::new(None)).collect::<Vec<_>>();
        std::thread::scope(|scope| {
            for _ in 0..files {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(input) = inputs.get(index) else {
                            break;
                        };
                        let report = self.process_one(input);
                        *slots[index].lock().expect("report slot poisoned") = Some(report);
                    }
                });
            }
        });
        slots
            .into_iter()
            .map(|slot| {
                slot.into_inner()
                    .expect("report slot poisoned")
                    .expect("every input is processed")
            })
            .collect()
    }

    fn file_workers(&self, inputs: usize) -> usize {
        self.config
            .parallelism
            .files
            .unwrap_or(self.config.workers)
            .clamp(1, inputs.max(1))
    }

    /// Like `process_files`, but the report is rewritten while the run goes on
    /// (at most every `CHECKPOINT_INTERVAL`, so a long batch is not rewritten
    /// after every image), and a cancelled run keeps what it finished.
    pub(super) fn run_with_checkpoints(&self, inputs: Vec<PathBuf>) -> Result<BatchSummary> {
        self.pool()?;
        let (resumed, resume_decision) = self.load_resume_reports(&inputs);
        let position = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| (input.display().to_string(), index))
            .collect::<HashMap<_, _>>();
        let resumed_names = resumed
            .iter()
            .map(|report| report.input.as_str())
            .collect::<HashSet<_>>();
        let pending = inputs
            .iter()
            .filter(|input| !resumed_names.contains(input.display().to_string().as_str()))
            .collect::<Vec<_>>();
        drop(resumed_names);
        // The first write fails the run early if the report cannot be written.
        self.write_checkpoint(
            &resumed,
            inputs.len(),
            RunStatus::Partial,
            resume_decision.as_deref(),
        )?;
        let state = Mutex::new((resumed, Instant::now(), None::<anyhow::Error>));
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..self.file_workers(pending.len()) {
                scope.spawn(|| {
                    while !cancellation_requested() {
                        let Some(input) = pending.get(next.fetch_add(1, Ordering::Relaxed)) else {
                            break;
                        };
                        let report = self.process_one(input);
                        let mut state = state.lock().unwrap_or_else(|held| held.into_inner());
                        let (reports, last_write, error) = &mut *state;
                        reports.push(report);
                        if error.is_none() && last_write.elapsed() >= CHECKPOINT_INTERVAL {
                            *last_write = Instant::now();
                            if let Err(failed) = self.write_checkpoint(
                                reports,
                                inputs.len(),
                                RunStatus::Partial,
                                resume_decision.as_deref(),
                            ) {
                                *error = Some(failed);
                            }
                        }
                        if error.is_some() {
                            break;
                        }
                    }
                });
            }
        });
        let (mut reports, _, error) = state.into_inner().unwrap_or_else(|held| held.into_inner());
        if let Some(error) = error {
            return Err(error);
        }
        reports.sort_by_key(|report| position.get(&report.input).copied());
        let cancelled = cancellation_requested() && reports.len() < inputs.len();
        let status = if cancelled {
            RunStatus::Partial
        } else {
            RunStatus::Complete
        };
        let failures = count_failures(&reports) + usize::from(cancelled);
        let manifest = self.manifest(&reports, inputs.len(), status, resume_decision.as_deref());
        let report = self.report_output(reports, failures, manifest);
        self.write_report_file(&report)?;
        self.progress.batch_finished();
        Ok(BatchSummary { report })
    }

    fn write_checkpoint(
        &self,
        reports: &[ImageReport],
        expected_inputs: usize,
        status: RunStatus,
        resume_decision: Option<&str>,
    ) -> Result<()> {
        let manifest = self.manifest(reports, expected_inputs, status, resume_decision);
        let report = self.report_output(reports.to_vec(), count_failures(reports), manifest);
        self.write_report_file(&report)
    }

    fn run_warnings(&self) -> Vec<String> {
        self.run_warnings
            .lock()
            .expect("run warning lock poisoned")
            .clone()
    }

    fn write_report_file(&self, report: &ReportOutput) -> Result<()> {
        let Some(path) = &self.config.report_out else {
            return Ok(());
        };
        let rendered = report.render(self.config.report)?;
        atomic_write(path, rendered.as_bytes())
            .with_context(|| format!("publishing report {}", path.display()))
    }

    pub(super) fn manifest(
        &self,
        reports: &[ImageReport],
        expected_inputs: usize,
        status: RunStatus,
        resume_decision: Option<&str>,
    ) -> RunManifest {
        RunManifest {
            run_id: self.run_id.clone(),
            status,
            started_unix_ms: self.run_started_unix_ms,
            completed_unix_ms: matches!(status, RunStatus::Complete).then(unix_time_ms),
            expected_inputs,
            completed_inputs: reports.len(),
            config_fingerprint: self.config_fingerprint.clone(),
            resume_decision: resume_decision.map(str::to_string),
        }
    }

    fn load_resume_reports(&self, inputs: &[PathBuf]) -> (Vec<ImageReport>, Option<String>) {
        let Some(path) = &self.config.resume_report else {
            return (Vec::new(), None);
        };
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => {
                return (
                    Vec::new(),
                    Some(format!(
                        "resume ignored: could not read {}: {error}",
                        path.display()
                    )),
                );
            }
        };
        let report = match ReportOutput::from_json(&text) {
            Ok(report) => report,
            Err(error) => {
                return (Vec::new(), Some(format!("resume ignored: {error:#}")));
            }
        };
        if let Err(error) = report.validate() {
            return (
                Vec::new(),
                Some(format!(
                    "resume ignored: report is incomplete or invalid: {error}"
                )),
            );
        }
        let Some(manifest) = report.manifest else {
            return (
                Vec::new(),
                Some("resume ignored: report has no resumable manifest".to_string()),
            );
        };
        if manifest.config_fingerprint != self.config_fingerprint {
            return (
                Vec::new(),
                Some("resume ignored: configuration fingerprint does not match".to_string()),
            );
        }
        let input_names = inputs
            .iter()
            .map(|input| input.display().to_string())
            .collect::<HashSet<_>>();
        let reports = report
            .reports
            .into_iter()
            .filter(|report| {
                // Only reuse inputs that finished cleanly, were run the same way
                // (a dry run wrote nothing), have not changed since, and still
                // have the output they claim.
                input_names.contains(&report.input)
                    && report.errors.is_empty()
                    && report.dry_run == self.config.dry_run
                    && match report.outcome {
                        ImageOutcome::Optimized => report
                            .output
                            .as_deref()
                            .is_some_and(|output| Path::new(output).is_file()),
                        ImageOutcome::Kept | ImageOutcome::DryRun => true,
                        ImageOutcome::Skipped | ImageOutcome::Failed => false,
                    }
                    && report.input_identity.is_some()
                    && report.input_identity == input_identity(Path::new(&report.input))
            })
            .collect::<Vec<_>>();
        let decision = format!("resumed {} compatible completed input(s)", reports.len());
        (reports, Some(decision))
    }
}

pub(super) fn stable_config_text(config: &CliConfig) -> String {
    // Left out for the staged search, so that the fingerprints of reports
    // written before the option existed still match.
    let search = match config.search {
        SearchMode::Staged => "",
        SearchMode::Predicted => ";search=Predicted",
    };
    format!(
        "preset={:?};target={:?};formats={:?};report={:?};min_quality={:?};quality_slack={};metadata={:?};alpha={:?};speed={:?};candidate_timeout={:?};max_image_time={:?};max_candidate_attempts={:?};max_temp_bytes={:?};max_run_temp_bytes={:?};max_memory_bytes={:?};workers={};parallelism={:?};allow_larger={};dry_run={};out_dir={:?};out={:?};overwrite={};protect_inputs={};max_input_bytes={};max_pixels={};max_width={};max_height={};max_decoded_bytes={};max_metadata_bytes={};max_bits_per_pixel={};gpu={:?};cache={:?};metrics={:?};verbose={};inputs={:?}{search}",
        config.preset,
        config.target,
        config.formats.iter().collect::<Vec<_>>(),
        config.report,
        config.min_quality,
        config.quality_slack,
        config.metadata,
        config.alpha,
        config.speed,
        config.candidate_timeout,
        config.max_image_time,
        config.max_candidate_attempts,
        config.max_temp_bytes,
        config.max_run_temp_bytes,
        config.max_memory_bytes,
        config.workers,
        config.parallelism,
        config.allow_larger,
        config.dry_run,
        config.out_dir,
        config.out,
        config.overwrite,
        config.protect_inputs,
        config.max_input_bytes,
        config.max_pixels,
        config.max_width,
        config.max_height,
        config.max_decoded_bytes,
        config.max_metadata_bytes,
        config.max_bits_per_pixel,
        config.gpu,
        config.cache,
        config.metrics,
        config.verbose,
        config.inputs,
    )
}
