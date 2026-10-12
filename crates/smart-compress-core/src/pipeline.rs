use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap, HashSet},
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use ignore::WalkBuilder;
use image::GenericImageView;
use rayon::ThreadPoolBuilder;
use smart_compress_codecs::{
    Budget, Codec, EncodeError, EncodeJob, EncodedArtifact, EncoderBackend, EncoderSettings,
    MetadataPayload, ProcessFailure, ToolInventory, WaitEnded, acquire_work,
    cancellation_requested, configure_work_budget, icc_is_plain_srgb, normalized_png, run_process,
    run_process_threads,
};
use smart_compress_metrics::{QualityScore, alpha_edge_similarity};

use crate::{
    atomic_file::{atomic_copy, atomic_write},
    cache::{CachedCandidateMetadata, CandidateCache, file_hash},
    candidate::{
        ActualScoreBackend, CacheHitKind, CandidateInfluence, CandidateReport, CandidateStatus,
        GpuVerificationReport, QualitySource,
    },
    candidate_probe::{check_candidate, probe_avif, read_prefix},
    config::{AlphaPolicy, CliConfig, MetadataPolicy, PrimaryMetric},
    decision::{AcceptedCandidate, DecisionPolicy},
    gpu::{GpuReport, GpuStatus, GpuWorkEstimate, detect_gpu_with_adapter},
    image_probe::{
        DecodedSource, ImageDecodeLimits, ImageProbe, SourceMetadata, apply_orientation,
        decode_image_with_limits, decode_source_with_limits, header_dimensions,
        icc_profile_mismatch, probe_image, read_pam, validate_decoder_dimensions,
    },
    memory,
    progress::{NoopProgress, ProgressObserver},
    report::{
        ImageOutcome, ImageReport, ImageStageTimings, METRIC_COLOR_SPACE, REPORT_SCHEMA_VERSION,
        ReportOutput, RunManifest, RunStatus,
    },
    scoring::{
        ActiveComputeBackend, ComputeBackend, CpuRayonBackend, FULL_SIZE_METRIC_PIXELS,
        GpuPreparedOriginal, GpuWgpuBackend, OriginalImage,
    },
};

const SCORING_VERSION: &str = "score-v10";

/// Folder next to an input that outputs go to when no output folder is given.
const DEFAULT_OUTPUT_DIR: &str = "optimized";

/// How often the report of a run with `--report-out` is rewritten at most.
const CHECKPOINT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(3);

/// Pixels one thread of an external encoder is worth giving; an image with
/// fewer pixels than this gets one thread however many are free. Measured on a
/// 7.7 MP photo, one candidate at a time, with every encoder given 1, 2, 4 and
/// 7 threads: mean encode time 1109, 903, 717 and 653 ms, so the extra threads
/// are worth 61%, 39% and 24% of a thread each. On a 0.26 MP image it made no
/// difference (162, 151, 146 and 146 ms). Three megapixels a thread stops
/// short of the point where a thread is worth less than half a thread.
const PIXELS_PER_ENCODER_THREAD: u64 = 3_000_000;

/// Threads the built-in CPU metric runs on, and the units it takes from the
/// work budget while it runs. On a 7.7 MP photo the metric took 89.5, 73.6, 40.8
/// and 26.1 ms a candidate on 1, 2, 4 and 7 threads (each thread worth 61%,
/// 55% and 49% of a thread beyond the first), against 1.3 s of external scoring
/// and 1.1 s of encoding for the same candidate. It is about 7% of a candidate,
/// so it gets two threads: enough to take its latency off the critical path, few
/// enough that it is not where the budget goes.
const DEFAULT_METRIC_THREADS: usize = 2;

/// One unit of the work budget for an in-process step on a single thread.
fn work(units: usize) -> Result<smart_compress_codecs::Permit<'static>> {
    Ok(acquire_work(units, None)?)
}

fn wait_stop_reason(ended: WaitEnded, image_time: Option<std::time::Duration>) -> String {
    match (ended, image_time) {
        (WaitEnded::TimedOut, Some(limit)) => {
            format!("image wall-time budget of {limit:?} reached")
        }
        _ => "cancelled".to_string(),
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

/// The folder a file is in; the working directory for a bare name.
fn folder_of(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

fn count_failures(reports: &[ImageReport]) -> usize {
    reports
        .iter()
        .filter(|report| report.outcome == ImageOutcome::Failed)
        .count()
}

/// The quality floor on the scale of the metric an image is scored with.
fn quality_floor(config: &CliConfig, perceptual: bool) -> f32 {
    config
        .min_quality
        .unwrap_or_else(|| config.preset.min_quality(perceptual))
}

/// The format a file name asks for, by its extension.
fn codec_for_extension(extension: &str) -> Option<Codec> {
    match extension.to_ascii_lowercase().as_str() {
        "png" => Some(Codec::Png),
        "jpg" | "jpeg" => Some(Codec::Jpeg),
        "webp" => Some(Codec::Webp),
        "avif" => Some(Codec::Avif),
        "jxl" => Some(Codec::Jxl),
        _ => None,
    }
}

fn codec_name(codec: Codec) -> String {
    format!("{codec:?}").to_ascii_lowercase()
}

fn input_identity(path: &Path) -> Option<String> {
    let mut file = fs::File::open(path).ok()?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Some(hasher.finalize().to_hex().to_string())
}

fn process_failure_summary(
    error: &anyhow::Error,
) -> Option<crate::candidate::ProcessFailureSummary> {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<ProcessFailure>())
        .map(crate::candidate::ProcessFailureSummary::from)
}

fn directory_bytes(path: &Path) -> u64 {
    let mut total = 0_u64;
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if let Ok(metadata) = entry.metadata() {
            if metadata.is_dir() {
                total = total.saturating_add(directory_bytes(&path));
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    total
}

struct TempUsageGuard {
    usage: Arc<AtomicU64>,
    /// Woken when usage drops, for candidates waiting for room.
    gate: Arc<Budget>,
    current: u64,
}

impl TempUsageGuard {
    fn new(usage: Arc<AtomicU64>, gate: Arc<Budget>) -> Self {
        Self {
            usage,
            gate,
            current: 0,
        }
    }

    fn set(&mut self, value: u64, limit: Option<u64>) -> Result<()> {
        if value > self.current {
            let delta = value - self.current;
            let total = self
                .usage
                .fetch_add(delta, Ordering::SeqCst)
                .saturating_add(delta);
            if let Some(limit) = limit
                && total > limit
            {
                self.usage.fetch_sub(delta, Ordering::SeqCst);
                return Err(anyhow!(
                    "run temporary storage budget of {limit} bytes exceeded ({total} bytes)"
                ));
            }
        } else {
            self.usage.fetch_sub(self.current - value, Ordering::SeqCst);
            self.gate.wake();
        }
        self.current = value;
        Ok(())
    }
}

impl Drop for TempUsageGuard {
    fn drop(&mut self) {
        self.usage.fetch_sub(self.current, Ordering::SeqCst);
        self.gate.wake();
    }
}

fn optional_delta(left: Option<f32>, right: Option<f32>) -> f32 {
    match (left, right) {
        (Some(left), Some(right)) => (left - right).abs(),
        (None, None) => 0.0,
        _ => f32::INFINITY,
    }
}

#[derive(Debug, Clone)]
pub struct BatchSummary {
    pub report: ReportOutput,
}

pub struct Optimizer {
    config: CliConfig,
    backend: Arc<EncoderBackend>,
    gpu: Arc<GpuReport>,
    compute: Arc<ActiveComputeBackend>,
    progress: Arc<dyn ProgressObserver>,
    cache: Option<Arc<CandidateCache>>,
    run_id: String,
    run_started_unix_ms: u64,
    config_fingerprint: String,
    run_temp_usage: Arc<AtomicU64>,
    ambiguous_outputs: Mutex<HashSet<PathBuf>>,
    /// Folders that hold an input, as `fs::canonicalize` names them.
    input_folders: Mutex<HashSet<PathBuf>>,
    use_ssimulacra2: bool,
    /// The floor on the scale of the metric in use.
    min_quality: f32,
    /// The format `--out` names by its extension; only it is tried.
    out_codec: Option<Codec>,
    /// Things about the run as a whole, for the report.
    run_warnings: Mutex<Vec<String>>,
    /// Memory the run plans on using, shared by the three budgets below.
    memory_limit: u64,
    /// Candidates being encoded or scored right now, across all images.
    active_candidates: AtomicUsize,
    /// Images past admission and not yet finished.
    active_files: AtomicUsize,
    /// Runs the candidates of every image and the built-in metrics; file
    /// workers are plain threads that only wait on it.
    pool: OnceLock<rayon::ThreadPool>,
    /// Threads for the built-in CPU metric; a metric holds this many units of
    /// the work budget, so its parallelism is what the budget counted.
    metric_pool: rayon::ThreadPool,
    metric_threads: usize,
    /// Bytes of memory images reserve before they are decoded, and bytes
    /// candidates reserve before they start. Separate budgets, so an image
    /// that holds its own memory never waits on candidates waiting on it.
    file_memory: Budget,
    candidate_memory: Budget,
    /// Bytes the external metric processes of all candidates may hold at once:
    /// about three at a time on 12 MP photos. Taken after the candidate memory
    /// and held only while waiting for a unit of work.
    metric_memory: Budget,
    /// Woken when temporary files are freed or a candidate finishes, for
    /// candidates waiting for room under the soft temporary-disk limit.
    temp_gate: Arc<Budget>,
}

impl Optimizer {
    pub fn new(config: CliConfig) -> Self {
        Self::with_progress(config, Arc::new(NoopProgress))
    }

    pub fn with_progress(config: CliConfig, progress: Arc<dyn ProgressObserver>) -> Self {
        configure_work_budget(config.workers);
        let run_started_unix_ms = unix_time_ms();
        let mut fingerprint_config = config.clone();
        fingerprint_config.report_out = None;
        fingerprint_config.resume_report = None;
        let config_fingerprint = blake3::hash(stable_config_text(&fingerprint_config).as_bytes())
            .to_hex()
            .to_string();
        let run_id = blake3::hash(format!("{run_started_unix_ms}:{config_fingerprint}").as_bytes())
            .to_hex()
            .to_string();
        let tools = ToolInventory::detect();
        let use_ssimulacra2 = config.metrics.ssimulacra2
            || match config.metrics.primary {
                PrimaryMetric::Auto => tools.ssimulacra2.enabled,
                PrimaryMetric::Ssimulacra2 => true,
                PrimaryMetric::Builtin => false,
            };
        let min_quality = quality_floor(&config, use_ssimulacra2);
        let out_codec = config
            .out
            .as_deref()
            .and_then(Path::extension)
            .and_then(|extension| extension.to_str())
            .and_then(codec_for_extension);
        let cache = CandidateCache::from_config(&config.cache, &tools).map(|cache| {
            let _ = cache.prune(config.cache.max_bytes, config.cache.max_age_days);
            Arc::new(cache)
        });
        let (mut gpu, gpu_adapter) = detect_gpu_with_adapter(&config.gpu);
        let use_gpu_metrics = config.gpu.enabled
            && config.gpu.prefer_gpu_metrics
            && matches!(gpu.status, GpuStatus::Enabled);
        let compute = if let Some(adapter) = gpu_adapter.filter(|_| use_gpu_metrics) {
            match GpuWgpuBackend::from_adapter(adapter, config.gpu.max_vram_bytes) {
                Ok(backend) => ActiveComputeBackend::Gpu(Box::new(backend)),
                Err(error) => {
                    gpu.warnings
                        .push(format!("GPU metric backend unavailable: {error:#}"));
                    if config.gpu.required {
                        gpu.status = GpuStatus::RequiredUnavailable;
                    }
                    ActiveComputeBackend::Cpu(CpuRayonBackend)
                }
            }
        } else {
            ActiveComputeBackend::Cpu(CpuRayonBackend)
        };
        let metric_threads = config
            .parallelism
            .metric_threads
            .unwrap_or(DEFAULT_METRIC_THREADS)
            .clamp(1, config.workers.max(1));
        let metric_pool = ThreadPoolBuilder::new()
            .num_threads(metric_threads)
            .build()
            .expect("building metric pool");
        let memory_limit = config
            .max_memory_bytes
            .unwrap_or_else(|| memory::default_memory_limit(memory::physical_memory()));
        let shares = memory::shares(memory_limit);
        Self {
            config,
            backend: Arc::new(EncoderBackend::new(tools)),
            gpu: Arc::new(gpu),
            compute: Arc::new(compute),
            progress,
            cache,
            run_id,
            run_started_unix_ms,
            config_fingerprint,
            run_temp_usage: Arc::new(AtomicU64::new(0)),
            ambiguous_outputs: Mutex::new(HashSet::new()),
            input_folders: Mutex::new(HashSet::new()),
            use_ssimulacra2,
            min_quality,
            out_codec,
            run_warnings: Mutex::new(Vec::new()),
            memory_limit,
            active_candidates: AtomicUsize::new(0),
            active_files: AtomicUsize::new(0),
            pool: OnceLock::new(),
            metric_pool,
            metric_threads,
            file_memory: Budget::new(shares.files),
            candidate_memory: Budget::new(shares.candidates),
            metric_memory: Budget::new(shares.external_metric),
            temp_gate: Arc::new(Budget::new(1)),
        }
    }

    pub fn run(&self) -> Result<BatchSummary> {
        let (inputs, warnings) = self.discover_inputs()?;
        self.check_out_target(inputs.len())?;
        *self.run_warnings.lock().expect("run warning lock poisoned") = warnings;
        *self
            .ambiguous_outputs
            .lock()
            .expect("output name lock poisoned") = self.colliding_inputs(&inputs);
        *self
            .input_folders
            .lock()
            .expect("input folder lock poisoned") = inputs
            .iter()
            .filter_map(|input| fs::canonicalize(folder_of(input)).ok())
            .collect();
        self.progress.batch_started(inputs.len());
        if self.config.report_out.is_some() {
            return self.run_with_checkpoints(inputs);
        }
        self.pool()?;
        let reports = self.process_files(&inputs);
        self.progress.batch_finished();
        let failures = count_failures(&reports);
        let status = if cancellation_requested() {
            RunStatus::Partial
        } else {
            RunStatus::Complete
        };
        let manifest = self.manifest(&reports, inputs.len(), status, None);
        Ok(BatchSummary {
            report: self.report_output(reports, failures, manifest),
        })
    }

    /// The report of a run that is going on or has ended.
    fn report_output(
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

    /// The pool candidates run on, built on first use. Twice as many threads as
    /// units of work: a candidate waiting for its units holds a thread and no
    /// CPU, and a few waiting keep the budget full while others start up.
    fn pool(&self) -> Result<&rayon::ThreadPool> {
        if let Some(pool) = self.pool.get() {
            return Ok(pool);
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(self.config.workers.max(1) * 2)
            .build()
            .context("building worker pool")?;
        Ok(self.pool.get_or_init(|| pool))
    }

    /// Processes the inputs on file workers, one thread each. They hold no
    /// shared resource while they wait, so they never block the candidates
    /// they are waiting for.
    fn process_files(&self, inputs: &[PathBuf]) -> Vec<ImageReport> {
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
    fn run_with_checkpoints(&self, inputs: Vec<PathBuf>) -> Result<BatchSummary> {
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

    /// `--out` names the one file to write, so exactly one image may be found,
    /// and its format must be among those allowed.
    fn check_out_target(&self, inputs: usize) -> Result<()> {
        let Some(out) = &self.config.out else {
            return Ok(());
        };
        anyhow::ensure!(
            inputs == 1,
            "--out names one file, but {inputs} input images were found; use --out-dir for several"
        );
        if let Some(codec) = self.out_codec {
            anyhow::ensure!(
                self.config.formats.contains(codec),
                "--out {} names the {} format, which is not among the selected formats",
                out.display(),
                codec_name(codec)
            );
        }
        Ok(())
    }

    fn write_report_file(&self, report: &ReportOutput) -> Result<()> {
        let Some(path) = &self.config.report_out else {
            return Ok(());
        };
        let rendered = report.render(self.config.report)?;
        atomic_write(path, rendered.as_bytes(), false)
            .with_context(|| format!("publishing report {}", path.display()))
    }

    fn manifest(
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

    /// The images to process, each once, and what went wrong while looking
    /// for them. Every file counts, whatever its folder is called or an ignore
    /// file says; the output folder of this run does not.
    fn discover_inputs(&self) -> Result<(Vec<PathBuf>, Vec<String>)> {
        let mut files = Vec::new();
        let mut warnings = Vec::new();
        let mut seen = HashSet::new();
        let output_dir = self
            .config
            .out_dir
            .as_deref()
            .and_then(|dir| fs::canonicalize(dir).ok());
        let mut add = |path: &Path| {
            if seen.insert(fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())) {
                files.push(path.to_path_buf());
            }
        };
        // Outputs of this run, and of earlier ones, are not found by a walk:
        // those under the output folder, or under a default one below the
        // folder walked. A file named on the command line is always an input.
        let is_output = |path: &Path, root: &Path| {
            if self.config.out_dir.is_some() {
                output_dir.as_ref().is_some_and(|dir| {
                    fs::canonicalize(path).is_ok_and(|path| path.starts_with(dir))
                })
            } else {
                path.strip_prefix(root)
                    .ok()
                    .and_then(Path::parent)
                    .is_some_and(|inside| {
                        inside
                            .components()
                            .any(|part| part.as_os_str().eq_ignore_ascii_case(DEFAULT_OUTPUT_DIR))
                    })
            }
        };
        for input in &self.config.inputs {
            if input.is_file() {
                add(input);
            } else if input.is_dir() {
                if !self.config.recursive {
                    anyhow::bail!(
                        "{} is a directory; pass --recursive to process it",
                        input.display()
                    );
                }
                let walker = WalkBuilder::new(input)
                    .follow_links(false)
                    .standard_filters(false)
                    .build();
                for entry in walker {
                    match entry {
                        Ok(entry)
                            if entry.file_type().is_some_and(|kind| kind.is_file())
                                && is_supported_image(entry.path())
                                && !is_output(entry.path(), input) =>
                        {
                            add(entry.path());
                        }
                        Ok(_) => {}
                        Err(error) => warnings.push(format!(
                            "could not read part of {}: {error}",
                            input.display()
                        )),
                    }
                }
            } else {
                anyhow::bail!("input does not exist: {}", input.display());
            }
        }

        if files.is_empty() {
            anyhow::bail!("no input images found");
        }
        Ok((files, warnings))
    }

    /// Inputs whose output directory and file stem coincide would publish to
    /// the same file whenever their winners share a codec.
    fn colliding_inputs(&self, inputs: &[PathBuf]) -> HashSet<PathBuf> {
        let mut by_target: HashMap<(PathBuf, String), Vec<&PathBuf>> = HashMap::new();
        for input in inputs {
            let stem = input
                .file_stem()
                .map(|stem| stem.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            by_target
                .entry((self.output_dir(input), stem))
                .or_default()
                .push(input);
        }
        by_target
            .into_values()
            .filter(|group| group.len() > 1)
            .flatten()
            .cloned()
            .collect()
    }

    fn output_dir(&self, input: &Path) -> PathBuf {
        self.config.out_dir.clone().unwrap_or_else(|| {
            input
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(DEFAULT_OUTPUT_DIR)
        })
    }

    fn should_use_gpu_preprocess(&self, image: &image::DynamicImage) -> bool {
        self.config.gpu.enabled
            && self.config.gpu.prefer_gpu_preprocess
            && self.compute.as_ref().as_gpu().is_some()
            && image.width() as u64 * image.height() as u64 > FULL_SIZE_METRIC_PIXELS
    }

    fn should_use_gpu_metrics(&self, image: &image::DynamicImage) -> bool {
        // A required GPU overrides the size threshold; routing small images to
        // CPU would be a silent fallback.
        self.compute.as_ref().as_gpu().is_some()
            && (self.config.gpu.required
                || image.width() as u64 * image.height() as u64
                    >= self.config.gpu.min_pixels_for_gpu_metrics)
    }

    fn process_one(&self, input: &Path) -> ImageReport {
        let total_start = Instant::now();
        let waited = AtomicU64::new(0);
        // A panic in a decoder or the GPU stack costs this image, not the batch.
        // Everything the image holds (memory and disk budgets, temporary files,
        // progress) is released by Drop while the stack unwinds.
        let mut report = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.process_one_inner(input, &waited)
        }))
        .unwrap_or_else(|panic| {
            let mut report = self.new_report(
                input,
                fs::metadata(input).map_or(0, |metadata| metadata.len()),
                input_identity(input),
            );
            report
                .errors
                .push(format!("internal error: {}", panic_message(panic.as_ref())));
            report
        });
        report.timings.total_ms = Some(total_start.elapsed().as_millis() as u64);
        report.timings.backpressure_wait_ms = Some(waited.load(Ordering::Relaxed));
        report
    }

    /// Waits, when the run's temporary files are over the soft limit (three
    /// quarters of the run budget), for room to open up. Only running work
    /// frees room, so with nothing running the wait is over: the work that
    /// cannot fit then runs alone instead of waiting for room nobody will make.
    fn wait_for_temp_room(
        &self,
        deadline: Option<Instant>,
        waited: &AtomicU64,
    ) -> Result<(), WaitEnded> {
        let Some(limit) = self.config.max_run_temp_bytes else {
            return Ok(());
        };
        let soft = limit / 4 * 3;
        let start = Instant::now();
        let result = self.temp_gate.wait_while(deadline, || {
            self.run_temp_usage.load(Ordering::SeqCst) > soft
                && self.active_candidates.load(Ordering::Relaxed) > 0
        });
        waited.fetch_add(start.elapsed().as_millis() as u64, Ordering::Relaxed);
        result
    }

    /// Pixels in the source from its header alone, so memory can be reserved
    /// before the image is decoded. An unreadable header falls back to a guess
    /// from the file size; the decode will report what is wrong with the file.
    fn source_pixels(&self, input: &Path, file_bytes: u64) -> u64 {
        let from_header = if input
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("avif"))
        {
            read_prefix(input)
                .and_then(|header| probe_avif(&header))
                .and_then(|header| header.dimensions)
        } else {
            header_dimensions(input, self.image_decode_limits())
        };
        from_header
            .map_or(file_bytes.saturating_mul(4), |(width, height)| {
                u64::from(width) * u64::from(height)
            })
            // Larger images are refused when they are decoded.
            .min(self.config.max_pixels)
    }

    fn new_report(
        &self,
        input: &Path,
        original_size: u64,
        source_identity: Option<String>,
    ) -> ImageReport {
        ImageReport {
            input: input.display().to_string(),
            input_identity: source_identity,
            output: None,
            original_size,
            saved_bytes: None,
            probe: None,
            gpu_work: None,
            gpu_preprocess_backend: None,
            compute_backend: self.compute.backend_id().to_string(),
            scoring_version: SCORING_VERSION.to_string(),
            quality_metric: if self.use_ssimulacra2 {
                "ssimulacra2"
            } else {
                "builtin"
            }
            .to_string(),
            min_quality: Some(self.min_quality),
            timings: ImageStageTimings::default(),
            candidates: Vec::new(),
            winner_index: None,
            outcome: ImageOutcome::Failed,
            errors: Vec::new(),
            warnings: Vec::new(),
            dry_run: self.config.dry_run,
            search_stop_reason: None,
            temporary_bytes_current: 0,
            temporary_bytes_peak: 0,
            temporary_budget_bytes: self.config.max_temp_bytes,
        }
    }

    fn process_one_inner(&self, input: &Path, waited: &AtomicU64) -> ImageReport {
        let image_start = Instant::now();
        let _file_progress = FileProgressGuard::new(Arc::clone(&self.progress), input);
        let source_open_start = Instant::now();
        let original_size = fs::metadata(input).map_or(0, |metadata| metadata.len());
        let source_identity = input_identity(input);
        let source_open_ms = source_open_start.elapsed().as_millis() as u64;
        let mut report = self.new_report(input, original_size, source_identity);
        report.timings.source_open_ms = Some(source_open_ms);

        if cancellation_requested() {
            report.errors.push("cancelled".to_string());
            return report;
        }
        if matches!(self.gpu.status, GpuStatus::RequiredUnavailable) {
            report.errors.push(
                "GPU is required by configuration, but no usable GPU backend is available"
                    .to_string(),
            );
            return report;
        }

        let workers = self.config.workers.max(1);
        let image_deadline = self.config.max_image_time.map(|limit| image_start + limit);
        let temp_budgeted =
            self.config.max_temp_bytes.is_some() || self.config.max_run_temp_bytes.is_some();
        let source_pixels = self.source_pixels(input, original_size);
        let alone = memory::single_image_bytes(
            source_pixels,
            self.use_ssimulacra2
                .then_some(source_pixels.min(FULL_SIZE_METRIC_PIXELS)),
        );
        if alone > self.memory_limit {
            report.warnings.push(format!(
                "this image alone is estimated to need {} MiB, more than the whole memory limit of {} MiB; it runs on its own and may use more than the limit",
                alone >> 20,
                self.memory_limit >> 20
            ));
        }
        // Candidates of one image run in the stages described below; with a
        // temporary-disk budget they run one at a time. The slots come from the
        // memory model; `concurrency` may be set apart from them to measure.
        let parallelism = if temp_budgeted {
            1
        } else {
            memory::candidate_slots(self.candidate_memory.limit(), source_pixels, workers)
        };
        let concurrency = self
            .config
            .parallelism
            .candidates
            .map_or(parallelism, |candidates| candidates.max(1));
        // The image reserves its memory before it is decoded. One larger than
        // the limit is granted the whole of it, and so runs alone.
        let admission_start = Instant::now();
        let reserved = self
            .file_memory
            .acquire(memory::file_bytes(source_pixels), image_deadline);
        waited.fetch_add(
            admission_start.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        let _reserved = match reserved {
            Ok(permit) => permit,
            Err(ended) => {
                report.errors.push(format!(
                    "{} while waiting for memory",
                    wait_stop_reason(ended, self.config.max_image_time)
                ));
                return report;
            }
        };

        self.active_files.fetch_add(1, Ordering::Relaxed);
        let _active_file = ActiveCandidate(&self.active_files, &self.temp_gate);
        #[cfg(test)]
        assert!(
            input.file_name().is_none_or(|name| name != "panic.png"),
            "injected panic"
        );

        let source_decode_start = Instant::now();
        let decoded = self.decode_input(input);
        report.timings.source_decode_ms = Some(source_decode_start.elapsed().as_millis() as u64);
        let mut decoded = match decoded {
            Ok(value) => value,
            Err(error) => {
                report.errors.push(format!("{error:#}"));
                return report;
            }
        };
        // The profile is judged against the pixels that will be written, so
        // the alpha policy comes first: it may turn grey into RGB.
        let alpha_changed = match self.apply_alpha_policy(&decoded.image) {
            Ok(Some(image)) => {
                decoded.image = image;
                true
            }
            Ok(None) => false,
            Err(error) => {
                report.errors.push(format!("{error:#}"));
                return report;
            }
        };
        if let Some(reason) = decoded
            .metadata
            .icc_profile
            .as_deref()
            .and_then(|profile| icc_profile_mismatch(profile, decoded.image.color()))
        {
            report
                .warnings
                .push(format!("embedded ICC profile dropped: {reason}"));
            decoded.metadata.icc_profile = None;
        }
        if decoded.metadata.original_color_type == "Cmyk8" {
            report.warnings.push(
                "CMYK source converted to RGB by the decoder without a colour-managed transform"
                    .to_string(),
            );
        }
        report.warnings.append(&mut decoded.metadata.warnings);
        let orientation_applied = decoded.metadata.orientation_applied();
        let mut prepared_original = None;
        let mut preprocess_ms = None;
        let mut gpu_preprocess_fallback_reason = None;
        if self.should_use_gpu_preprocess(&decoded.image)
            && let Some(gpu) = self.compute.as_ref().as_gpu()
        {
            let preprocess_start = Instant::now();
            let gpu_unit = match work(1) {
                Ok(unit) => unit,
                Err(error) => {
                    report.errors.push(format!("{error:#}"));
                    return report;
                }
            };
            let prepared = gpu.prepare_original(&decoded.image, decoded.metadata.orientation);
            drop(gpu_unit);
            match prepared {
                Ok(prepared) => {
                    report.gpu_preprocess_backend = Some(prepared.preprocess_mode().to_string());
                    if prepared.preprocess_mode().starts_with("cpu") {
                        report.warnings.push(format!(
                            "prepared preprocessing used {} because the full source exceeded the GPU dispatch budget",
                            prepared.preprocess_mode()
                        ));
                    }
                    prepared_original = Some(prepared);
                }
                Err(error) => {
                    let reason = format!("GPU preprocess failed: {error:#}");
                    if self.config.gpu.required {
                        report.errors.push(reason);
                        return report;
                    }
                    report
                        .warnings
                        .push(format!("CPU scoring fallback: {reason}"));
                    gpu_preprocess_fallback_reason = Some(reason);
                }
            }
            preprocess_ms = Some(preprocess_start.elapsed().as_millis() as u64);
            report.timings.gpu_preprocess_ms = preprocess_ms;
        }

        let orientation_start = Instant::now();
        apply_orientation(&mut decoded.image, decoded.metadata.orientation);
        report.timings.orientation_ms = Some(orientation_start.elapsed().as_millis() as u64);
        let original = OriginalImage::from(decoded.image);
        // ssimulacra2 cannot score a side under 8 px; such an image is scored
        // with the built-in metric, against the floor of that metric.
        let perceptual = self.scores_ssimulacra2(original.width(), original.height());
        let min_quality = quality_floor(&self.config, perceptual);
        if perceptual != self.use_ssimulacra2 {
            report.quality_metric = "builtin".to_string();
            report.min_quality = Some(min_quality);
            report.warnings.push(format!(
                "a side is under 8 px, which ssimulacra2 cannot score: scored with the built-in metric against its floor of {min_quality:.1}"
            ));
        }
        let probe_start = Instant::now();
        let probe = probe_image(
            input,
            &original,
            &decoded.metadata,
            prepared_original
                .as_ref()
                .map(GpuPreparedOriginal::overview),
        );
        report.timings.probe_ms = Some(probe_start.elapsed().as_millis() as u64);
        let has_alpha = probe.has_alpha;
        let metadata_payload = metadata_payload(&decoded.metadata, self.config.metadata);
        if self.config.metadata == MetadataPolicy::Strip && metadata_payload.icc_profile.is_some() {
            report.warnings.push(
                "ICC profile kept under --metadata strip: it is not plain sRGB, so dropping it would change the colours"
                    .to_string(),
            );
        }
        if self.compute.backend_id().starts_with("gpu") {
            report.gpu_work = Some(GpuWorkEstimate::for_image(
                probe.width,
                probe.height,
                self.config.gpu.max_vram_bytes,
                self.gpu.selected_adapter.as_ref(),
            ));
        }
        // Report the bands that were actually uploaded, not the estimate.
        if let (Some(work), Some(band_pixels)) = (
            report.gpu_work.as_mut(),
            prepared_original
                .as_ref()
                .and_then(GpuPreparedOriginal::band_pixels),
        ) {
            work.tiling_required_for_full_source_preprocess = true;
            work.selected_tile_pixels = Some(band_pixels);
        }
        report.probe = Some(probe);

        if self.config.dry_run {
            report.outcome = ImageOutcome::DryRun;
            return report;
        }
        if self.use_ssimulacra2 && !self.backend.tools().ssimulacra2.enabled {
            report.errors.push(
                "SSIMULACRA2 metric requested, but ssimulacra2 tool is not available".to_string(),
            );
            return report;
        }
        if self.config.metrics.butteraugli && !self.backend.tools().butteraugli.enabled {
            report.errors.push(
                "Butteraugli metric requested, but butteraugli_main tool is not available"
                    .to_string(),
            );
            return report;
        }

        let probe = report
            .probe
            .as_ref()
            .expect("probe set after successful decode");
        let mut candidate_plan = self.plan_candidates(probe);
        let gated = !self.config.verbose && lossless_gate_applies(probe, &candidate_plan);
        if candidate_plan.is_empty() {
            report.errors.push(format!(
                "no candidate could be planned for this image: {}",
                self.unplanned_reasons(probe)
            ));
            return report;
        }
        // The destination is known before the search, so an image whose output
        // is already there is not searched at all.
        if !self.config.overwrite {
            let mut existing = None;
            for (codec, _) in &candidate_plan {
                match self.output_path(input, *codec) {
                    Ok(path) if path.exists() => {
                        existing = Some(path);
                        break;
                    }
                    Ok(_) => {}
                    Err(error) => {
                        report.errors.push(format!("{error:#}"));
                        return report;
                    }
                }
            }
            if let Some(path) = existing {
                report.warnings.push(format!(
                    "output already exists, left as it is: {} (pass --overwrite to replace it)",
                    path.display()
                ));
                report.outcome = ImageOutcome::Skipped;
                return report;
            }
        }

        // External encoders copy EXIF/XMP from a source file they read
        // directly, so only `keep` may hand them the untouched input.
        let needs_normalized_source = orientation_applied
            || alpha_changed
            || !external_tools_accept_input(input)
            || !metadata_payload.is_empty()
            || !matches!(self.config.metadata, MetadataPolicy::Keep);
        let normalize_start = Instant::now();
        let normalized = if needs_normalized_source {
            match work(1).and_then(|_unit| normalized_png(&original, &metadata_payload)) {
                Ok(value) => Some(value),
                Err(error) => {
                    report.errors.push(format!("{error:#}"));
                    return report;
                }
            }
        } else {
            None
        };
        report.timings.normalize_ms = Some(normalize_start.elapsed().as_millis() as u64);
        // Tools run in their own folder and read a leading `-` as an option,
        // so the input they are given is absolute.
        let absolute_input = std::path::absolute(input).unwrap_or_else(|_| input.to_path_buf());
        let encoder_source = normalized
            .as_ref()
            .map_or(absolute_input.as_path(), |(_, path)| path.as_path());
        let cache_lookup_start = Instant::now();
        let cache_source_hash = self
            .cache
            .as_ref()
            .and_then(|cache| cache.source_hash(encoder_source).ok());
        report.timings.cache_lookup_ms = Some(cache_lookup_start.elapsed().as_millis() as u64);
        let candidate_dir = match tempfile::tempdir() {
            Ok(dir) => dir,
            Err(error) => {
                report.errors.push(format!("{error:#}"));
                return report;
            }
        };
        let mut temp_usage = TempUsageGuard::new(
            Arc::clone(&self.run_temp_usage),
            Arc::clone(&self.temp_gate),
        );
        let external_metric_prepare_start = Instant::now();
        let external_metric_original = if self.use_ssimulacra2 || self.config.metrics.butteraugli {
            match &normalized {
                // Images this size are scored as they are, so a normalized PNG
                // without metadata already holds the metric original.
                Some((_, path))
                    if metadata_payload.is_empty()
                        && original.width() as u64 * original.height() as u64
                            <= FULL_SIZE_METRIC_PIXELS =>
                {
                    Some(path.clone())
                }
                _ => {
                    let path = candidate_dir.path().join("external-metric-original.png");
                    match write_metric_png(&original, &path) {
                        Ok(()) => Some(path),
                        Err(error) => {
                            report.errors.push(format!(
                                "preparing external metric original failed: {error:#}"
                            ));
                            return report;
                        }
                    }
                }
            }
        } else {
            None
        };
        report.timings.external_metric_prepare_ms =
            Some(external_metric_prepare_start.elapsed().as_millis() as u64);
        let initial_temp_bytes = directory_bytes(candidate_dir.path());
        report.temporary_bytes_current = initial_temp_bytes;
        report.temporary_bytes_peak = initial_temp_bytes;
        if let Err(error) = temp_usage.set(initial_temp_bytes, self.config.max_run_temp_bytes) {
            report.errors.push(format!("{error:#}"));
            return report;
        }
        if let Some(limit) = self.config.max_temp_bytes
            && initial_temp_bytes > limit
        {
            report.errors.push(format!(
                "temporary storage budget of {limit} bytes exceeded ({initial_temp_bytes} bytes)"
            ));
            return report;
        }

        let policy = DecisionPolicy {
            min_quality,
            quality_slack: self.config.quality_slack,
            allow_larger: self.config.allow_larger,
            original_size,
            prefer_compatibility_delta: self.config.preset.compat_delta(),
            target: self.config.target,
        };

        let search_start = Instant::now();
        let mut best: Option<AcceptedCandidate> = None;
        let candidate_total = candidate_plan.len();
        self.progress.candidates_planned(input, candidate_total);
        // Encoding, decoding and scoring a candidate depend only on the source,
        // so the planned grid is evaluated concurrently and the decisions below
        // still run in plan order. Temporary-disk budgets need the one-at-a-time
        // path to stop before the next file is written.
        let pool = self.pool().expect("run builds the worker pool");
        let make_request =
            |candidate_index: usize, codec: Codec, settings: EncoderSettings, peers: usize| {
                CandidateEvaluationRequest {
                    input,
                    codec,
                    settings,
                    encoder_source,
                    original: &original,
                    metadata: &metadata_payload,
                    prepared_original: prepared_original.as_ref(),
                    preprocess_ms,
                    gpu_preprocess_fallback_reason: gpu_preprocess_fallback_reason.as_deref(),
                    has_alpha,
                    candidate_dir: candidate_dir.path(),
                    policy: &policy,
                    cache_source_hash: cache_source_hash.as_deref(),
                    external_metric_original: external_metric_original.as_deref(),
                    candidate_index,
                    progress_index: candidate_index + 1,
                    progress_total: candidate_total,
                    peers,
                    deadline: image_deadline,
                    waited,
                }
            };
        if parallelism == 1 {
            // One at a time, walk each format from its lowest quality up so
            // the loop below can stop a format as soon as more quality can
            // only mean a larger file. The lossless probe goes first, so the
            // other lossless encoders can be gated on it.
            candidate_plan.sort_by_key(|(codec, settings)| {
                let quality = match settings {
                    EncoderSettings::Quality { quality, .. } => 1 + u16::from(*quality),
                    _ => 0,
                };
                // Cheapest encoders first, so slow ones meet a best to be measured against.
                let order = [
                    Codec::Jpeg,
                    Codec::Avif,
                    Codec::Jxl,
                    Codec::Webp,
                    Codec::Png,
                ];
                (
                    !(gated && is_lossless_probe(*codec, settings)),
                    order.iter().position(|other| other == codec),
                    quality,
                )
            });
        }
        // Evaluates one stage of the plan and records it in plan order, whichever
        // candidate finishes first. Returns why the search had to stop, if it did.
        let mut run_stage = |stage: Vec<(Codec, EncoderSettings)>,
                             report: &mut ImageReport,
                             best: &mut Option<AcceptedCandidate>|
         -> Option<String> {
            let first_index = report.candidates.len();
            let stage_best = best.clone();
            let mut evaluated: Vec<Option<CandidateReport>> = Vec::new();
            if parallelism > 1 {
                use rayon::prelude::*;
                let budget = self
                    .config
                    .max_candidate_attempts
                    .map_or(usize::MAX, |limit| limit.saturating_sub(attempts(report)));
                let planned = stage.iter().take(budget).enumerate().collect::<Vec<_>>();
                for chunk in planned.chunks(concurrency) {
                    if self.wait_for_temp_room(image_deadline, waited).is_err() {
                        break;
                    }
                    pool.install(|| {
                        evaluated.par_extend(chunk.par_iter().map(|(index, (codec, settings))| {
                            if cancellation_requested()
                                || self
                                    .config
                                    .max_image_time
                                    .is_some_and(|limit| image_start.elapsed() >= limit)
                            {
                                return None;
                            }
                            // The pass or fail of a quality places the bracket of its
                            // format, so only candidates outside any bracket may be
                            // skipped as dominated.
                            let best = match settings {
                                EncoderSettings::Quality { .. } => None,
                                _ => stage_best.as_ref(),
                            };
                            self.score_candidate_report(
                                make_request(
                                    first_index + index,
                                    *codec,
                                    settings.clone(),
                                    chunk.len(),
                                ),
                                best,
                            )
                        }));
                    });
                }
            }
            let mut search_stop_reason = None;
            // Formats whose remaining, higher qualities cannot win (one-at-a-time path only).
            let mut settled: HashSet<Codec> = HashSet::new();
            for (offset, (codec, settings)) in stage.into_iter().enumerate() {
                if cancellation_requested() {
                    search_stop_reason = Some("cancelled".to_string());
                    break;
                }
                if let Some(limit) = self.config.max_candidate_attempts
                    && attempts(report) >= limit
                {
                    search_stop_reason =
                        Some(format!("candidate attempt budget of {limit} reached"));
                    break;
                }
                if let Some(limit) = self.config.max_image_time
                    && image_start.elapsed() >= limit
                {
                    search_stop_reason =
                        Some(format!("image wall-time budget of {:?} reached", limit));
                    break;
                }
                let lossy = matches!(settings, EncoderSettings::Quality { .. });
                let candidate = match evaluated.get_mut(offset).and_then(Option::take) {
                    Some(candidate) => candidate,
                    None if parallelism == 1 && lossy && settled.contains(&codec) => {
                        CandidateReport::unevaluated(
                            codec,
                            settings,
                            "a lower quality of this format already settled it".to_string(),
                        )
                    }
                    None => match (parallelism == 1 && gated)
                        .then(|| {
                            lossless_gate(
                                &report.candidates,
                                codec,
                                &settings,
                                &policy,
                                best.as_ref(),
                            )
                        })
                        .flatten()
                    {
                        Some(reason) => CandidateReport::unevaluated(codec, settings, reason),
                        None => {
                            if let Err(ended) = self.wait_for_temp_room(image_deadline, waited) {
                                search_stop_reason =
                                    Some(wait_stop_reason(ended, self.config.max_image_time));
                                break;
                            }
                            pool.install(|| {
                                self.score_candidate_report(
                                    make_request(first_index + offset, codec, settings.clone(), 1),
                                    best.as_ref(),
                                )
                            })
                            .unwrap_or_else(|| {
                                CandidateReport::unevaluated(
                                    codec,
                                    settings,
                                    "stopped while waiting for memory".to_string(),
                                )
                            })
                        }
                    },
                };
                // File size grows with quality, so once a quality passes, is
                // dominated, or is already too large, higher ones cannot win.
                let settles = lossy
                    && match &candidate.status {
                        CandidateStatus::Accepted => true,
                        CandidateStatus::Skipped(reason) => reason.starts_with("dominated"),
                        // Too large to be worth scoring; a scored one is a quality verdict.
                        CandidateStatus::Rejected(_) => candidate.quality.is_none(),
                        CandidateStatus::Failed(_) => false,
                    };
                if settles {
                    settled.insert(codec);
                }
                record_candidate(&policy, candidate, report, best);
                let temporary_bytes = directory_bytes(candidate_dir.path());
                report.temporary_bytes_current = temporary_bytes;
                report.temporary_bytes_peak = report.temporary_bytes_peak.max(temporary_bytes);
                if let Err(error) = temp_usage.set(temporary_bytes, self.config.max_run_temp_bytes)
                {
                    search_stop_reason = Some(format!("{error:#}"));
                    break;
                }
                if let Some(limit) = self.config.max_temp_bytes
                    && temporary_bytes > limit
                {
                    search_stop_reason = Some(format!(
                        "temporary storage budget of {limit} bytes exceeded ({temporary_bytes} bytes)"
                    ));
                    break;
                }
            }
            search_stop_reason
        };
        // With candidates running in parallel the coarse grid is staged: the middle
        // quality of every lossy format first, then only the side of it the quality
        // floor must lie on. Verbose mode keeps the whole grid in one stage.
        let StagedPlan { first, held, mids } = if parallelism > 1 && !self.config.verbose {
            stage_plan(candidate_plan, gated)
        } else {
            StagedPlan {
                first: candidate_plan,
                held: Vec::new(),
                mids: HashMap::new(),
            }
        };
        let mut search_stop_reason = run_stage(first, &mut report, &mut best);
        if search_stop_reason.is_none() && !held.is_empty() {
            let mut run = Vec::new();
            let mut skipped = Vec::new();
            for (codec, settings) in held {
                let reason = match (&settings, mids.get(&codec)) {
                    (EncoderSettings::Quality { quality, .. }, Some(mid)) => coarse_followup(
                        &report.candidates,
                        &policy,
                        best.as_ref(),
                        codec,
                        *mid,
                        *quality,
                    ),
                    _ if gated => {
                        lossless_gate(&report.candidates, codec, &settings, &policy, best.as_ref())
                    }
                    _ => None,
                };
                match reason {
                    Some(reason) => {
                        skipped.push(CandidateReport::unevaluated(codec, settings, reason));
                    }
                    None => run.push((codec, settings)),
                }
            }
            search_stop_reason = run_stage(run, &mut report, &mut best);
            // A format that broke the monotonic assumption gets the qualities that
            // were assumed; so does one whose bracket needs a second failing point.
            while search_stop_reason.is_none() {
                let (picked, warnings) = held_back_to_run(
                    &report.candidates,
                    &skipped,
                    self.config.speed.refinement_rounds() > 0,
                );
                if picked.is_empty() {
                    break;
                }
                report.warnings.extend(warnings);
                let run = picked
                    .iter()
                    .map(|&index| (skipped[index].codec, skipped[index].settings.clone()))
                    .collect();
                for &index in picked.iter().rev() {
                    skipped.remove(index);
                }
                search_stop_reason = run_stage(run, &mut report, &mut best);
            }
            for candidate in skipped {
                record_candidate(&policy, candidate, &mut report, &mut best);
            }
        }
        if search_stop_reason.is_none() {
            search_stop_reason = self.refine_quality_search(
                QualityRefinementRequest {
                    input,
                    parallelism: concurrency,
                    image_start,
                    waited,
                    encoder_source,
                    original: &original,
                    metadata: &metadata_payload,
                    prepared_original: prepared_original.as_ref(),
                    preprocess_ms,
                    gpu_preprocess_fallback_reason: gpu_preprocess_fallback_reason.as_deref(),
                    has_alpha,
                    candidate_dir: candidate_dir.path(),
                    policy: &policy,
                    cache_source_hash: cache_source_hash.as_deref(),
                    external_metric_original: external_metric_original.as_deref(),
                },
                &mut report,
                &mut best,
            );
        }
        report.search_stop_reason = search_stop_reason;
        self.progress.candidates_settled(input);
        report.timings.search_ms = Some(search_start.elapsed().as_millis() as u64);
        if cancellation_requested() {
            report.errors.push("cancelled".to_string());
            for candidate in &mut report.candidates {
                candidate.path = None;
            }
            return report;
        }
        if let Some(reason) = &report.search_stop_reason {
            report.warnings.push(format!(
                "the search stopped early ({reason}); a smaller passing candidate may have gone untried"
            ));
        }

        let winner_index = policy.choose(&report.candidates).and_then(|winner| {
            report
                .candidates
                .iter()
                .position(|candidate| std::ptr::eq(candidate, winner))
        });
        report.winner_index = winner_index;

        let output_write_start = Instant::now();
        if let Some(index) = winner_index {
            match self.write_winner(input, &report.candidates[index]) {
                Ok(output) => {
                    let winner_size = report.candidates[index].size_bytes.unwrap_or(original_size);
                    report.saved_bytes = Some(original_size as i64 - winner_size as i64);
                    report.output = Some(output.display().to_string());
                    report.outcome = ImageOutcome::Optimized;
                    if self.config.out.is_some() && self.out_codec.is_none() {
                        report.warnings.push(format!(
                            "--out has no image extension, so .{} was added: {}",
                            report.candidates[index].codec.extension(),
                            output.display()
                        ));
                    }
                }
                Err(error) => report.errors.push(format!("{error:#}")),
            }
        } else if let Some(failed) = report
            .candidates
            .iter()
            .find_map(|candidate| match &candidate.status {
                CandidateStatus::Failed(reason) => Some(reason.clone()),
                _ => None,
            })
        {
            // A candidate that failed has no verdict, so "nothing passed" is not known.
            let count = report
                .candidates
                .iter()
                .filter(|candidate| matches!(candidate.status, CandidateStatus::Failed(_)))
                .count();
            report.errors.push(format!(
                "no acceptable candidate found, and {count} candidate(s) failed before they could be judged (first: {failed})"
            ));
        } else {
            report.warnings.push(format!(
                "nothing scored at least {:.1} while staying smaller than the original; the original is left as it is",
                policy.min_quality - self.config.quality_slack
            ));
            report.outcome = ImageOutcome::Kept;
        }
        report.timings.output_write_ms = Some(output_write_start.elapsed().as_millis() as u64);
        report.temporary_bytes_current = directory_bytes(candidate_dir.path());
        report.temporary_bytes_peak = report
            .temporary_bytes_peak
            .max(report.temporary_bytes_current);
        for candidate in &mut report.candidates {
            candidate.path = None;
        }

        report
    }

    fn plan_candidates(&self, probe: &ImageProbe) -> Vec<(Codec, EncoderSettings)> {
        let mut plan = Vec::new();
        let speed = self.config.speed;
        let large_lossy_source = probe.pixels > 50_000_000
            && probe
                .format
                .as_deref()
                .is_some_and(|format| format.eq_ignore_ascii_case("Jpeg"))
            && !probe.has_alpha
            && !matches!(self.config.target, crate::config::TargetPolicy::Archive);
        for codec in self
            .config
            .formats
            .iter()
            .filter(|codec| self.out_codec.is_none_or(|named| named == *codec))
        {
            if probe.has_alpha
                && !codec.supports_alpha()
                && matches!(self.config.alpha, AlphaPolicy::Preserve)
            {
                continue;
            }
            if !self.backend.tools().available_for(codec) {
                continue;
            }
            match codec {
                Codec::Png => {
                    if !large_lossy_source {
                        plan.push((codec, EncoderSettings::Lossless));
                        if self.backend.tools().pngquant.enabled
                            && probe
                                .format
                                .as_deref()
                                .is_some_and(|format| format.eq_ignore_ascii_case("Png"))
                        {
                            for colors in [256, 128, 64, 32] {
                                plan.push((codec, EncoderSettings::Palette { colors }));
                            }
                        }
                    }
                }
                Codec::Jpeg => {
                    for quality in speed.qualities() {
                        plan.push((
                            codec,
                            EncoderSettings::Quality {
                                quality: *quality,
                                effort: speed.probe_effort(codec),
                            },
                        ));
                    }
                }
                Codec::Webp | Codec::Avif | Codec::Jxl => {
                    if !large_lossy_source {
                        plan.push((codec, EncoderSettings::Lossless));
                    }
                    for quality in speed.qualities() {
                        plan.push((
                            codec,
                            EncoderSettings::Quality {
                                quality: *quality,
                                effort: speed.probe_effort(codec),
                            },
                        ));
                    }
                }
            }
        }
        plan
    }

    /// Why no candidate could be planned, per selected format.
    fn unplanned_reasons(&self, probe: &ImageProbe) -> String {
        self.config
            .formats
            .iter()
            .filter(|codec| self.out_codec.is_none_or(|named| named == *codec))
            .map(|codec| {
                let missing = self.backend.tools().missing_for(codec);
                let name = codec_name(codec);
                if probe.has_alpha
                    && !codec.supports_alpha()
                    && matches!(self.config.alpha, AlphaPolicy::Preserve)
                {
                    format!("{name} cannot store transparency (try --alpha flatten)")
                } else if !missing.is_empty() {
                    format!(
                        "{name} needs {}, which was not found",
                        missing.join(" and ")
                    )
                } else {
                    format!("{name} is not tried on a very large JPEG")
                }
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Reserves the memory a candidate needs, waiting while the candidates in
    /// flight hold it all, and then evaluates it. `None` when the image ran
    /// out of time or the run was cancelled during the wait. A candidate
    /// larger than the limit is granted the whole of it and runs alone.
    fn score_candidate_report(
        &self,
        request: CandidateEvaluationRequest<'_>,
        best: Option<&AcceptedCandidate>,
    ) -> Option<CandidateReport> {
        let pixels = u64::from(request.original.width()) * u64::from(request.original.height());
        let wait_start = Instant::now();
        let reserved = self
            .candidate_memory
            .acquire(memory::candidate_bytes(pixels), request.deadline);
        request
            .waited
            .fetch_add(wait_start.elapsed().as_millis() as u64, Ordering::Relaxed);
        let _reserved = reserved.ok()?;
        Some(self.evaluate_candidate(request, best))
    }

    /// Encodes, decodes and scores one candidate without touching shared
    /// state, so planned candidates can be evaluated concurrently.
    fn evaluate_candidate(
        &self,
        request: CandidateEvaluationRequest<'_>,
        best: Option<&AcceptedCandidate>,
    ) -> CandidateReport {
        let candidate_start = Instant::now();
        let CandidateEvaluationRequest {
            input,
            codec,
            settings,
            encoder_source,
            original,
            metadata,
            prepared_original,
            preprocess_ms,
            gpu_preprocess_fallback_reason,
            has_alpha,
            candidate_dir,
            policy,
            cache_source_hash,
            external_metric_original,
            candidate_index,
            progress_index,
            progress_total,
            peers,
            deadline: _,
            waited: _,
        } = request;
        self.active_candidates.fetch_add(1, Ordering::Relaxed);
        let _active = ActiveCandidate(&self.active_candidates, &self.temp_gate);
        // Split the threads between the candidates that run together and the
        // images in progress, so the budget goes to candidates side by side,
        // which scales better than threads inside one encoder. Encoder output
        // does not depend on the thread count, so the split is free to change.
        let pixels = u64::from(original.width()) * u64::from(original.height());
        let encoder_threads = self.config.parallelism.encoder_threads.unwrap_or_else(|| {
            let useful =
                usize::try_from(pixels / PIXELS_PER_ENCODER_THREAD.max(1)).unwrap_or(usize::MAX);
            let images = self.active_files.load(Ordering::Relaxed).max(1);
            (self.config.workers.max(1) / (images * peers.max(1))).clamp(1, useful.max(1))
        });
        let _candidate_progress = CandidateProgressGuard::new(
            Arc::clone(&self.progress),
            input,
            progress_index,
            progress_total,
            codec,
            &settings,
        );
        let output_path = candidate_path(candidate_dir, codec, &settings);
        let encode_and_cache_start = Instant::now();
        let artifact = self.encode_candidate(CandidateEncodeRequest {
            codec,
            settings: settings.clone(),
            normalized_path: encoder_source,
            original,
            metadata,
            output_path: &output_path,
            has_alpha,
            cache_source_hash,
            // Artifacts that can never be accepted are not worth cache space.
            cache_size_limit: (!policy.allow_larger).then_some(policy.original_size),
            threads: encoder_threads,
        });
        let encode_and_cache_ms = encode_and_cache_start.elapsed().as_millis() as u64;
        let artifact = match artifact {
            Ok(artifact) => artifact,
            Err(error) => {
                let (status, influence) = encode_failure(&error);
                let candidate = CandidateReport {
                    codec,
                    settings,
                    path: None,
                    size_bytes: None,
                    encode_ms: None,
                    decode_ms: None,
                    decode_process_ms: None,
                    decode_io_ms: None,
                    preprocess_ms: None,
                    score_ms: None,
                    cache_ms: None,
                    external_metric_ms: None,
                    gpu_upload_ms: None,
                    gpu_transform_ms: None,
                    gpu_compute_ms: None,
                    gpu_readback_ms: None,
                    gpu_timestamp_ns: None,
                    total_ms: Some(candidate_start.elapsed().as_millis() as u64),
                    quality: None,
                    backend: None,
                    score_backend: None,
                    quality_source: QualitySource::Fresh,
                    cache_hit_kind: None,
                    process_failure: process_failure_summary(&error),
                    gpu_verification: None,
                    influence: Some(influence),
                    bytes: Default::default(),
                    status,
                    warnings: Vec::new(),
                };
                return candidate;
            }
        };

        let mut candidate = CandidateReport::from_artifact(artifact);
        if candidate.backend.as_deref() == Some("cache") {
            candidate.cache_ms = Some(encode_and_cache_ms);
            candidate.cache_hit_kind = Some(CacheHitKind::EncodedArtifact);
        }
        let size = candidate.size_bytes.unwrap_or(u64::MAX);
        if !policy.can_accept_size(size) {
            candidate.status = CandidateStatus::Rejected("not smaller than original".to_string());
            candidate.influence = Some(CandidateInfluence::Rejected);
            candidate.total_ms = Some(candidate_start.elapsed().as_millis() as u64);
            return candidate;
        }
        if !policy.candidate_can_beat(size, best, codec) && !self.config.verbose {
            candidate.status =
                CandidateStatus::Skipped("dominated by current accepted candidate".to_string());
            candidate.influence = Some(CandidateInfluence::Skipped);
            candidate.total_ms = Some(candidate_start.elapsed().as_millis() as u64);
            return candidate;
        }

        let quality_cache_start = Instant::now();
        let quality_cache = self.load_quality_cache(
            cache_source_hash,
            codec,
            &settings,
            has_alpha,
            &mut candidate,
        );
        if quality_cache.is_some() {
            candidate.cache_ms = Some(
                candidate.cache_ms.unwrap_or(0) + quality_cache_start.elapsed().as_millis() as u64,
            );
            candidate.quality_source = QualitySource::ScoreCache;
            candidate.cache_hit_kind = Some(CacheHitKind::ScoreMetadata);
        }
        let quality = match quality_cache {
            Some(quality) => quality,
            None => {
                let decode_start = Instant::now();
                let decoded = match self.decode_candidate(
                    codec,
                    candidate.path.as_ref().expect("candidate path"),
                    original,
                    has_alpha,
                    encoder_threads,
                ) {
                    Ok(decoded) => decoded,
                    Err(error) => {
                        candidate.status =
                            CandidateStatus::Failed(format!("decode failed: {error}"));
                        candidate.influence = Some(CandidateInfluence::Failed);
                        candidate.process_failure = process_failure_summary(&error);
                        candidate.total_ms = Some(candidate_start.elapsed().as_millis() as u64);
                        return candidate;
                    }
                };
                candidate.decode_ms = Some(decode_start.elapsed().as_millis() as u64);
                candidate.decode_process_ms = decoded.process_ms;
                candidate.decode_io_ms = Some(decoded.io_ms);
                let decoded = decoded.image;
                let decoded_bytes = u64::from(decoded.width())
                    .checked_mul(u64::from(decoded.height()))
                    .and_then(|pixels| pixels.checked_mul(4));
                candidate.bytes.decoded_buffer_bytes = decoded_bytes;
                candidate.bytes.peak_estimated_memory_bytes = decoded_bytes.and_then(|decoded| {
                    decoded.checked_add(
                        u64::from(original.width())
                            .checked_mul(u64::from(original.height()))?
                            .checked_mul(4)?,
                    )
                });

                let quality = match self.score_decoded_candidate(
                    ScoreDecodedRequest {
                        original,
                        prepared_original,
                        preprocess_ms,
                        gpu_preprocess_fallback_reason,
                        decoded: &decoded,
                        has_alpha,
                        external_metric_original,
                        candidate_dir,
                        candidate_index,
                    },
                    &mut candidate,
                ) {
                    Ok(score) => score,
                    Err(error) => {
                        candidate.status = CandidateStatus::Failed(format!("{error:#}"));
                        candidate.influence = Some(CandidateInfluence::Failed);
                        candidate.process_failure = process_failure_summary(&error);
                        candidate.total_ms = Some(candidate_start.elapsed().as_millis() as u64);
                        return candidate;
                    }
                };
                self.store_quality_cache(StoreQualityCacheRequest {
                    cache_source_hash,
                    codec,
                    settings: &settings,
                    has_alpha,
                    size_bytes: size,
                    quality,
                    candidate: &candidate,
                });
                quality
            }
        };
        candidate.quality = Some(quality);
        if policy.passes_quality(quality) {
            candidate.status = CandidateStatus::Accepted;
        } else {
            candidate.status = CandidateStatus::Rejected(format!(
                "quality {:.1} below {:.1}",
                quality.overall, self.min_quality
            ));
            candidate.influence = Some(CandidateInfluence::Rejected);
        }
        candidate.total_ms = Some(candidate_start.elapsed().as_millis() as u64);
        candidate
    }

    /// Probes evenly spaced qualities between each codec's highest rejected
    /// and lowest accepted grid point. The probes are independent, so they run
    /// concurrently instead of as a one-at-a-time bisection.
    fn refine_quality_search(
        &self,
        request: QualityRefinementRequest<'_>,
        report: &mut ImageReport,
        best: &mut Option<AcceptedCandidate>,
    ) -> Option<String> {
        for round in 0..self.config.speed.refinement_rounds() {
            let mut points =
                self.plan_refinement(&report.candidates, best.as_ref(), request.policy, round);
            // Next to a point that broke the monotonic assumption the bracket
            // cannot be trusted, so its neighbours are probed too.
            for codec in [Codec::Jpeg, Codec::Webp, Codec::Avif, Codec::Jxl] {
                let Some((qualities, warning)) =
                    neighbour_probes(&report.candidates, codec, best.as_ref(), request.policy)
                else {
                    continue;
                };
                report.warnings.push(warning);
                for quality in qualities {
                    if !points.contains(&(codec, quality)) {
                        points.push((codec, quality));
                    }
                }
            }
            if points.is_empty() {
                break;
            }
            let mut stop_reason = None;
            if let Some(limit) = self.config.max_candidate_attempts {
                let remaining = limit.saturating_sub(attempts(report));
                if points.len() > remaining {
                    points.truncate(remaining);
                    stop_reason = Some(format!("candidate attempt budget of {limit} reached"));
                }
            }
            let total = report.candidates.len() + points.len();
            for chunk in points.chunks(request.parallelism.max(1)) {
                if cancellation_requested() {
                    return Some("cancelled".to_string());
                }
                if let Some(limit) = self.config.max_image_time
                    && request.image_start.elapsed() >= limit
                {
                    return Some(format!("image wall-time budget of {:?} reached", limit));
                }
                let deadline = self
                    .config
                    .max_image_time
                    .map(|limit| request.image_start + limit);
                if let Err(ended) = self.wait_for_temp_room(deadline, request.waited) {
                    return Some(wait_stop_reason(ended, self.config.max_image_time));
                }
                use rayon::prelude::*;
                let first_index = report.candidates.len();
                let current_best = best.as_ref();
                let pool = self.pool().expect("run builds the worker pool");
                let evaluated = pool.install(|| {
                    chunk
                        .par_iter()
                        .enumerate()
                        .map(|(offset, (codec, quality))| {
                            self.score_candidate_report(
                                CandidateEvaluationRequest {
                                    input: request.input,
                                    codec: *codec,
                                    settings: EncoderSettings::Quality {
                                        quality: *quality,
                                        effort: self.config.speed.effort(*codec),
                                    },
                                    encoder_source: request.encoder_source,
                                    original: request.original,
                                    metadata: request.metadata,
                                    prepared_original: request.prepared_original,
                                    preprocess_ms: request.preprocess_ms,
                                    gpu_preprocess_fallback_reason: request
                                        .gpu_preprocess_fallback_reason,
                                    has_alpha: request.has_alpha,
                                    candidate_dir: request.candidate_dir,
                                    policy: request.policy,
                                    cache_source_hash: request.cache_source_hash,
                                    external_metric_original: request.external_metric_original,
                                    candidate_index: first_index + offset,
                                    progress_index: first_index + offset + 1,
                                    progress_total: total,
                                    peers: chunk.len(),
                                    deadline,
                                    waited: request.waited,
                                },
                                current_best,
                            )
                        })
                        .collect::<Vec<_>>()
                });
                for candidate in evaluated.into_iter().flatten() {
                    record_candidate(request.policy, candidate, report, best);
                }
            }
            if stop_reason.is_some() {
                return stop_reason;
            }
        }
        None
    }

    /// Picks the qualities one refinement round should try: for each format
    /// that can still win, a few around where its score should cross the floor.
    fn plan_refinement(
        &self,
        candidates: &[CandidateReport],
        best: Option<&AcceptedCandidate>,
        policy: &DecisionPolicy,
        round: usize,
    ) -> Vec<(Codec, u8)> {
        let offsets = self.config.speed.refinement_offsets(round);
        let floor = policy.min_quality - self.config.quality_slack;
        let mut plans = Vec::new();
        for codec in [Codec::Jpeg, Codec::Webp, Codec::Avif, Codec::Jxl] {
            if !self.config.formats.contains(codec) || !self.backend.tools().available_for(codec) {
                continue;
            }
            let Some(bracket) = quality_bracket(candidates, codec) else {
                continue;
            };
            // File size only grows with quality. If even the rejected lower
            // end is no smaller than the best so far, nothing between the
            // ends can win.
            if let (Some(low_size), Some(current)) = (bracket.low.size, best)
                && !policy.candidate_can_beat(low_size, Some(current), codec)
            {
                continue;
            }
            let crossing = bracket.crossing(floor);
            let inside = |quality: u8| {
                quality > bracket.low.quality
                    && quality < bracket.high.quality
                    && !bracket.seen.contains(&quality)
            };
            let mut qualities = Vec::new();
            if bracket.low.score.is_some() {
                for offset in offsets {
                    let quality = (crossing.round() as i16 + offset).clamp(1, 100) as u8;
                    if inside(quality) && !qualities.contains(&quality) {
                        qualities.push(quality);
                    }
                }
                // The estimate can land on the ends; still split the range.
                let middle = bracket.low.quality + (bracket.high.quality - bracket.low.quality) / 2;
                if qualities.is_empty() && inside(middle) {
                    qualities.push(middle);
                }
            } else {
                // Every coarse quality passed; spread the probes below them.
                let span = u32::from(bracket.high.quality - bracket.low.quality);
                let steps = (offsets.len() as u32 + 1).min(span);
                for step in 1..steps {
                    let quality = bracket.low.quality + (span * step / steps) as u8;
                    if inside(quality) && !qualities.contains(&quality) {
                        qualities.push(quality);
                    }
                }
            }
            if let Some(estimated_size) = bracket.estimated_size(crossing)
                && !qualities.is_empty()
            {
                plans.push(RefinementPlan {
                    codec,
                    qualities,
                    estimated_size,
                });
            }
        }
        // Rank the formats by the size each should reach at the floor and
        // refine only those close enough to the front to still win.
        // The margin is a guess that covers the estimate's error and the gap
        // between probe and final effort. It should be measured per codec if a
        // format that should have won is ever left out.
        const RANKING_MARGIN: f64 = 1.25;
        let front = plans
            .iter()
            .map(|plan| plan.estimated_size)
            .fold(f64::INFINITY, f64::min);
        plans
            .into_iter()
            .filter(|plan| plan.estimated_size <= front * RANKING_MARGIN)
            .flat_map(|plan| {
                plan.qualities
                    .into_iter()
                    .map(move |quality| (plan.codec, quality))
            })
            .collect()
    }

    fn encode_candidate(&self, request: CandidateEncodeRequest<'_>) -> Result<EncodedArtifact> {
        let CandidateEncodeRequest {
            codec,
            settings,
            normalized_path,
            original,
            metadata,
            output_path,
            has_alpha,
            cache_source_hash,
            cache_size_limit,
            threads,
        } = request;
        let mut cache_warnings = Vec::new();
        if let (Some(cache), Some(source_hash)) = (&self.cache, cache_source_hash) {
            match cache.load(source_hash, codec, &settings, has_alpha, output_path) {
                Ok(Some(artifact)) => return Ok(artifact),
                Ok(None) => {}
                Err(error) => cache_warnings.push(format!("cache load skipped: {error:#}")),
            }
        }

        let mut artifact = self.backend.encode(EncodeJob {
            codec,
            settings,
            normalized_png: normalized_path,
            original,
            output_path,
            timeout: self.config.candidate_timeout(),
            has_alpha,
            metadata,
            threads,
        })?;
        artifact.warnings.extend(cache_warnings);

        if let (Some(cache), Some(source_hash)) = (&self.cache, cache_source_hash)
            && cache_size_limit.is_none_or(|limit| artifact.size_bytes < limit)
            && let Err(error) = cache.store(source_hash, &artifact, has_alpha)
        {
            artifact
                .warnings
                .push(format!("cache store skipped: {error:#}"));
        }
        Ok(artifact)
    }

    fn load_quality_cache(
        &self,
        cache_source_hash: Option<&str>,
        codec: Codec,
        settings: &EncoderSettings,
        has_alpha: bool,
        candidate: &mut CandidateReport,
    ) -> Option<QualityScore> {
        if self.config.gpu.verify_gpu {
            return None;
        }
        let (Some(cache), Some(source_hash), Some(artifact)) =
            (&self.cache, cache_source_hash, candidate.path.as_deref())
        else {
            return None;
        };
        match cache.load_metadata(
            source_hash,
            codec,
            settings,
            has_alpha,
            &self.scoring_signature(),
            artifact,
        ) {
            Ok(Some(metadata)) => {
                candidate.decode_ms = metadata.decode_ms;
                candidate.decode_process_ms = metadata.decode_process_ms;
                candidate.decode_io_ms = metadata.decode_io_ms;
                candidate.preprocess_ms = metadata.preprocess_ms;
                candidate.score_ms = metadata.score_ms;
                candidate.score_backend = metadata.score_backend;
                candidate.warnings.push("quality cache hit".to_string());
                Some(metadata.quality)
            }
            Ok(None) => None,
            Err(error) => {
                candidate
                    .warnings
                    .push(format!("quality cache load skipped: {error:#}"));
                None
            }
        }
    }

    fn store_quality_cache(&self, request: StoreQualityCacheRequest<'_>) {
        let StoreQualityCacheRequest {
            cache_source_hash,
            codec,
            settings,
            has_alpha,
            size_bytes,
            quality,
            candidate,
        } = request;
        if self.config.gpu.verify_gpu {
            return;
        }
        if matches!(
            candidate.score_backend,
            Some(ActualScoreBackend::CpuFallback { .. })
        ) {
            return;
        }
        let (Some(cache), Some(source_hash), Some(artifact)) =
            (&self.cache, cache_source_hash, candidate.path.as_deref())
        else {
            return;
        };
        let Ok(artifact_blake3) = file_hash(artifact) else {
            return;
        };
        let metadata = CachedCandidateMetadata {
            size_bytes,
            artifact_blake3,
            decode_ms: candidate.decode_ms,
            decode_process_ms: candidate.decode_process_ms,
            decode_io_ms: candidate.decode_io_ms,
            preprocess_ms: candidate.preprocess_ms,
            score_ms: candidate.score_ms,
            quality,
            encoded_backend: candidate.backend.clone(),
            score_backend: candidate.score_backend.clone(),
        };
        let _ = cache.store_metadata(
            source_hash,
            codec,
            settings,
            has_alpha,
            &self.scoring_signature(),
            &metadata,
        );
    }

    fn score_decoded_candidate(
        &self,
        request: ScoreDecodedRequest<'_>,
        candidate: &mut CandidateReport,
    ) -> Result<QualityScore> {
        let ScoreDecodedRequest {
            original,
            prepared_original,
            preprocess_ms,
            gpu_preprocess_fallback_reason,
            decoded,
            has_alpha,
            external_metric_original,
            candidate_dir,
            candidate_index,
        } = request;
        candidate.preprocess_ms = preprocess_ms;
        let score_start = Instant::now();
        let result = (|| -> Result<QualityScore> {
            let mut quality = if let Some(reason) = gpu_preprocess_fallback_reason {
                candidate.score_backend = Some(ActualScoreBackend::CpuFallback {
                    reason: reason.to_string(),
                });
                self.on_metric_pool(self.metric_threads, || {
                    CpuRayonBackend.score_original(original, decoded, has_alpha)
                })?
                .with_context(|| format!("quality metric failed after {reason}"))?
            } else {
                match (
                    prepared_original,
                    self.should_use_gpu_metrics(original)
                        .then(|| self.compute.as_ref().as_gpu())
                        .flatten(),
                ) {
                    (Some(prepared_original), Some(gpu)) => {
                        // The unit is back before any CPU fallback asks for its own.
                        let scored = self.on_metric_pool(1, || {
                            gpu.score_candidate_prepared_with_timings(
                                prepared_original,
                                decoded,
                                has_alpha,
                            )
                        })?;
                        match scored {
                            Ok((score, timings)) => {
                                candidate.score_backend = Some(ActualScoreBackend::Gpu);
                                candidate.gpu_upload_ms = timings.upload_ms;
                                candidate.gpu_transform_ms = timings.transform_ms;
                                candidate.gpu_compute_ms = timings.compute_ms;
                                candidate.gpu_readback_ms = timings.readback_ms;
                                candidate.gpu_timestamp_ns = timings.timestamp_ns;
                                score
                            }
                            Err(error) => {
                                if self.config.gpu.required {
                                    return Err(anyhow!(
                                        "GPU is required but prepared GPU scoring failed: {error:#}"
                                    ));
                                }
                                let reason = format!("prepared GPU scoring failed: {error:#}");
                                candidate
                                    .warnings
                                    .push(format!("CPU scoring fallback: {reason}"));
                                candidate.score_backend = Some(ActualScoreBackend::CpuFallback {
                                    reason: reason.clone(),
                                });
                                self.on_metric_pool(self.metric_threads, || {
                                    CpuRayonBackend.score_original(original, decoded, has_alpha)
                                })?
                                .with_context(|| format!("quality metric failed after {reason}"))?
                            }
                        }
                    }
                    _ => {
                        if let Some(gpu) = self
                            .should_use_gpu_metrics(original)
                            .then(|| self.compute.as_ref().as_gpu())
                            .flatten()
                        {
                            let outcome = self
                                .on_metric_pool(1, || {
                                    gpu.score_original_with_fallback(original, decoded, has_alpha)
                                })?
                                .context("quality metric failed")?;
                            candidate.gpu_compute_ms = outcome.timings.compute_ms;
                            candidate.gpu_upload_ms = outcome.timings.upload_ms;
                            candidate.gpu_transform_ms = outcome.timings.transform_ms;
                            candidate.gpu_readback_ms = outcome.timings.readback_ms;
                            candidate.gpu_timestamp_ns = outcome.timings.timestamp_ns;
                            if self.config.gpu.required && outcome.fallback_reason.is_some() {
                                return Err(anyhow!(
                                    "GPU is required but scoring fell back to CPU: {}",
                                    outcome
                                        .fallback_reason
                                        .as_deref()
                                        .unwrap_or("unknown reason")
                                ));
                            }
                            candidate.score_backend = Some(match outcome.fallback_reason {
                                Some(reason) => {
                                    candidate
                                        .warnings
                                        .push(format!("CPU scoring fallback: {reason}"));
                                    ActualScoreBackend::CpuFallback { reason }
                                }
                                None => ActualScoreBackend::Gpu,
                            });
                            outcome.quality
                        } else {
                            candidate.score_backend = Some(ActualScoreBackend::Cpu);
                            self.on_metric_pool(self.metric_threads, || {
                                CpuRayonBackend.score_original(original, decoded, has_alpha)
                            })?
                            .context("quality metric failed")?
                        }
                    }
                }
            };
            if has_alpha {
                // Same CPU check for every backend, so it cannot split their verdicts.
                let edge = self.on_metric_pool(self.metric_threads, || {
                    alpha_edge_similarity(original.full_rgba(), &decoded.to_rgba8())
                })?;
                quality.alpha_edge = Some(edge);
                quality.overall = quality.overall.min(edge);
            }
            if self.config.gpu.verify_gpu && self.should_use_gpu_metrics(original) {
                match self.on_metric_pool(self.metric_threads, || {
                    CpuRayonBackend.score_original(original, decoded, has_alpha)
                })? {
                    Ok(cpu_score) => {
                        const TOLERANCE: f32 = 0.25;
                        let component_deltas = BTreeMap::from([
                            ("rgb".to_string(), (cpu_score.rgb - quality.rgb).abs()),
                            ("edge".to_string(), (cpu_score.edge - quality.edge).abs()),
                            (
                                "alpha".to_string(),
                                optional_delta(cpu_score.alpha, quality.alpha),
                            ),
                            (
                                "composite".to_string(),
                                optional_delta(cpu_score.composite_min, quality.composite_min),
                            ),
                        ]);
                        let max_delta = component_deltas.values().copied().fold(0.0_f32, f32::max);
                        let passed = max_delta <= TOLERANCE;
                        candidate.gpu_verification = Some(GpuVerificationReport {
                            tolerance: TOLERANCE,
                            cpu_overall: cpu_score.overall,
                            gpu_overall: quality.overall,
                            overall_delta: (cpu_score.overall - quality.overall).abs(),
                            component_deltas,
                            passed,
                        });
                        candidate.warnings.push(format!(
                            "verify-gpu passed={passed} max_component_delta={max_delta:.3}"
                        ));
                    }
                    Err(error) => candidate
                        .warnings
                        .push(format!("verify-gpu CPU reference failed: {error}")),
                }
            }
            if let Some(original_path) = external_metric_original {
                let external_metric_start = Instant::now();
                // SSIMULACRA2 needs at least an 8x8 image; smaller ones keep
                // the built-in score.
                let run_ssimulacra2 = self.scores_ssimulacra2(decoded.width(), decoded.height());
                // Both tools read the same pixels, so one PNG serves them. It is
                // deleted once scored, whatever the outcome.
                let candidate_png =
                    candidate_dir.join(format!("metric-candidate-{candidate_index}.png"));
                let _scratch = RemoveOnDrop(&candidate_png);
                if run_ssimulacra2 || self.config.metrics.butteraugli {
                    write_metric_png(decoded, &candidate_png)?;
                }
                if run_ssimulacra2 {
                    let score = self.score_ssimulacra2(
                        original_path,
                        &candidate_png,
                        u64::from(decoded.width()) * u64::from(decoded.height()),
                    )?;
                    quality.ssimulacra2 = Some(score);
                    // The perceptual score replaces the PSNR-derived one; the
                    // alpha-channel check still guards transparency.
                    quality.overall = score
                        .min(quality.alpha.unwrap_or(100.0))
                        .min(quality.alpha_edge.unwrap_or(100.0));
                }
                if self.config.metrics.butteraugli {
                    let distance = self.score_butteraugli(original_path, &candidate_png)?;
                    quality.butteraugli = Some(distance);
                    quality.overall = quality.overall.min(butteraugli_quality(distance));
                }
                candidate.external_metric_ms =
                    Some(external_metric_start.elapsed().as_millis() as u64);
            }
            Ok(quality)
        })();
        candidate.score_ms = Some(score_start.elapsed().as_millis() as u64);
        if matches!(
            candidate.score_backend,
            Some(ActualScoreBackend::Gpu | ActualScoreBackend::CpuFallback { .. })
        ) {
            candidate.bytes.source_upload_bytes = u64::from(original.width())
                .checked_mul(u64::from(original.height()))
                .and_then(|pixels| pixels.checked_mul(4));
            candidate.bytes.candidate_upload_bytes = u64::from(decoded.width())
                .checked_mul(u64::from(decoded.height()))
                .and_then(|pixels| pixels.checked_mul(4));
        }
        result
    }

    /// Runs a leaf of metric work, which may use Rayon, on the metric threads
    /// while holding `units` of the work budget.
    ///
    /// The leaf is handed to a thread of its own and the caller joins it. A
    /// candidate runs on a worker of the candidate pool, and a Rayon worker that
    /// waits inside `install` or a parallel iterator runs other jobs of its pool
    /// meanwhile: other candidates, which would wait for the units held here by
    /// the very thread they are stacked on.
    fn on_metric_pool<R: Send>(&self, units: usize, leaf: impl FnOnce() -> R + Send) -> Result<R> {
        let _units = work(units)?;
        Ok(std::thread::scope(|scope| {
            scope
                .spawn(|| self.metric_pool.install(leaf))
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        }))
    }

    fn scoring_signature(&self) -> String {
        format!(
            "{SCORING_VERSION};backend={};ssimulacra2={};butteraugli={};gpu-min-pixels={};gpu-preprocess={}",
            self.compute.backend_id(),
            self.use_ssimulacra2,
            self.config.metrics.butteraugli,
            self.config.gpu.min_pixels_for_gpu_metrics,
            self.config.gpu.prefer_gpu_preprocess
        )
    }

    fn write_winner(&self, input: &Path, candidate: &CandidateReport) -> Result<PathBuf> {
        let source = candidate
            .path
            .as_ref()
            .ok_or_else(|| anyhow!("winner does not have a candidate path"))?;
        let destination = self.output_path(input, candidate.codec)?;
        if self.config.protect_inputs && destination.exists() {
            if let (Ok(existing), Ok(source_input)) =
                (fs::canonicalize(&destination), fs::canonicalize(input))
                && existing == source_input
            {
                anyhow::bail!(
                    "output {} would replace the original; choose another output location or pass --protect-inputs=false",
                    destination.display()
                );
            }
            // A file that is already there, in a folder with images to optimize,
            // may be an original of the user's that is not part of this run.
            if fs::canonicalize(folder_of(&destination)).is_ok_and(|folder| {
                self.input_folders
                    .lock()
                    .expect("input folder lock poisoned")
                    .contains(&folder)
            }) {
                anyhow::bail!(
                    "output {} would replace a file that was already there, in a folder that holds input images; --protect-inputs protects it. Choose another output folder or pass --protect-inputs=false",
                    destination.display()
                );
            }
        }
        atomic_copy(source, &destination, !self.config.overwrite)
            .with_context(|| format!("copying winner to {}", destination.display()))?;
        Ok(destination)
    }

    fn decode_candidate(
        &self,
        codec: Codec,
        path: &Path,
        original: &image::DynamicImage,
        has_alpha: bool,
        threads: usize,
    ) -> Result<DecodedCandidate> {
        // The header decides first: a candidate that cannot be the image that
        // was encoded never reaches an external decoder.
        check_candidate(codec, path, original, has_alpha)?;
        match codec {
            Codec::Avif => {
                let decoded = path.with_extension("decoded.png");
                let tool = self
                    .backend
                    .tools()
                    .avifdec
                    .path
                    .as_ref()
                    .ok_or_else(|| anyhow!("avifdec is not available"))?;
                let process = run_decode_command(
                    tool,
                    &[
                        // The PNG is a lossless temporary read straight back;
                        // storing it uncompressed gives the same pixels and
                        // skips the compression and inflation that made up
                        // most of the decode time.
                        OsString::from("--png-compress"),
                        OsString::from("0"),
                        path.as_os_str().to_os_string(),
                        decoded.as_os_str().to_os_string(),
                    ],
                    self.config.candidate_timeout(),
                    1,
                )?;
                let io_start = Instant::now();
                let _unit = work(1)?;
                let image = decode_image_with_limits(&decoded, self.pipeline_decode_limits());
                // Uncompressed, the file is as large as the pixels. Once they are
                // in memory it must not count against temporary storage for the
                // rest of the search.
                let _ = fs::remove_file(&decoded);
                let image = image.context("opening AVIF decode PNG")?;
                Ok(DecodedCandidate {
                    image,
                    process_ms: Some(process.elapsed_ms),
                    io_ms: io_start.elapsed().as_millis() as u64,
                })
            }
            Codec::Jxl => {
                // PAM is uncompressed and carries alpha and 16-bit samples.
                let decoded = path.with_extension("decoded.pam");
                let tool = self
                    .backend
                    .tools()
                    .djxl
                    .path
                    .as_ref()
                    .ok_or_else(|| anyhow!("djxl not found"))?;
                let process = run_decode_command(
                    tool,
                    &[
                        OsString::from(format!("--num_threads={threads}")),
                        path.as_os_str().to_os_string(),
                        decoded.as_os_str().to_os_string(),
                    ],
                    self.config.candidate_timeout(),
                    threads,
                )?;
                let io_start = Instant::now();
                let _unit = work(1)?;
                let image = read_pam(&decoded, self.pipeline_decode_limits())
                    .context("opening JXL decode output")?;
                // The pixels are in memory; do not let the file count
                // against temporary storage for the rest of the search.
                let _ = fs::remove_file(&decoded);
                Ok(DecodedCandidate {
                    image,
                    process_ms: Some(process.elapsed_ms),
                    io_ms: io_start.elapsed().as_millis() as u64,
                })
            }
            _ => {
                let io_start = Instant::now();
                let _unit = work(1)?;
                let image = decode_image_with_limits(path, self.pipeline_decode_limits())
                    .with_context(|| format!("opening candidate {}", path.display()))?;
                Ok(DecodedCandidate {
                    image,
                    process_ms: None,
                    io_ms: io_start.elapsed().as_millis() as u64,
                })
            }
        }
    }

    fn image_decode_limits(&self) -> ImageDecodeLimits {
        ImageDecodeLimits {
            max_input_bytes: self.config.max_input_bytes,
            max_pixels: self.config.max_pixels,
            max_width: self.config.max_width,
            max_height: self.config.max_height,
            max_decoded_bytes: self.config.max_decoded_bytes,
            max_metadata_bytes: self.config.max_metadata_bytes,
            max_bits_per_pixel: self.config.max_bits_per_pixel,
        }
    }

    /// Limits for files the pipeline wrote itself, such as a decoded candidate.
    /// The limit on input bytes is about the user's files, not these.
    fn pipeline_decode_limits(&self) -> ImageDecodeLimits {
        ImageDecodeLimits {
            max_input_bytes: u64::MAX,
            ..self.image_decode_limits()
        }
    }

    /// Whether ssimulacra2 scores an image of this size: it needs 8 px a side.
    fn scores_ssimulacra2(&self, width: u32, height: u32) -> bool {
        self.use_ssimulacra2 && width >= 8 && height >= 8
    }

    fn decode_input(&self, input: &Path) -> Result<DecodedSource> {
        if input
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("avif"))
        {
            let source_size = fs::metadata(input)
                .with_context(|| format!("reading AVIF source metadata {}", input.display()))?
                .len();
            anyhow::ensure!(
                source_size <= self.config.max_input_bytes,
                "input is {source_size} bytes, exceeding the configured {} byte limit",
                self.config.max_input_bytes
            );
            let header = read_prefix(input)
                .with_context(|| format!("reading AVIF header {}", input.display()))?;
            let probe = probe_avif(&header);
            anyhow::ensure!(
                !probe.as_ref().is_some_and(|probe| probe.animated),
                "animated AVIF is not supported; only still images are optimized"
            );
            // The header's size is checked before the decoder tool allocates it.
            let limits = self.image_decode_limits();
            if let Some((width, height)) = probe.as_ref().and_then(|probe| probe.dimensions) {
                validate_decoder_dimensions(width, height, limits)?;
                // At most 8 bits a sample unless the header says more, four samples a pixel.
                let wide = probe
                    .and_then(|probe| probe.bit_depth)
                    .is_some_and(|bits| bits > 8);
                let decoded_bytes = u64::from(width) * u64::from(height) * if wide { 8 } else { 4 };
                anyhow::ensure!(
                    decoded_bytes <= limits.max_decoded_bytes,
                    "decoded image requires {decoded_bytes} bytes, exceeding the configured {} byte limit",
                    limits.max_decoded_bytes
                );
            }
            let tool = self
                .backend
                .tools()
                .avifdec
                .path
                .as_ref()
                .ok_or_else(|| anyhow!("avifdec is not available"))?;
            let temp = tempfile::tempdir().context("creating AVIF decode directory")?;
            let decoded_path = temp.path().join("source.png");
            run_process(
                tool,
                &[
                    OsString::from("--png-compress"),
                    OsString::from("1"),
                    // The tool refuses what the configured limits refuse, not only its own.
                    OsString::from("--size-limit"),
                    OsString::from(self.config.max_pixels.min(u64::from(u32::MAX)).to_string()),
                    OsString::from("--dimension-limit"),
                    OsString::from(
                        self.config
                            .max_width
                            .max(self.config.max_height)
                            .to_string(),
                    ),
                    std::path::absolute(input)?.into_os_string(),
                    decoded_path.as_os_str().to_os_string(),
                ],
                self.config.candidate_timeout(),
            )?;
            let _unit = work(1)?;
            let mut decoded =
                decode_source_with_limits(&decoded_path, self.pipeline_decode_limits())
                    .context("decoding AVIF source through avifdec")?;
            decoded.metadata.format = Some("Avif".to_string());
            Ok(decoded)
        } else {
            let _unit = work(1)?;
            decode_source_with_limits(input, self.image_decode_limits())
        }
    }

    fn output_path(&self, input: &Path, codec: Codec) -> Result<PathBuf> {
        if let Some(out) = &self.config.out {
            // The extension names the format, and only that format was tried; a
            // name without a known image extension gets the winner's added.
            if self.out_codec.is_some() {
                return Ok(out.clone());
            }
            let mut named = out.clone().into_os_string();
            named.push(".");
            named.push(codec.extension());
            return Ok(PathBuf::from(named));
        }
        let file_stem = input
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| anyhow!("input has no valid file stem: {}", input.display()))?;
        let ambiguous = self
            .ambiguous_outputs
            .lock()
            .expect("output name lock poisoned")
            .contains(input);
        let filename = if ambiguous {
            let tag = blake3::hash(input.to_string_lossy().as_bytes()).to_hex();
            format!("{file_stem}-{}.{}", &tag[..8], codec.extension())
        } else {
            format!("{file_stem}.{}", codec.extension())
        };
        Ok(self.output_dir(input).join(filename))
    }

    /// Returns the replacement image when the alpha policy changes the pixels.
    fn apply_alpha_policy(
        &self,
        image: &image::DynamicImage,
    ) -> Result<Option<image::DynamicImage>> {
        if !image.color().has_alpha() {
            return Ok(None);
        }
        let deep = matches!(
            image.color(),
            image::ColorType::Rgba16 | image::ColorType::La16
        );
        match &self.config.alpha {
            AlphaPolicy::Preserve => Ok(None),
            AlphaPolicy::Auto => {
                let opaque = image.to_rgba16().pixels().all(|pixel| pixel[3] == u16::MAX);
                Ok(opaque.then(|| {
                    if deep {
                        image::DynamicImage::ImageRgb16(image.to_rgb16())
                    } else {
                        image::DynamicImage::ImageRgb8(image.to_rgb8())
                    }
                }))
            }
            AlphaPolicy::Flatten { background } => {
                let background = parse_hex_color(background)?;
                let blend = |channel: u64, background: u64, alpha: u64, max: u64| {
                    (channel * alpha + background * (max - alpha) + max / 2) / max
                };
                Ok(Some(if deep {
                    let mut flat = image.to_rgb16();
                    for (out, source) in flat.pixels_mut().zip(image.to_rgba16().pixels()) {
                        for channel in 0..3 {
                            out[channel] = blend(
                                u64::from(source[channel]),
                                u64::from(background[channel]) * 257,
                                u64::from(source[3]),
                                65_535,
                            ) as u16;
                        }
                    }
                    image::DynamicImage::ImageRgb16(flat)
                } else {
                    let mut flat = image.to_rgb8();
                    for (out, source) in flat.pixels_mut().zip(image.to_rgba8().pixels()) {
                        for channel in 0..3 {
                            out[channel] = blend(
                                u64::from(source[channel]),
                                u64::from(background[channel]),
                                u64::from(source[3]),
                                255,
                            ) as u8;
                        }
                    }
                    image::DynamicImage::ImageRgb8(flat)
                }))
            }
        }
    }

    fn score_ssimulacra2(
        &self,
        original_path: &Path,
        candidate_path: &Path,
        pixels: u64,
    ) -> Result<f32> {
        let tool = self
            .backend
            .tools()
            .ssimulacra2
            .path
            .as_ref()
            .ok_or_else(|| anyhow!("ssimulacra2 not found"))?;
        let timeout = self.config.candidate_timeout();
        let _reserved = self.metric_memory.acquire(
            memory::external_metric_bytes(pixels.min(FULL_SIZE_METRIC_PIXELS)),
            Some(Instant::now() + timeout),
        )?;
        let output = run_process(
            tool,
            &[
                original_path.as_os_str().to_os_string(),
                candidate_path.as_os_str().to_os_string(),
            ],
            timeout,
        )?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow!("SSIMULACRA2 produced no score"))?
            .parse::<f32>()
            .context("parsing SSIMULACRA2 score")
    }

    fn score_butteraugli(&self, original_path: &Path, candidate_path: &Path) -> Result<f32> {
        let tool = self
            .backend
            .tools()
            .butteraugli
            .path
            .as_ref()
            .ok_or_else(|| anyhow!("butteraugli_main not found"))?;
        let output = run_process(
            tool,
            &[
                original_path.as_os_str().to_os_string(),
                candidate_path.as_os_str().to_os_string(),
            ],
            self.config.candidate_timeout(),
        )?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow!("Butteraugli produced no score"))?
            .parse::<f32>()
            .context("parsing Butteraugli score")
    }
}

fn stable_config_text(config: &CliConfig) -> String {
    format!(
        "preset={:?};target={:?};formats={:?};report={:?};min_quality={:?};quality_slack={};metadata={:?};alpha={:?};speed={:?};candidate_timeout={:?};max_image_time={:?};max_candidate_attempts={:?};max_temp_bytes={:?};max_run_temp_bytes={:?};max_memory_bytes={:?};workers={};parallelism={:?};allow_larger={};dry_run={};out_dir={:?};out={:?};overwrite={};protect_inputs={};max_input_bytes={};max_pixels={};max_width={};max_height={};max_decoded_bytes={};max_metadata_bytes={};max_bits_per_pixel={};gpu={:?};cache={:?};metrics={:?};verbose={};inputs={:?}",
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

fn butteraugli_quality(distance: f32) -> f32 {
    (100.0 - distance * 10.0).clamp(0.0, 100.0)
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "webp" | "avif"
            )
        })
        .unwrap_or(false)
}

fn external_tools_accept_input(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png"
            )
        })
        .unwrap_or(false)
}

fn metadata_payload(metadata: &SourceMetadata, policy: MetadataPolicy) -> MetadataPayload {
    match policy {
        // A profile that is not plain sRGB says what the pixel values mean, so
        // it is colour data: dropping it would show different colours.
        MetadataPolicy::Strip => MetadataPayload {
            icc_profile: metadata
                .icc_profile
                .clone()
                .filter(|profile| !icc_is_plain_srgb(profile)),
            exif_metadata: None,
        },
        MetadataPolicy::Safe => MetadataPayload {
            icc_profile: metadata.icc_profile.clone(),
            exif_metadata: None,
        },
        MetadataPolicy::Keep => MetadataPayload {
            icc_profile: metadata.icc_profile.clone(),
            exif_metadata: metadata.sanitized_exif_metadata(),
        },
    }
}

fn parse_hex_color(value: &str) -> Result<[u8; 3]> {
    let hex = value.trim().trim_start_matches('#');
    anyhow::ensure!(
        hex.len() == 6 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "background colour '{value}' is not in #rrggbb form"
    );
    let channel = |index: usize| u8::from_str_radix(&hex[index..index + 2], 16);
    Ok([channel(0)?, channel(2)?, channel(4)?])
}

fn candidate_path(dir: &Path, codec: Codec, settings: &EncoderSettings) -> PathBuf {
    let label = settings.label().replace(['/', ':'], "-");
    dir.join(format!("{codec:?}-{label}.{}", codec.extension()).to_ascii_lowercase())
}

/// How a candidate that the encoder did not produce is recorded: one it
/// declined to make has no verdict and is merely skipped; any other error is
/// a failure.
fn encode_failure(error: &anyhow::Error) -> (CandidateStatus, CandidateInfluence) {
    let declined = error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<EncodeError>(),
            Some(EncodeError::Declined(_))
        )
    });
    let message = format!("{error:#}");
    if declined {
        (
            CandidateStatus::Skipped(message),
            CandidateInfluence::Skipped,
        )
    } else {
        (CandidateStatus::Failed(message), CandidateInfluence::Failed)
    }
}

/// Deletes a scratch file when dropped.
struct RemoveOnDrop<'a>(&'a Path);

impl Drop for RemoveOnDrop<'_> {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.0);
    }
}

fn write_metric_png(image: &image::DynamicImage, path: &Path) -> Result<()> {
    let _unit = work(1)?;
    let metric = bounded_metric_image(image);
    metric
        .save_with_format(path, image::ImageFormat::Png)
        .with_context(|| format!("writing metric PNG {}", path.display()))
}

/// Image handed to the external metrics. Scoring a reduced copy of a large
/// image hides compression damage that is plain at full size, so large images
/// are scored on a mosaic of full-resolution samples from a regular grid.
// A fixed 3x3 grid of 1024 px tiles. Sampling by detail, or more tiles, would
// help if large images with small busy regions are scored too kindly.
fn bounded_metric_image(image: &image::DynamicImage) -> Cow<'_, image::DynamicImage> {
    const GRID: u32 = 3;
    const TILE: u32 = 1024;
    let (width, height) = image.dimensions();
    if width as u64 * height as u64 <= FULL_SIZE_METRIC_PIXELS {
        return Cow::Borrowed(image);
    }
    let (cell_width, cell_height) = (width / GRID, height / GRID);
    let (tile_width, tile_height) = (TILE.min(cell_width), TILE.min(cell_height));
    let mut mosaic = image::RgbaImage::new(tile_width * GRID, tile_height * GRID);
    for row in 0..GRID {
        for column in 0..GRID {
            // The tile sits in the middle of its grid cell.
            let tile = image
                .crop_imm(
                    cell_width * column + (cell_width - tile_width) / 2,
                    cell_height * row + (cell_height - tile_height) / 2,
                    tile_width,
                    tile_height,
                )
                .to_rgba8();
            image::imageops::replace(
                &mut mosaic,
                &tile,
                i64::from(tile_width * column),
                i64::from(tile_height * row),
            );
        }
    }
    Cow::Owned(if image.color().has_alpha() {
        image::DynamicImage::ImageRgba8(mosaic)
    } else {
        image::DynamicImage::ImageRgb8(image::DynamicImage::ImageRgba8(mosaic).to_rgb8())
    })
}

fn run_decode_command(
    executable: &Path,
    args: &[OsString],
    timeout: std::time::Duration,
    threads: usize,
) -> Result<smart_compress_codecs::ProcessOutput> {
    run_process_threads(executable, args, timeout, threads)
}

/// Qualities worth probing for one format, and its estimated size at the floor.
struct RefinementPlan {
    codec: Codec,
    qualities: Vec<u8>,
    estimated_size: f64,
}

/// Where a format's score crosses the floor. The upper end is its lowest
/// quality that did not fail on quality: one that passed, or one that was
/// already too large to be worth scoring. The lower end is the highest quality
/// below that which was scored and failed.
fn quality_bracket(candidates: &[CandidateReport], codec: Codec) -> Option<QualityBracket> {
    let mut seen = HashSet::new();
    let mut upper: Option<BracketEnd> = None;
    let mut rejected = Vec::new();
    for candidate in candidates {
        if candidate.codec != codec {
            continue;
        }
        let EncoderSettings::Quality { quality, .. } = candidate.settings else {
            continue;
        };
        seen.insert(quality);
        let end = BracketEnd {
            quality,
            score: candidate.quality.map(|quality| quality.overall),
            size: candidate.size_bytes,
        };
        match &candidate.status {
            _ if candidate.failed_quality() => {
                rejected.push(end);
            }
            // Passed, or too large to be worth scoring.
            CandidateStatus::Accepted | CandidateStatus::Rejected(_) => {
                if upper.is_none_or(|current| quality < current.quality) {
                    upper = Some(end);
                }
            }
            // No verdict.
            CandidateStatus::Skipped(_) | CandidateStatus::Failed(_) => {}
        }
    }

    let high = upper?;
    if high.quality <= 1 {
        return None;
    }
    rejected.retain(|end| end.quality < high.quality);
    rejected.sort_by_key(|end| std::cmp::Reverse(end.quality));
    let low = rejected.first().copied().unwrap_or(BracketEnd {
        quality: 1,
        score: None,
        size: None,
    });
    Some(QualityBracket {
        low,
        high,
        below: rejected.get(1).copied(),
        seen,
    })
}

/// Candidates that were encoded or tried to be; placeholders for qualities the
/// search chose not to evaluate do not use up the attempt budget.
fn attempts(report: &ImageReport) -> usize {
    report
        .candidates
        .iter()
        .filter(|candidate| candidate.total_ms.is_some())
        .count()
}

/// A coarse plan split into the candidates to run first and the ones held back.
struct StagedPlan {
    first: Vec<(Codec, EncoderSettings)>,
    held: Vec<(Codec, EncoderSettings)>,
    /// The middle quality of every lossy format in `first`.
    mids: HashMap<Codec, u8>,
}

/// How much larger than the largest file that could still win the lossless WebP
/// probe must be before the other lossless encoders are skipped.
///
/// Another lossless encoder can only get below the ceiling if it beats the
/// probe by more than this factor. Measured on 63 JPEG sources (the six stock
/// photos, 15 re-encodes of three of them at q40 to q100, six corpus JPEGs and
/// 36 corpus graphics saved as JPEG at q60/q90) the probe was at most 1.65x the
/// smallest of the four lossless encoders (1.23x on photographs), and 2.10x on
/// 24 corpus PNG sources. The native PNG encoder and AVIF are cheaper to run
/// but up to 4400x off on graphics, so they cannot gate. On the 22 of those
/// JPEG sources where some lossless encoder does beat the source, the probe was
/// at most 0.95x the source. The gate stops being safe if a source ever has a
/// ratio above 3.0.
const LOSSLESS_GATE_MARGIN: f64 = 3.0;

/// The lossless candidate the others are gated on. WebP lossless tracks the
/// smallest lossless size most closely of the encoders that are cheap enough.
fn is_lossless_probe(codec: Codec, settings: &EncoderSettings) -> bool {
    codec == Codec::Webp && *settings == EncoderSettings::Lossless
}

/// Whether the lossless probe may gate the other lossless encoders for this
/// image. Every lossless candidate is a pixel-exact re-encode (`cjxl` is run
/// with `--lossless_jpeg=0`, so a JPEG is never transcoded), which only a
/// lossless source can beat by much: sources that may be lossless (PNG, WebP,
/// AVIF) run all of them, as does a plan with nothing to gate.
fn lossless_gate_applies(probe: &ImageProbe, plan: &[(Codec, EncoderSettings)]) -> bool {
    probe
        .format
        .as_deref()
        .is_some_and(|format| format.eq_ignore_ascii_case("Jpeg"))
        && plan
            .iter()
            .any(|(codec, settings)| is_lossless_probe(*codec, settings))
        && plan.iter().any(|(codec, settings)| {
            *settings == EncoderSettings::Lossless && *codec != Codec::Webp
        })
}

/// Why a lossless candidate other than the probe is not worth encoding, given
/// the probe's size: no lossless encoder gets under the largest file that could
/// still win if the probe is [`LOSSLESS_GATE_MARGIN`] times over it.
fn lossless_gate(
    candidates: &[CandidateReport],
    codec: Codec,
    settings: &EncoderSettings,
    policy: &DecisionPolicy,
    best: Option<&AcceptedCandidate>,
) -> Option<String> {
    if *settings != EncoderSettings::Lossless || is_lossless_probe(codec, settings) {
        return None;
    }
    let probe = candidates
        .iter()
        .find(|candidate| is_lossless_probe(candidate.codec, &candidate.settings))?
        .size_bytes?;
    let ceiling = policy.size_ceiling(best)?;
    (probe as f64 > LOSSLESS_GATE_MARGIN * ceiling as f64).then(|| {
        format!(
            "cannot win: lossless WebP is {probe} bytes, over {LOSSLESS_GATE_MARGIN}x the {ceiling} bytes a winner can be, which no lossless encoder closes"
        )
    })
}

/// Runs only the middle quality of each lossy format first; qualities of other
/// kinds (lossless, palette) and the rest of each format's grid are held back.
/// When the lossless probe gates the other lossless encoders it runs with the
/// first stage, last, so that the second stage knows its size.
fn stage_plan(plan: Vec<(Codec, EncoderSettings)>, gated: bool) -> StagedPlan {
    let mut grids: HashMap<Codec, Vec<u8>> = HashMap::new();
    for (codec, settings) in &plan {
        if let EncoderSettings::Quality { quality, .. } = settings {
            grids.entry(*codec).or_default().push(*quality);
        }
    }
    let mids: HashMap<Codec, u8> = grids
        .into_iter()
        .map(|(codec, qualities)| (codec, qualities[qualities.len() / 2]))
        .collect();
    let (mut first, mut held): (Vec<_>, Vec<_>) = plan.into_iter().partition(|(codec, settings)| {
        matches!(settings, EncoderSettings::Quality { quality, .. } if mids.get(codec) == Some(quality))
    });
    if gated
        && let Some(at) = held
            .iter()
            .position(|(codec, settings)| is_lossless_probe(*codec, settings))
    {
        first.push(held.remove(at));
    }
    StagedPlan { first, held, mids }
}

/// Why a coarse quality held back at the first stage is not worth running,
/// given how the middle quality of its format came out. Both rest on a lower
/// quality scoring lower and encoding smaller; `held_back_to_run` checks that
/// assumption once more points are known.
fn coarse_followup(
    candidates: &[CandidateReport],
    policy: &DecisionPolicy,
    best: Option<&AcceptedCandidate>,
    codec: Codec,
    mid: u8,
    quality: u8,
) -> Option<String> {
    let middle = candidates.iter().find(|candidate| {
        candidate.codec == codec
            && matches!(candidate.settings, EncoderSettings::Quality { quality, .. } if quality == mid)
    })?;
    let higher = quality > mid;
    if middle.failed_quality() {
        if !higher {
            return Some(format!(
                "assumed to fail: lower than q{mid} of this format, which failed the quality floor"
            ));
        }
        // The refinement skips a format whose failing end is already no smaller
        // than the best; higher qualities are larger still.
        let size = middle.size_bytes?;
        (!policy.candidate_can_beat(size, best, codec)).then(|| {
            format!(
                "cannot win: q{mid} of this format failed the quality floor at {size} bytes, no smaller than the best accepted candidate"
            )
        })
    } else if matches!(
        middle.status,
        CandidateStatus::Accepted | CandidateStatus::Rejected(_)
    ) {
        higher.then(|| {
            format!("assumed to pass: higher than q{mid} of this format, which did not fail the quality floor")
        })
    } else {
        // No verdict for the middle quality, so nothing can be assumed.
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InversionKind {
    /// A lower quality passed while a higher one failed.
    Score,
    /// A lower quality produced a larger file than a higher one.
    Size,
}

/// Two evaluated qualities of one format, at the same effort, that break the
/// assumption that a lower quality scores lower and encodes smaller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Inversion {
    kind: InversionKind,
    lower: u8,
    higher: u8,
    lower_size: Option<u64>,
}

impl Inversion {
    fn describe(&self) -> String {
        let (lower, higher) = (self.lower, self.higher);
        match self.kind {
            InversionKind::Score => format!("q{lower} passes while q{higher} fails"),
            InversionKind::Size => format!("q{lower} encodes larger than q{higher}"),
        }
    }
}

/// Inversions between neighbouring evaluated qualities of a format. Efforts
/// are compared separately: a higher effort legitimately scores and sizes
/// differently at the same quality.
fn inversions(candidates: &[CandidateReport], codec: Codec) -> Vec<Inversion> {
    let mut points = candidates
        .iter()
        .filter(|candidate| candidate.codec == codec)
        .filter_map(|candidate| match candidate.settings {
            EncoderSettings::Quality { quality, effort } => Some((effort, quality, candidate)),
            _ => None,
        })
        .collect::<Vec<_>>();
    points.sort_by_key(|(effort, quality, _)| (*effort, *quality));
    let mut found = Vec::new();
    for pair in points.windows(2) {
        let [(effort, lower, low), (other_effort, higher, high)] = pair else {
            continue;
        };
        if effort != other_effort {
            continue;
        }
        let inversion = |kind| Inversion {
            kind,
            lower: *lower,
            higher: *higher,
            lower_size: low.size_bytes,
        };
        if matches!(low.status, CandidateStatus::Accepted) && high.failed_quality() {
            found.push(inversion(InversionKind::Score));
        }
        if let (Some(low_size), Some(high_size)) = (low.size_bytes, high.size_bytes)
            && low_size > high_size
        {
            found.push(inversion(InversionKind::Size));
        }
    }
    found
}

fn describe_inversions(codec: Codec, inversions: &[Inversion]) -> String {
    let details = inversions
        .iter()
        .map(Inversion::describe)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "non-monotonic behaviour detected for {}: {details}",
        format!("{codec:?}").to_ascii_lowercase()
    )
}

/// Which of the coarse qualities held back (as skipped placeholders) should run
/// after all, as indexes into `held`, with a warning for each format that
/// triggers it: every format whose evaluated points show an inversion, and,
/// when refinement follows, the next quality down for a format whose upper
/// bracket end was never scored, since its crossing is then estimated from
/// two failures.
fn held_back_to_run(
    candidates: &[CandidateReport],
    held: &[CandidateReport],
    refine: bool,
) -> (Vec<usize>, Vec<String>) {
    let mut picked = Vec::new();
    let mut warnings = Vec::new();
    for codec in [Codec::Jpeg, Codec::Webp, Codec::Avif, Codec::Jxl] {
        let own = held
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| match candidate.settings {
                EncoderSettings::Quality { quality, .. } if candidate.codec == codec => {
                    Some((index, quality))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if own.is_empty() {
            continue;
        }
        let found = inversions(candidates, codec);
        if !found.is_empty() {
            warnings.push(format!(
                "{}; evaluating the coarse qualities the staged search had skipped ({})",
                describe_inversions(codec, &found),
                own.iter()
                    .map(|(_, quality)| format!("q{quality}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            picked.extend(own.iter().map(|(index, _)| *index));
        } else if refine
            && let Some(bracket) = quality_bracket(candidates, codec)
            && bracket.high.score.is_none()
            && bracket.low.score.is_some()
            && bracket.below.is_none()
            && let Some((index, _)) = own
                .iter()
                .filter(|(_, quality)| *quality < bracket.low.quality)
                .max_by_key(|(_, quality)| *quality)
        {
            picked.push(*index);
        }
    }
    picked.sort_unstable();
    (picked, warnings)
}

/// Untried qualities next to the lower point of every inversion among a
/// format's evaluated results, with the warning that explains them. Formats
/// that cannot beat the best so far are left alone.
fn neighbour_probes(
    candidates: &[CandidateReport],
    codec: Codec,
    best: Option<&AcceptedCandidate>,
    policy: &DecisionPolicy,
) -> Option<(Vec<u8>, String)> {
    let found = inversions(candidates, codec)
        .into_iter()
        .filter(|inversion| {
            inversion
                .lower_size
                .is_none_or(|size| policy.candidate_can_beat(size, best, codec))
        })
        .collect::<Vec<_>>();
    let seen = candidates
        .iter()
        .filter(|candidate| candidate.codec == codec)
        .filter_map(|candidate| match candidate.settings {
            EncoderSettings::Quality { quality, .. } => Some(quality),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let mut qualities = found
        .iter()
        .flat_map(|inversion| {
            [
                inversion.lower.saturating_sub(1).max(1),
                inversion.lower + 1,
            ]
        })
        .filter(|quality| *quality <= 100 && !seen.contains(quality))
        .collect::<Vec<_>>();
    qualities.sort_unstable();
    qualities.dedup();
    if qualities.is_empty() {
        return None;
    }
    let list = qualities
        .iter()
        .map(|quality| format!("q{quality}"))
        .collect::<Vec<_>>()
        .join(", ");
    Some((
        qualities,
        format!(
            "{}; probing the neighbouring qualities {list}",
            describe_inversions(codec, &found)
        ),
    ))
}

#[derive(Clone, Copy)]
struct BracketEnd {
    quality: u8,
    score: Option<f32>,
    size: Option<u64>,
}

struct QualityBracket {
    low: BracketEnd,
    high: BracketEnd,
    /// The next scored failure under `low`, for when `high` has no score.
    below: Option<BracketEnd>,
    seen: HashSet<u8>,
}

impl QualityBracket {
    /// Quality at which the score should reach `floor`, assuming it rises
    /// linearly: between the two ends when both were scored, otherwise along
    /// the slope of the two failures below. With nothing to go on, the middle.
    fn crossing(&self, floor: f32) -> f32 {
        let (low, high) = (f32::from(self.low.quality), f32::from(self.high.quality));
        let slope_from = |from: BracketEnd, to: BracketEnd| {
            let rise = to.score? - from.score?;
            let run = f32::from(to.quality) - f32::from(from.quality);
            (rise > 0.0 && run > 0.0).then_some(rise / run)
        };
        let slope = slope_from(self.low, self.high)
            .or_else(|| self.below.and_then(|below| slope_from(below, self.low)));
        match (self.low.score, slope) {
            (Some(below_floor), Some(slope)) => {
                (low + (floor - below_floor) / slope).clamp(low, high)
            }
            _ => (low + high) / 2.0,
        }
    }

    /// File size expected at `quality`, interpolating the logarithm of the
    /// size between the two ends (encoded size is roughly exponential in the
    /// quality setting).
    fn estimated_size(&self, quality: f32) -> Option<f64> {
        let high = (self.high.size? as f64).max(1.0).ln();
        let Some(low) = self.low.size else {
            return Some(high.exp());
        };
        let low = (low as f64).max(1.0).ln();
        let span = f32::from(self.high.quality) - f32::from(self.low.quality);
        let position = f64::from((quality - f32::from(self.low.quality)) / span.max(1.0));
        Some((low + (high - low) * position).exp())
    }
}

struct DecodedCandidate {
    image: image::DynamicImage,
    process_ms: Option<u64>,
    io_ms: u64,
}

struct CandidateEncodeRequest<'a> {
    codec: Codec,
    settings: EncoderSettings,
    normalized_path: &'a Path,
    original: &'a image::DynamicImage,
    metadata: &'a MetadataPayload,
    output_path: &'a Path,
    has_alpha: bool,
    cache_source_hash: Option<&'a str>,
    cache_size_limit: Option<u64>,
    threads: usize,
}

struct CandidateEvaluationRequest<'a> {
    input: &'a Path,
    codec: Codec,
    settings: EncoderSettings,
    encoder_source: &'a Path,
    original: &'a OriginalImage,
    metadata: &'a MetadataPayload,
    prepared_original: Option<&'a GpuPreparedOriginal>,
    preprocess_ms: Option<u64>,
    gpu_preprocess_fallback_reason: Option<&'a str>,
    has_alpha: bool,
    candidate_dir: &'a Path,
    policy: &'a DecisionPolicy,
    cache_source_hash: Option<&'a str>,
    external_metric_original: Option<&'a Path>,
    candidate_index: usize,
    progress_index: usize,
    progress_total: usize,
    /// Candidates of the image that are evaluated together with this one.
    peers: usize,
    /// When the image runs out of time; ends the wait for candidate memory.
    deadline: Option<Instant>,
    /// Milliseconds the image has spent waiting on memory and temporary room.
    waited: &'a AtomicU64,
}

struct ActiveCandidate<'a>(&'a AtomicUsize, &'a Budget);

impl Drop for ActiveCandidate<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
        self.1.wake();
    }
}

/// Appends an evaluated candidate and promotes it when it beats the current best.
fn record_candidate(
    policy: &DecisionPolicy,
    mut candidate: CandidateReport,
    report: &mut ImageReport,
    best: &mut Option<AcceptedCandidate>,
) {
    if matches!(candidate.status, CandidateStatus::Accepted)
        && let (Some(size_bytes), Some(quality)) = (candidate.size_bytes, candidate.quality)
        && policy.candidate_can_beat(size_bytes, best.as_ref(), candidate.codec)
    {
        candidate.influence = Some(if best.is_some() {
            CandidateInfluence::ImprovedWinner
        } else {
            CandidateInfluence::FirstPassing
        });
        *best = Some(AcceptedCandidate {
            codec: candidate.codec,
            size_bytes,
            quality,
        });
    }
    report.candidates.push(candidate);
}

struct QualityRefinementRequest<'a> {
    input: &'a Path,
    parallelism: usize,
    image_start: Instant,
    waited: &'a AtomicU64,
    encoder_source: &'a Path,
    original: &'a OriginalImage,
    metadata: &'a MetadataPayload,
    prepared_original: Option<&'a GpuPreparedOriginal>,
    preprocess_ms: Option<u64>,
    gpu_preprocess_fallback_reason: Option<&'a str>,
    has_alpha: bool,
    candidate_dir: &'a Path,
    policy: &'a DecisionPolicy,
    cache_source_hash: Option<&'a str>,
    external_metric_original: Option<&'a Path>,
}

struct StoreQualityCacheRequest<'a> {
    cache_source_hash: Option<&'a str>,
    codec: Codec,
    settings: &'a EncoderSettings,
    has_alpha: bool,
    size_bytes: u64,
    quality: QualityScore,
    candidate: &'a CandidateReport,
}

struct ScoreDecodedRequest<'a> {
    original: &'a OriginalImage,
    prepared_original: Option<&'a GpuPreparedOriginal>,
    preprocess_ms: Option<u64>,
    gpu_preprocess_fallback_reason: Option<&'a str>,
    decoded: &'a image::DynamicImage,
    has_alpha: bool,
    external_metric_original: Option<&'a Path>,
    candidate_dir: &'a Path,
    candidate_index: usize,
}

struct FileProgressGuard {
    progress: Arc<dyn ProgressObserver>,
    path: PathBuf,
}

impl FileProgressGuard {
    fn new(progress: Arc<dyn ProgressObserver>, path: &Path) -> Self {
        progress.file_started(path);
        Self {
            progress,
            path: path.to_path_buf(),
        }
    }
}

impl Drop for FileProgressGuard {
    fn drop(&mut self) {
        self.progress.file_finished(&self.path);
    }
}

struct CandidateProgressGuard {
    progress: Arc<dyn ProgressObserver>,
    path: PathBuf,
    index: usize,
    total: usize,
}

impl CandidateProgressGuard {
    fn new(
        progress: Arc<dyn ProgressObserver>,
        path: &Path,
        index: usize,
        total: usize,
        codec: Codec,
        settings: &EncoderSettings,
    ) -> Self {
        progress.candidate_started(path, index, total, codec, settings);
        Self {
            progress,
            path: path.to_path_buf(),
            index,
            total,
        }
    }
}

impl Drop for CandidateProgressGuard {
    fn drop(&mut self) {
        self.progress
            .candidate_finished(&self.path, self.index, self.total);
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};
    use smart_compress_codecs::{Codec, EncoderSettings};

    use super::*;
    use crate::config::{CacheConfig, CodecSelection, Preset, ReportFormat, TargetPolicy};
    use crate::scoring::gpu_or_skip;

    #[test]
    fn dry_run_reports_probe_without_candidates() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 16, Rgb([120, 40, 10])));
        image
            .save_with_format(&input, image::ImageFormat::Png)
            .unwrap();

        let config = CliConfig {
            inputs: vec![input],
            dry_run: true,
            report: ReportFormat::Json,
            ..CliConfig::default()
        };
        let summary = Optimizer::new(config).run().unwrap();
        assert_eq!(summary.report.failures, 0);
        assert!(summary.report.reports[0].probe.is_some());
        assert!(summary.report.reports[0].candidates.is_empty());
    }

    #[test]
    fn native_png_pipeline_writes_atomic_output() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        let out_dir = dir.path().join("out");
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([10, 20, 30])));
        image
            .save_with_format(&input, image::ImageFormat::Png)
            .unwrap();

        let config = CliConfig {
            inputs: vec![input],
            out_dir: Some(out_dir.clone()),
            preset: Preset::Balanced,
            target: TargetPolicy::MaxCompat,
            formats: CodecSelection::new([Codec::Png]),
            allow_larger: true,
            max_pixels: 100_000_000,
            ..CliConfig::default()
        };
        let summary = Optimizer::new(config).run().unwrap();
        assert_eq!(summary.report.failures, 0);
        let report = &summary.report.reports[0];
        assert!(report.winner_index.is_some());
        assert!(
            report
                .output
                .as_ref()
                .is_some_and(|path| path.ends_with("input.png"))
        );
        assert!(out_dir.join("input.png").exists());
    }

    #[test]
    fn quality_bracket_finds_lowest_accepted_and_nearest_rejected() {
        let candidates = vec![
            quality_candidate(Codec::Jxl, 74, CandidateStatus::Accepted),
            quality_candidate(
                Codec::Jxl,
                68,
                CandidateStatus::Rejected("quality 89.0 below 90.0".to_string()),
            ),
            quality_candidate(Codec::Jxl, 86, CandidateStatus::Accepted),
            quality_candidate(
                Codec::Webp,
                50,
                CandidateStatus::Rejected("quality 80.0 below 90.0".to_string()),
            ),
        ];

        let mut bracket = quality_bracket(&candidates, Codec::Jxl).unwrap();
        assert_eq!(bracket.low.quality, 68);
        assert_eq!(bracket.high.quality, 74);
        assert!(bracket.seen.contains(&68));
        assert!(bracket.seen.contains(&74));

        // Scores of 76 at q68 and 82 at q74 cross a floor of 80 two thirds
        // of the way up; sizes interpolate geometrically.
        bracket.low.score = Some(76.0);
        bracket.high.score = Some(82.0);
        bracket.low.size = Some(1_000);
        bracket.high.size = Some(4_000);
        assert!((bracket.crossing(80.0) - 72.0).abs() < 0.01);
        assert!((bracket.estimated_size(71.0).unwrap() - 2_000.0).abs() < 1.0);
        bracket.low.score = None;
        assert_eq!(bracket.crossing(80.0), 71.0);
    }

    fn lossy(codec: Codec, quality: u8, size: u64, status: CandidateStatus) -> CandidateReport {
        let mut candidate = quality_candidate(codec, quality, status);
        candidate.size_bytes = Some(size);
        candidate
    }

    fn failed() -> CandidateStatus {
        CandidateStatus::Rejected("quality 70.0 below 80.0".to_string())
    }

    fn smallest_policy() -> DecisionPolicy {
        DecisionPolicy {
            min_quality: 80.0,
            quality_slack: 0.0,
            allow_larger: false,
            original_size: 10_000,
            prefer_compatibility_delta: 0.05,
            target: crate::config::TargetPolicy::Smallest,
        }
    }

    fn best_of(codec: Codec, size_bytes: u64) -> AcceptedCandidate {
        AcceptedCandidate {
            codec,
            size_bytes,
            quality: QualityScore::lossless(),
        }
    }

    #[test]
    fn staged_plan_holds_back_all_but_the_middle_quality() {
        let quality = |quality| EncoderSettings::Quality { quality, effort: 0 };
        let plan = vec![
            (Codec::Webp, EncoderSettings::Lossless),
            (Codec::Webp, quality(92)),
            (Codec::Webp, quality(78)),
            (Codec::Webp, quality(60)),
            (Codec::Jpeg, quality(88)),
            (Codec::Jpeg, quality(70)),
            (Codec::Png, EncoderSettings::Palette { colors: 64 }),
        ];

        let staged = stage_plan(plan.clone(), false);

        assert_eq!(
            staged.first,
            vec![(Codec::Webp, quality(78)), (Codec::Jpeg, quality(70))]
        );
        assert_eq!(staged.held.len(), 5);
        assert_eq!(staged.held[0], (Codec::Webp, EncoderSettings::Lossless));
        assert_eq!(staged.mids[&Codec::Webp], 78);
        assert_eq!(staged.mids[&Codec::Jpeg], 70);
        assert!(!staged.mids.contains_key(&Codec::Png));

        // The lossless probe joins the first stage, after the lossy ones.
        let gated = stage_plan(plan, true);
        assert_eq!(
            gated.first,
            vec![
                (Codec::Webp, quality(78)),
                (Codec::Jpeg, quality(70)),
                (Codec::Webp, EncoderSettings::Lossless),
            ]
        );
        assert_eq!(gated.held.len(), 4);
    }

    fn lossless_candidate(codec: Codec, size: Option<u64>) -> CandidateReport {
        let mut candidate = quality_candidate(codec, 0, CandidateStatus::Accepted);
        candidate.settings = EncoderSettings::Lossless;
        candidate.size_bytes = size;
        candidate
    }

    #[test]
    fn lossless_gate_compares_the_probe_with_the_largest_winning_size() {
        let policy = smallest_policy();
        let gate = |probe: Option<u64>,
                    codec,
                    settings: EncoderSettings,
                    best: Option<&AcceptedCandidate>| {
            let candidates = [lossless_candidate(Codec::Webp, probe)];
            lossless_gate(&candidates, codec, &settings, &policy, best)
        };
        let lossless = || EncoderSettings::Lossless;

        // Against the original (10_000 bytes): the margin is 3x and exclusive.
        assert_eq!(gate(Some(30_000), Codec::Jxl, lossless(), None), None);
        let reason = gate(Some(30_001), Codec::Jxl, lossless(), None).unwrap();
        assert!(
            reason.starts_with("cannot win: lossless WebP is 30001 bytes"),
            "{reason}"
        );
        // Against a best accepted candidate that is smaller than the original.
        let best = best_of(Codec::Jpeg, 2_000);
        assert_eq!(
            gate(Some(6_000), Codec::Avif, lossless(), Some(&best)),
            None
        );
        assert!(gate(Some(6_001), Codec::Avif, lossless(), Some(&best)).is_some());
        // The probe, lossy candidates and a probe without a size are never gated.
        assert_eq!(gate(Some(99_999), Codec::Webp, lossless(), None), None);
        let lossy = EncoderSettings::Quality {
            quality: 70,
            effort: 0,
        };
        assert_eq!(gate(Some(99_999), Codec::Avif, lossy, None), None);
        assert_eq!(gate(None, Codec::Jxl, lossless(), None), None);
        // Nothing bounds the winner when larger outputs are allowed and none passed yet.
        let unbounded = DecisionPolicy {
            allow_larger: true,
            ..smallest_policy()
        };
        let candidates = [lossless_candidate(Codec::Webp, Some(99_999))];
        assert_eq!(
            lossless_gate(
                &candidates,
                Codec::Jxl,
                &EncoderSettings::Lossless,
                &unbounded,
                None
            ),
            None
        );
    }

    /// Counts progress the way the terminal bar and the GUI do.
    #[derive(Default)]
    struct ProgressTrace {
        planned: AtomicUsize,
        total: AtomicUsize,
        finished: AtomicUsize,
        at_file_end: Mutex<Option<(usize, usize)>>,
    }

    impl ProgressObserver for ProgressTrace {
        fn candidates_planned(&self, _path: &Path, total: usize) {
            self.planned.store(total, Ordering::SeqCst);
            self.total.store(total, Ordering::SeqCst);
        }
        fn candidate_started(
            &self,
            _: &Path,
            _: usize,
            total: usize,
            _: Codec,
            _: &EncoderSettings,
        ) {
            self.total.fetch_max(total, Ordering::SeqCst);
        }
        fn candidate_finished(&self, _path: &Path, _index: usize, _total: usize) {
            self.finished.fetch_add(1, Ordering::SeqCst);
        }
        fn candidates_settled(&self, _path: &Path) {
            self.total
                .store(self.finished.load(Ordering::SeqCst), Ordering::SeqCst);
        }
        fn file_finished(&self, _path: &Path) {
            *self.at_file_end.lock().unwrap() = Some((
                self.finished.load(Ordering::SeqCst),
                self.total.load(Ordering::SeqCst),
            ));
        }
    }

    #[test]
    fn a_finished_image_has_completed_every_candidate_it_counts() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, y| {
            Rgb([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8])
        }))
        .save_with_format(&input, image::ImageFormat::Png)
        .unwrap();
        let run = |label: &str, adjust: fn(&mut CliConfig)| {
            let mut config = CliConfig {
                inputs: vec![input.clone()],
                out_dir: Some(dir.path().join(label)),
                formats: CodecSelection::new([Codec::Jpeg]),
                allow_larger: true,
                min_quality: Some(0.0),
                workers: 2,
                gpu: crate::config::GpuConfig {
                    enabled: false,
                    ..crate::config::GpuConfig::default()
                },
                ..CliConfig::default()
            };
            adjust(&mut config);
            let trace = Arc::new(ProgressTrace::default());
            Optimizer::with_progress(config, trace.clone())
                .run()
                .unwrap();
            let (finished, total) = trace.at_file_end.lock().unwrap().unwrap();
            let planned = trace.planned.load(Ordering::SeqCst);
            println!("{label}: planned {planned}, finished {finished}, total at the end {total}");
            assert_eq!(finished, total, "{label}");
            assert!(finished > 0, "{label}");
            (finished, planned)
        };

        // The higher quality is assumed to pass, so fewer candidates finish than were planned.
        let (finished, planned) = run("staged", |config| {
            config.speed = crate::SpeedPreset::Fast;
        });
        assert!(finished < planned);
        // A budget that stops the search early leaves the rest unstarted.
        let (finished, planned) = run("budget", |config| config.max_candidate_attempts = Some(1));
        assert!(finished < planned);
        // The one-at-a-time path skips a settled format's higher qualities.
        let (finished, planned) = run("serial", |config| {
            config.speed = crate::SpeedPreset::Fast;
            config.max_temp_bytes = Some(u64::MAX);
        });
        assert!(finished < planned);
    }

    fn noisy_source(dir: &Path, name: &str) -> PathBuf {
        // Smooth colour under noise: tiny as a JPEG, large to store exactly.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut noise = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 16) as u32
        };
        let path = dir.join(name);
        RgbImage::from_fn(96, 96, |x, y| {
            Rgb([
                (x + noise()) as u8,
                (y + noise()) as u8,
                ((x + y) / 2 + noise()) as u8,
            ])
        })
        .save(&path)
        .unwrap();
        path
    }

    /// The lossless candidates of a report: (codec, size if encoded, skip reason if gated).
    fn lossless_outcomes(report: &ImageReport) -> Vec<(Codec, Option<u64>, Option<String>)> {
        report
            .candidates
            .iter()
            .filter(|candidate| candidate.settings == EncoderSettings::Lossless)
            .map(|candidate| {
                let gated = match &candidate.status {
                    CandidateStatus::Skipped(reason)
                        if reason.starts_with("cannot win: lossless") =>
                    {
                        Some(reason.clone())
                    }
                    _ => None,
                };
                (candidate.codec, candidate.size_bytes, gated)
            })
            .collect()
    }

    #[test]
    fn lossless_encoders_are_gated_on_webp_for_jpeg_sources_only() {
        let tools = ToolInventory::detect();
        let every_codec = [
            Codec::Jpeg,
            Codec::Webp,
            Codec::Avif,
            Codec::Jxl,
            Codec::Png,
        ];
        let codecs = every_codec
            .into_iter()
            .filter(|codec| tools.available_for(*codec))
            .collect::<Vec<_>>();
        if codecs.len() < every_codec.len() {
            // Says what is missing, or fails when the tools are required.
            let _ = smart_compress_codecs::tools_or_skip(&smart_compress_codecs::tool_names_for(
                &every_codec,
            ));
        }
        if !codecs.contains(&Codec::Webp) || codecs.len() < 4 {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let jpeg = noisy_source(dir.path(), "source.jpg");
        let png = noisy_source(dir.path(), "source.png");
        let run = |input: &Path, label: &str, formats: &[Codec], adjust: fn(&mut CliConfig)| {
            let mut config = CliConfig {
                inputs: vec![input.to_path_buf()],
                out_dir: Some(dir.path().join(label)),
                formats: CodecSelection::new(formats.iter().copied()),
                workers: 4,
                gpu: crate::config::GpuConfig {
                    enabled: false,
                    ..crate::config::GpuConfig::default()
                },
                ..CliConfig::default()
            };
            adjust(&mut config);
            let mut summary = Optimizer::new(config).run().unwrap();
            summary.report.reports.remove(0)
        };
        let others = |outcomes: &[(Codec, Option<u64>, Option<String>)]| {
            outcomes
                .iter()
                .filter(|(codec, ..)| *codec != Codec::Webp)
                .cloned()
                .collect::<Vec<_>>()
        };
        let all_gated = |report: &ImageReport| {
            let outcomes = lossless_outcomes(report);
            assert!(outcomes.len() >= 3, "{outcomes:?}");
            // The probe is encoded, and every other lossless encoder is skipped unencoded.
            let probe = outcomes
                .iter()
                .find(|(codec, ..)| *codec == Codec::Webp)
                .unwrap();
            assert!(probe.1.is_some() && probe.2.is_none(), "{outcomes:?}");
            assert!(
                probe.1.unwrap() > 3 * report.original_size,
                "fixture too compressible"
            );
            for (codec, size, gated) in others(&outcomes) {
                assert!(
                    size.is_none() && gated.is_some(),
                    "{codec:?}: {size:?} {gated:?}"
                );
            }
            assert!(report.winner_index.is_some(), "{:?}", report.errors);
            let winner = &report.candidates[report.winner_index.unwrap()];
            assert_ne!(winner.settings, EncoderSettings::Lossless);
        };
        let none_gated = |report: &ImageReport| {
            let outcomes = lossless_outcomes(report);
            assert!(outcomes.len() >= 3, "{outcomes:?}");
            for (codec, size, gated) in outcomes {
                assert!(
                    size.is_some() && gated.is_none(),
                    "{codec:?}: {size:?} {gated:?}"
                );
            }
        };

        // Concurrent search, one-at-a-time search.
        all_gated(&run(&jpeg, "staged", &codecs, |_| {}));
        all_gated(&run(&jpeg, "serial", &codecs, |config| {
            config.max_temp_bytes = Some(u64::MAX);
        }));
        // Verbose mode evaluates everything.
        none_gated(&run(&jpeg, "verbose", &codecs, |config| {
            config.verbose = true
        }));
        // A source that may already be lossless runs every lossless encoder.
        none_gated(&run(&png, "png", &codecs, |_| {}));
        // With larger outputs allowed and nothing passing a high floor, lossless may be the only choice.
        let strict = run(&jpeg, "strict", &codecs, |config| {
            config.allow_larger = true;
            config.min_quality = Some(99.99);
        });
        none_gated(&strict);
        assert_eq!(
            strict.candidates[strict.winner_index.unwrap()].settings,
            EncoderSettings::Lossless
        );
        // Without a second lossless encoder there is nothing to gate.
        let alone = run(&jpeg, "alone", &[Codec::Jpeg, Codec::Webp], |_| {});
        let outcomes = lossless_outcomes(&alone);
        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].1.is_some() && outcomes[0].2.is_none());
    }

    #[test]
    fn coarse_followup_skips_the_side_the_middle_quality_rules_out() {
        let policy = smallest_policy();
        let followup = |candidates: &[CandidateReport], best: Option<&AcceptedCandidate>, q| {
            coarse_followup(candidates, &policy, best, Codec::Jpeg, 78, q)
        };

        // A passing middle quality makes the higher one pointless and the lower one necessary.
        let passed = [lossy(Codec::Jpeg, 78, 500, CandidateStatus::Accepted)];
        assert!(
            followup(&passed, None, 92)
                .unwrap()
                .starts_with("assumed to pass")
        );
        assert_eq!(followup(&passed, None, 60), None);

        // A failing one needs the higher quality and rules the lower one out.
        let failing = [lossy(Codec::Jpeg, 78, 500, failed())];
        assert!(
            followup(&failing, None, 60)
                .unwrap()
                .starts_with("assumed to fail")
        );
        assert_eq!(followup(&failing, None, 92), None);
        // ... unless it is already no smaller than the best, so the format cannot win.
        let best = best_of(Codec::Avif, 400);
        assert!(
            followup(&failing, Some(&best), 92)
                .unwrap()
                .starts_with("cannot win")
        );
        let best = best_of(Codec::Avif, 600);
        assert_eq!(followup(&failing, Some(&best), 92), None);

        // Without a verdict on the middle quality, nothing is assumed.
        let unknown = [lossy(
            Codec::Jpeg,
            78,
            500,
            CandidateStatus::Skipped("encoder crashed".to_string()),
        )];
        assert_eq!(followup(&unknown, None, 92), None);
        assert_eq!(followup(&unknown, None, 60), None);
    }

    #[test]
    fn a_candidate_that_failed_to_run_has_no_verdict_whatever_its_message_says() {
        let policy = smallest_policy();
        // A timeout, a decode failure, and a metric error that begins like a
        // quality verdict: none of them says anything about the quality.
        for message in [
            "tools/ssimulacra2.exe Timeout after 120000 ms: terminated",
            "decode failed: opening candidate",
            "quality metric failed: GPU is required",
            "quality 12.0 below 80.0",
        ] {
            let broken = [lossy(
                Codec::Jpeg,
                78,
                500,
                CandidateStatus::Failed(message.to_string()),
            )];
            assert!(!broken[0].failed_quality(), "{message}");
            for quality in [92, 60] {
                assert_eq!(
                    coarse_followup(&broken, &policy, None, Codec::Jpeg, 78, quality),
                    None,
                    "{message}, q{quality}"
                );
            }
            // It is not an end of the bracket either: the upper end is the next one up.
            let candidates = [
                broken[0].clone(),
                lossy(Codec::Jpeg, 92, 900, CandidateStatus::Accepted),
            ];
            let bracket = quality_bracket(&candidates, Codec::Jpeg).unwrap();
            assert_eq!((bracket.low.quality, bracket.high.quality), (1, 92));
            assert!(inversions(&candidates, Codec::Jpeg).is_empty());
        }
        // A rejection for size is a verdict: the format is known to be too large.
        let too_large = lossy(
            Codec::Jpeg,
            78,
            20_000,
            CandidateStatus::Rejected("not smaller than original".to_string()),
        );
        assert!(!too_large.failed_quality());
        assert!(
            coarse_followup(&[too_large], &policy, None, Codec::Jpeg, 78, 92)
                .unwrap()
                .starts_with("assumed to pass")
        );
    }

    #[test]
    fn a_dry_run_report_is_not_resumed_by_a_real_run_and_a_missing_output_is_redone() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        gradient(&input);
        let out = dir.path().join("out");
        let config = CliConfig {
            inputs: vec![input],
            out_dir: Some(out.clone()),
            formats: CodecSelection::new([Codec::Png]),
            report: ReportFormat::Json,
            ..base_config()
        };

        // A dry run leaves a report; a real run asked to resume it still runs.
        let dry_report = dir.path().join("dry.json");
        Optimizer::new(CliConfig {
            dry_run: true,
            report_out: Some(dry_report.clone()),
            ..config.clone()
        })
        .run()
        .unwrap();
        let real_report = dir.path().join("real.json");
        let real = Optimizer::new(CliConfig {
            report_out: Some(real_report.clone()),
            resume_report: Some(dry_report),
            ..config.clone()
        })
        .run()
        .unwrap();
        assert!(out.join("input.png").is_file());
        assert_eq!(real.report.reports[0].outcome, ImageOutcome::Optimized);
        assert!(
            real.report
                .manifest
                .unwrap()
                .resume_decision
                .unwrap()
                .contains("does not match")
        );

        // The same run resumed finds its output and reuses the image ...
        let resumed = Optimizer::new(CliConfig {
            report_out: Some(dir.path().join("again.json")),
            resume_report: Some(real_report.clone()),
            ..config.clone()
        })
        .run()
        .unwrap();
        assert_eq!(
            resumed.report.manifest.unwrap().resume_decision.as_deref(),
            Some("resumed 1 compatible completed input(s)")
        );
        // ... unless the output has gone, which makes it do the work again.
        fs::remove_file(out.join("input.png")).unwrap();
        let redone = Optimizer::new(CliConfig {
            report_out: Some(dir.path().join("third.json")),
            resume_report: Some(real_report),
            ..config
        })
        .run()
        .unwrap();
        assert_eq!(
            redone.report.manifest.unwrap().resume_decision.as_deref(),
            Some("resumed 0 compatible completed input(s)")
        );
        assert!(out.join("input.png").is_file());
    }

    #[test]
    fn the_resume_fingerprint_covers_where_and_whether_outputs_are_written() {
        let fingerprint = |adjust: fn(&mut CliConfig)| {
            let mut config = CliConfig::default();
            adjust(&mut config);
            Optimizer::new(config).config_fingerprint
        };
        let base = fingerprint(|_| {});
        for (label, changed) in [
            ("dry_run", fingerprint(|config| config.dry_run = true)),
            (
                "out_dir",
                fingerprint(|config| config.out_dir = Some(PathBuf::from("elsewhere"))),
            ),
            (
                "out",
                fingerprint(|config| config.out = Some(PathBuf::from("one.webp"))),
            ),
            ("overwrite", fingerprint(|config| config.overwrite = true)),
            (
                "protect_inputs",
                fingerprint(|config| config.protect_inputs = false),
            ),
        ] {
            assert_ne!(base, changed, "{label}");
        }
    }

    #[test]
    fn a_second_run_skips_what_is_already_there_without_searching_or_failing() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        gradient(&input);
        let config = CliConfig {
            inputs: vec![input],
            formats: CodecSelection::new([Codec::Png]),
            ..base_config()
        };

        let first = Optimizer::new(config.clone()).run().unwrap();
        assert_eq!(first.report.reports[0].outcome, ImageOutcome::Optimized);
        let written = fs::read(dir.path().join("optimized").join("input.png")).unwrap();

        let second = Optimizer::new(config.clone()).run().unwrap();
        let report = &second.report.reports[0];
        assert_eq!(second.report.failures, 0);
        assert_eq!(report.outcome, ImageOutcome::Skipped);
        assert!(report.candidates.is_empty() && report.errors.is_empty());
        assert!(report.warnings[0].starts_with("output already exists"));
        assert!(
            second
                .report
                .outcome_summary()
                .starts_with("0 improved, 0 kept as they were, 1 skipped, 0 failed")
        );
        assert_eq!(
            fs::read(dir.path().join("optimized").join("input.png")).unwrap(),
            written
        );

        // With overwriting on, the image is searched again and the file replaced.
        let third = Optimizer::new(CliConfig {
            overwrite: true,
            ..config
        })
        .run()
        .unwrap();
        assert_eq!(third.report.reports[0].outcome, ImageOutcome::Optimized);
    }

    #[test]
    fn a_walk_includes_hidden_and_ignored_images_once_and_leaves_its_own_outputs_alone() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for path in [
            "visible.png",
            ".hidden/inside.png",
            "ignored/skipped.png",
            "optimized/earlier-output.png",
            "nested/optimized/earlier-output.png",
            "nested/deeper/kept.png",
        ] {
            let path = root.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            gradient(&path);
        }
        fs::write(
            root.join(".gitignore"),
            "ignored/
",
        )
        .unwrap();
        fs::write(
            root.join(".ignore"),
            "nested/deeper/
",
        )
        .unwrap();
        let names = |config: CliConfig| {
            let (files, warnings) = Optimizer::new(config).discover_inputs().unwrap();
            assert!(warnings.is_empty(), "{warnings:?}");
            let mut names = files
                .iter()
                .map(|file| {
                    file.strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/")
                })
                .collect::<Vec<_>>();
            names.sort();
            names
        };
        let walk = CliConfig {
            // The same folder twice, and a file in it that the folder also finds.
            inputs: vec![
                root.to_path_buf(),
                root.to_path_buf(),
                root.join("visible.png"),
            ],
            recursive: true,
            ..base_config()
        };

        assert_eq!(
            names(walk.clone()),
            [
                ".hidden/inside.png",
                "ignored/skipped.png",
                "nested/deeper/kept.png",
                "visible.png"
            ]
        );
        // With an output folder named, a folder called "optimized" is a user's
        // folder and only the output folder itself is left out.
        let with_out_dir = names(CliConfig {
            out_dir: Some(root.join("optimized")),
            ..walk
        });
        assert!(with_out_dir.contains(&"nested/optimized/earlier-output.png".to_string()));
        assert!(!with_out_dir.contains(&"optimized/earlier-output.png".to_string()));
    }

    #[test]
    fn out_names_one_file_and_its_extension_decides_the_format() {
        let dir = tempfile::tempdir().unwrap();
        let input = noisy_source(dir.path(), "source.png");
        let run = |out: &str| {
            Optimizer::new(CliConfig {
                inputs: vec![input.clone()],
                out: Some(dir.path().join(out)),
                formats: CodecSelection::new([Codec::Png, Codec::Jpeg]),
                min_quality: Some(0.0),
                ..base_config()
            })
            .run()
        };

        // A JPEG name gets a JPEG: only that format is tried.
        let summary = run("named.jpg").unwrap();
        let report = &summary.report.reports[0];
        assert!(report.candidates.iter().all(|c| c.codec == Codec::Jpeg));
        assert_eq!(
            &fs::read(dir.path().join("named.jpg")).unwrap()[..2],
            &[0xff, 0xd8]
        );
        assert_eq!(report.outcome, ImageOutcome::Optimized);

        // A name that names no format gets the winner's extension added.
        let summary = run("named.bin").unwrap();
        let report = &summary.report.reports[0];
        let output = report.output.as_ref().unwrap();
        assert!(
            output.ends_with("named.bin.png") || output.ends_with("named.bin.jpg"),
            "{output}"
        );
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.starts_with("--out has no image extension"))
        );
        assert!(!dir.path().join("named.bin").exists());

        // A format that is not selected is refused before any work.
        let error = Optimizer::new(CliConfig {
            inputs: vec![input.clone()],
            out: Some(dir.path().join("x.webp")),
            formats: CodecSelection::new([Codec::Png]),
            ..base_config()
        })
        .run()
        .unwrap_err();
        assert!(
            error.to_string().contains("names the webp format"),
            "{error}"
        );

        // So is one file for several images.
        let second = noisy_source(dir.path(), "second.png");
        let error = Optimizer::new(CliConfig {
            inputs: vec![input, second],
            out: Some(dir.path().join("only.png")),
            ..base_config()
        })
        .run()
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("--out names one file, but 2 input images"),
            "{error}"
        );
        let folder = Optimizer::new(CliConfig {
            inputs: vec![dir.path().to_path_buf()],
            recursive: true,
            out: Some(dir.path().join("only.png")),
            ..base_config()
        })
        .run()
        .unwrap_err();
        assert!(
            folder.to_string().contains("--out names one file"),
            "{folder}"
        );
    }

    #[test]
    fn only_a_failed_image_is_a_failure_of_the_run() {
        let dir = tempfile::tempdir().unwrap();
        let fine = noisy_source(dir.path(), "fine.png");
        let broken = dir.path().join("broken.png");
        fs::write(&broken, b"this is not an image").unwrap();
        let transparent = dir.path().join("transparent.png");
        image::RgbaImage::from_pixel(16, 16, image::Rgba([1, 2, 3, 100]))
            .save(&transparent)
            .unwrap();
        let run = |inputs: Vec<PathBuf>, adjust: fn(&mut CliConfig)| {
            let mut config = CliConfig {
                inputs,
                out_dir: Some(dir.path().join("out")),
                formats: CodecSelection::new([Codec::Jpeg]),
                allow_larger: true,
                ..base_config()
            };
            adjust(&mut config);
            Optimizer::new(config).run().unwrap().report
        };

        let improved = run(vec![fine.clone()], |config| config.min_quality = Some(0.0));
        assert_eq!(
            (improved.failures, improved.reports[0].outcome),
            (0, ImageOutcome::Optimized)
        );

        // Nothing passes the floor: the original is kept, which is no failure.
        let kept = run(vec![fine.clone()], |config| {
            config.min_quality = Some(1000.0);
            config.out_dir = Some(config.out_dir.clone().unwrap().join("kept"));
        });
        assert_eq!(
            (kept.failures, kept.reports[0].outcome),
            (0, ImageOutcome::Kept)
        );
        assert!(kept.reports[0].errors.is_empty() && kept.reports[0].winner_index.is_none());
        assert!(
            kept.reports[0]
                .warnings
                .iter()
                .any(|w| w.starts_with("nothing scored at least 1000.0"))
        );
        assert!(
            kept.outcome_summary()
                .starts_with("0 improved, 1 kept as they were, 0 skipped, 0 failed")
        );

        // An image that cannot be decoded fails, and so does one with no candidate to try.
        let report = run(vec![broken], |_| {});
        assert_eq!(
            (report.failures, report.reports[0].outcome),
            (1, ImageOutcome::Failed)
        );
        let report = run(vec![transparent], |_| {});
        assert_eq!(
            (report.failures, report.reports[0].outcome),
            (1, ImageOutcome::Failed)
        );
        assert!(
            report.reports[0].errors[0].starts_with(
                "no candidate could be planned for this image: jpeg cannot store transparency"
            ),
            "{:?}",
            report.reports[0].errors
        );
    }

    #[test]
    fn a_budget_that_stops_the_search_is_a_warning() {
        let dir = tempfile::tempdir().unwrap();
        let input = noisy_source(dir.path(), "source.png");
        let report = Optimizer::new(CliConfig {
            inputs: vec![input],
            out_dir: Some(dir.path().join("out")),
            formats: CodecSelection::new([Codec::Jpeg]),
            min_quality: Some(0.0),
            max_candidate_attempts: Some(1),
            ..base_config()
        })
        .run()
        .unwrap()
        .report;
        let image = &report.reports[0];
        assert_eq!(
            image.search_stop_reason.as_deref(),
            Some("candidate attempt budget of 1 reached")
        );
        assert!(image.warnings.iter().any(|warning| {
            warning.starts_with("the search stopped early (candidate attempt budget of 1 reached)")
        }));
        assert!(
            report
                .to_text()
                .contains("Search stopped early: candidate attempt budget of 1 reached")
        );
        assert!(
            report
                .to_markdown()
                .contains("Search stopped early: candidate attempt budget of 1 reached")
        );
    }

    #[test]
    fn an_encoder_that_declines_is_a_skip_and_any_other_error_is_a_failure() {
        let declined = anyhow!(EncodeError::Declined("pngquant made no file".to_string()))
            .context("encoding palette-64");
        let (status, influence) = encode_failure(&declined);
        assert!(
            matches!(&status, CandidateStatus::Skipped(reason) if reason.contains("pngquant made no file")),
            "{status:?}"
        );
        assert_eq!(influence, CandidateInfluence::Skipped);

        let crashed = anyhow!("cwebp exited with 1");
        let (status, influence) = encode_failure(&crashed);
        assert!(matches!(status, CandidateStatus::Failed(_)), "{status:?}");
        assert_eq!(influence, CandidateInfluence::Failed);
    }

    #[test]
    fn scratch_files_are_deleted_on_success_and_on_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metric-candidate-0.png");
        let attempt = |fail: bool| -> Result<()> {
            fs::write(&path, b"png")?;
            let _scratch = RemoveOnDrop(&path);
            anyhow::ensure!(!fail, "scoring failed");
            Ok(())
        };
        attempt(false).unwrap();
        assert!(!path.exists());
        attempt(true).unwrap_err();
        assert!(!path.exists());
    }

    #[test]
    fn only_images_with_a_side_of_8_px_are_scored_by_ssimulacra2() {
        let optimizer = Optimizer::new(CliConfig {
            metrics: crate::config::MetricsConfig {
                primary: PrimaryMetric::Ssimulacra2,
                ..Default::default()
            },
            ..base_config()
        });
        assert!(optimizer.scores_ssimulacra2(8, 8));
        assert!(!optimizer.scores_ssimulacra2(7, 100));
        assert!(!optimizer.scores_ssimulacra2(100, 7));
        // The floor of the built-in score is lower than the perceptual one.
        assert_eq!(quality_floor(&optimizer.config, true), 80.0);
        assert_eq!(quality_floor(&optimizer.config, false), 73.0);
        let builtin = Optimizer::new(CliConfig {
            metrics: crate::config::MetricsConfig {
                primary: PrimaryMetric::Builtin,
                ..Default::default()
            },
            ..base_config()
        });
        assert!(!builtin.scores_ssimulacra2(1000, 1000));
    }

    #[test]
    fn a_tiny_image_reports_the_metric_and_floor_it_was_scored_with() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("tiny.png");
        DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 4, Rgb([9, 99, 199])))
            .save(&input)
            .unwrap();
        let report = Optimizer::new(CliConfig {
            inputs: vec![input],
            out_dir: Some(dir.path().join("out")),
            formats: CodecSelection::new([Codec::Png]),
            metrics: crate::config::MetricsConfig {
                primary: PrimaryMetric::Ssimulacra2,
                ..Default::default()
            },
            ..base_config()
        })
        .run()
        .unwrap()
        .report;
        let image = &report.reports[0];
        assert_eq!(image.quality_metric, "builtin");
        assert_eq!(image.min_quality, Some(73.0));
        assert!(image.warnings.iter().any(|w| w.contains("under 8 px")));
    }

    #[test]
    fn a_lowered_input_limit_does_not_apply_to_the_pipelines_own_files() {
        let dir = tempfile::tempdir().unwrap();
        let candidate = dir.path().join("candidate.png");
        let original = DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, y| {
            Rgb([(x * 4) as u8, (y * 4) as u8, 7])
        }));
        original.save(&candidate).unwrap();
        let optimizer = Optimizer::new(CliConfig {
            max_input_bytes: 10,
            ..base_config()
        });

        assert_eq!(optimizer.pipeline_decode_limits().max_input_bytes, u64::MAX);
        assert_eq!(optimizer.image_decode_limits().max_input_bytes, 10);
        let decoded = optimizer
            .decode_candidate(Codec::Png, &candidate, &original, false, 1)
            .unwrap();
        assert_eq!(decoded.image.dimensions(), (64, 64));
        // A user's input of that size is still refused.
        assert!(optimizer.decode_input(&candidate).is_err());
    }

    #[test]
    fn the_icc_profile_is_judged_against_the_pixels_that_are_written() {
        // The smallest thing that says it is a grey ICC profile.
        let mut profile = vec![0_u8; 128];
        profile[16..20].copy_from_slice(b"GRAY");
        profile[36..40].copy_from_slice(b"acsp");
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("grey.png");
        {
            let mut encoder =
                image::codecs::png::PngEncoder::new(fs::File::create(&input).unwrap());
            image::ImageEncoder::set_icc_profile(&mut encoder, profile).unwrap();
            DynamicImage::ImageLumaA8(image::GrayAlphaImage::from_pixel(
                16,
                16,
                image::LumaA([90, 128]),
            ))
            .write_with_encoder(encoder)
            .unwrap();
        }
        let warnings = |alpha: AlphaPolicy, label: &str| {
            let report = Optimizer::new(CliConfig {
                inputs: vec![input.clone()],
                out_dir: Some(dir.path().join(label)),
                formats: CodecSelection::new([Codec::Png]),
                alpha,
                metadata: MetadataPolicy::Keep,
                ..base_config()
            })
            .run()
            .unwrap()
            .report;
            assert!(
                report.reports[0].errors.is_empty(),
                "{:?}",
                report.reports[0].errors
            );
            report.reports[0].warnings.clone()
        };

        // Grey stays grey under the profile ...
        assert!(
            warnings(AlphaPolicy::Preserve, "preserve")
                .iter()
                .all(|w| !w.contains("ICC"))
        );
        // ... but flattening makes RGB, and a grey profile no longer fits it.
        assert!(
            warnings(
                AlphaPolicy::Flatten {
                    background: "#ffffff".to_string()
                },
                "flatten"
            )
            .iter()
            .any(|w| w.starts_with("embedded ICC profile dropped"))
        );
    }

    #[test]
    fn inversions_are_found_in_score_and_in_size_but_not_in_monotonic_results() {
        let accepted = || CandidateStatus::Accepted;
        let monotonic = [
            lossy(Codec::Jpeg, 60, 300, failed()),
            lossy(Codec::Jpeg, 78, 500, accepted()),
            lossy(Codec::Jpeg, 92, 900, accepted()),
        ];
        assert!(inversions(&monotonic, Codec::Jpeg).is_empty());
        assert!(neighbour_probes(&monotonic, Codec::Jpeg, None, &smallest_policy()).is_none());
        assert!(held_back_to_run(&monotonic, &[], true).0.is_empty());

        // q60 passes although q78 fails.
        let score = [
            lossy(Codec::Jpeg, 60, 300, accepted()),
            lossy(Codec::Jpeg, 78, 500, failed()),
        ];
        let found = inversions(&score, Codec::Jpeg);
        assert_eq!(
            found,
            vec![Inversion {
                kind: InversionKind::Score,
                lower: 60,
                higher: 78,
                lower_size: Some(300)
            }]
        );
        let (qualities, warning) =
            neighbour_probes(&score, Codec::Jpeg, None, &smallest_policy()).unwrap();
        assert_eq!(qualities, vec![59, 61]);
        assert!(
            warning
                .contains("non-monotonic behaviour detected for jpeg: q60 passes while q78 fails")
        );
        assert!(warning.contains("q59, q61"));

        // q60 encodes larger than q78.
        let size = [
            lossy(Codec::Jpeg, 60, 700, accepted()),
            lossy(Codec::Jpeg, 78, 500, accepted()),
        ];
        let found = inversions(&size, Codec::Jpeg);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, InversionKind::Size);
        assert_eq!(
            neighbour_probes(&size, Codec::Jpeg, None, &smallest_policy())
                .unwrap()
                .0,
            vec![59, 61]
        );
        // A neighbour that was already tried is not tried again.
        let mut tried = size.to_vec();
        tried.push(lossy(Codec::Jpeg, 61, 480, accepted()));
        assert_eq!(
            neighbour_probes(&tried, Codec::Jpeg, None, &smallest_policy())
                .unwrap()
                .0,
            vec![59]
        );
        // A format that cannot beat the best is not probed.
        let best = best_of(Codec::Avif, 200);
        assert!(neighbour_probes(&size, Codec::Jpeg, Some(&best), &smallest_policy()).is_none());
        // Other formats and other efforts are not compared.
        assert!(inversions(&size, Codec::Webp).is_empty());
        let mut other_effort = size.to_vec();
        other_effort[1].settings = EncoderSettings::Quality {
            quality: 78,
            effort: 9,
        };
        assert!(inversions(&other_effort, Codec::Jpeg).is_empty());
    }

    #[test]
    fn held_back_qualities_run_after_an_inversion_or_for_an_unscored_upper_end() {
        let held = |quality| {
            CandidateReport::unevaluated(
                Codec::Jpeg,
                EncoderSettings::Quality { quality, effort: 3 },
                "assumed".to_string(),
            )
        };
        let held = [held(92), held(60), held(48)];
        let scored = |quality, overall| {
            let mut candidate = lossy(Codec::Jpeg, quality, 500, failed());
            candidate.quality = Some(QualityScore {
                overall,
                ..QualityScore::lossless()
            });
            candidate
        };

        // q78 failed and q92 is too large to be scored: the crossing needs q60 too.
        let oversize = lossy(
            Codec::Jpeg,
            92,
            20_000,
            CandidateStatus::Rejected("not smaller than original".to_string()),
        );
        let candidates = [scored(78, 70.0), oversize];
        assert_eq!(
            held_back_to_run(&candidates, &held, true),
            (vec![1], vec![])
        );
        assert!(held_back_to_run(&candidates, &held, false).0.is_empty());

        // q60 passing under a failing q78 is an inversion: everything held back runs.
        let inverted = [
            lossy(Codec::Jpeg, 60, 300, CandidateStatus::Accepted),
            scored(78, 70.0),
        ];
        let (picked, warnings) = held_back_to_run(&inverted, &held, false);
        assert_eq!(picked, vec![0, 1, 2]);
        assert!(warnings[0].contains("non-monotonic behaviour detected for jpeg"));
        assert!(warnings[0].contains("q92, q60, q48"));
    }

    #[test]
    fn staged_search_skips_what_the_middle_quality_rules_out() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, y| {
            Rgb([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8])
        }))
        .save_with_format(&input, image::ImageFormat::Png)
        .unwrap();
        let reasons = |min_quality: f32| {
            let summary = Optimizer::new(CliConfig {
                inputs: vec![input.clone()],
                out_dir: Some(dir.path().join(format!("out-{min_quality}"))),
                formats: CodecSelection::new([Codec::Jpeg]),
                allow_larger: true,
                min_quality: Some(min_quality),
                workers: 2,
                gpu: crate::config::GpuConfig {
                    enabled: false,
                    ..crate::config::GpuConfig::default()
                },
                ..CliConfig::default()
            })
            .run()
            .unwrap();
            let report = &summary.report.reports[0];
            assert!(report.warnings.iter().all(|w| !w.contains("non-monotonic")));
            report
                .candidates
                .iter()
                .take(3)
                .map(|candidate| match &candidate.status {
                    CandidateStatus::Skipped(reason) => reason.clone(),
                    _ => "evaluated".to_string(),
                })
                .collect::<Vec<_>>()
        };

        // Everything passes: q78 first, then q60; q92 is assumed to pass.
        let passing = reasons(0.0);
        assert_eq!(passing[0], "evaluated");
        assert_eq!(passing[1], "evaluated");
        assert!(passing[2].starts_with("assumed to pass"), "{passing:?}");
        // Nothing passes: q78 first, then q92; q60 is assumed to fail.
        let failing = reasons(1000.0);
        assert_eq!(failing[0], "evaluated");
        assert_eq!(failing[1], "evaluated");
        assert!(failing[2].starts_with("assumed to fail"), "{failing:?}");
    }

    #[test]
    fn candidate_cache_reuses_encoded_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        let cache_dir = dir.path().join("cache");
        let first_out = dir.path().join("first");
        let second_out = dir.path().join("second");
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([50, 80, 110])));
        image
            .save_with_format(&input, image::ImageFormat::Png)
            .unwrap();

        let base_config = CliConfig {
            inputs: vec![input],
            preset: Preset::Balanced,
            target: TargetPolicy::MaxCompat,
            formats: CodecSelection::new([Codec::Png]),
            allow_larger: true,
            max_pixels: 100_000_000,
            cache: CacheConfig {
                enabled: true,
                dir: Some(cache_dir),
                ..CacheConfig::default()
            },
            ..CliConfig::default()
        };

        let first = Optimizer::new(CliConfig {
            out_dir: Some(first_out),
            ..base_config.clone()
        })
        .run()
        .unwrap();
        assert_eq!(first.report.failures, 0);

        let second = Optimizer::new(CliConfig {
            out_dir: Some(second_out),
            ..base_config
        })
        .run()
        .unwrap();
        assert_eq!(second.report.failures, 0);
        let candidate = &second.report.reports[0].candidates[0];
        assert_eq!(candidate.backend.as_deref(), Some("cache"));
        assert_eq!(candidate.encode_ms, Some(0));
        assert!(
            candidate
                .warnings
                .iter()
                .any(|warning| warning == "cache hit")
        );
        assert!(
            candidate
                .warnings
                .iter()
                .any(|warning| warning.starts_with("quality cache hit"))
        );
        assert!(candidate.quality.is_some());
        assert!(candidate.score_backend.is_some());
        assert!(candidate.score_ms.is_some());
        assert!(candidate.total_ms.is_some());
    }

    #[test]
    fn a_candidate_of_the_wrong_size_is_rejected_before_any_decoder_runs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("candidate.jxl");
        // The start of a bare JPEG XL codestream for a 37x21 image.
        fs::write(
            &path,
            [
                0xff, 0x0a, 0xa0, 0xd0, 0x01, 0x00, 0x14, 0x08, 0x02, 0x00, 0xa4, 0x02,
            ],
        )
        .unwrap();
        let original = DynamicImage::ImageRgb8(RgbImage::new(64, 48));

        let error = Optimizer::new(CliConfig::default())
            .decode_candidate(Codec::Jxl, &path, &original, false, 1)
            .err()
            .expect("the size mismatch must fail the candidate");

        // The message comes from the header check, not from djxl, and the
        // decoder's output file was never created.
        assert!(error.to_string().contains("37x21 but 64x48"), "{error}");
        assert!(!path.with_extension("decoded.pam").exists());
    }

    #[test]
    fn cpu_and_gpu_backends_reach_the_same_decisions() {
        if gpu_or_skip(GpuWgpuBackend::new()).is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        // Smooth colour with fine texture, so lossy candidates land on both
        // sides of the floor.
        DynamicImage::ImageRgb8(RgbImage::from_fn(640, 400, |x, y| {
            let texture = ((x * 7 + y * 13) % 23) as u8;
            Rgb([
                (x / 3) as u8 + texture,
                (y / 2) as u8 + texture / 2,
                ((x + y) / 5) as u8,
            ])
        }))
        .save_with_format(&input, image::ImageFormat::Png)
        .unwrap();
        let run = |gpu_enabled: bool, name: &str| {
            let mut config = CliConfig {
                inputs: vec![input.clone()],
                out_dir: Some(dir.path().join(name)),
                formats: CodecSelection::new([Codec::Jpeg, Codec::Png]),
                metadata: MetadataPolicy::Keep,
                allow_larger: true,
                // The built-in score is the one the backends compute.
                metrics: crate::config::MetricsConfig {
                    primary: PrimaryMetric::Builtin,
                    ..Default::default()
                },
                ..CliConfig::default()
            };
            config.gpu.enabled = gpu_enabled;
            config.gpu.min_pixels_for_gpu_metrics = 0;
            let mut summary = Optimizer::new(config).run().unwrap();
            summary.report.reports.remove(0)
        };
        let (cpu, gpu) = (run(false, "cpu"), run(true, "gpu"));
        assert!(
            gpu.candidates
                .iter()
                .any(|candidate| candidate.score_backend == Some(ActualScoreBackend::Gpu))
        );
        assert_eq!(cpu.winner_index, gpu.winner_index);
        assert_eq!(cpu.candidates.len(), gpu.candidates.len());
        let mut verdicts = HashSet::new();
        for (cpu, gpu) in cpu.candidates.iter().zip(&gpu.candidates) {
            assert_eq!((cpu.codec, &cpu.settings), (gpu.codec, &gpu.settings));
            assert_eq!(cpu.size_bytes, gpu.size_bytes);
            // Same verdict, and scores too close to move a candidate across the floor.
            assert_eq!(
                std::mem::discriminant(&cpu.status),
                std::mem::discriminant(&gpu.status),
                "{:?} vs {:?}",
                cpu.status,
                gpu.status
            );
            if let (Some(cpu), Some(gpu)) = (cpu.quality, gpu.quality) {
                assert!((cpu.overall - gpu.overall).abs() <= 0.01, "{cpu:?} {gpu:?}");
            }
            verdicts.insert(std::mem::discriminant(&cpu.status));
        }
        assert!(
            verdicts.len() > 1,
            "the fixture should produce mixed verdicts"
        );
    }

    #[test]
    fn large_images_are_scored_on_full_resolution_samples() {
        let large = DynamicImage::ImageRgb8(RgbImage::from_fn(5000, 4000, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 7])
        }));
        let metric = bounded_metric_image(&large);
        assert_eq!(metric.dimensions(), (3072, 3072));
        assert!(!metric.color().has_alpha());
        // The first tile is the centre of the top-left grid cell, unscaled.
        let (x, y) = ((5000 / 3 - 1024) / 2, (4000 / 3 - 1024) / 2);
        assert_eq!(
            metric.to_rgb8().get_pixel(0, 0),
            &Rgb([(x % 256) as u8, (y % 256) as u8, 7])
        );

        let small = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
        assert_eq!(bounded_metric_image(&small).dimensions(), (64, 64));
    }

    #[test]
    fn one_at_a_time_search_stops_a_format_once_it_is_settled() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, y| {
            Rgb([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8])
        }))
        .save_with_format(&input, image::ImageFormat::Png)
        .unwrap();
        let summary = Optimizer::new(CliConfig {
            inputs: vec![input],
            out_dir: Some(dir.path().join("out")),
            formats: CodecSelection::new([Codec::Jpeg]),
            allow_larger: true,
            min_quality: Some(0.0),
            // A temporary-disk budget selects the one-at-a-time path.
            max_temp_bytes: Some(u64::MAX),
            gpu: crate::config::GpuConfig {
                enabled: false,
                ..crate::config::GpuConfig::default()
            },
            ..CliConfig::default()
        })
        .run()
        .unwrap();

        let report = &summary.report.reports[0];
        assert_eq!(summary.report.failures, 0);
        let settled = report
            .candidates
            .iter()
            .filter(|candidate| {
                matches!(&candidate.status, CandidateStatus::Skipped(reason) if reason.contains("settled"))
            })
            .count();
        // The lowest grid quality passes a floor of zero; the rest are never encoded.
        assert_eq!(settled, crate::SpeedPreset::Balanced.qualities().len() - 1);
        assert!(report.candidates.iter().all(|candidate| {
            candidate.size_bytes.is_some()
                || matches!(candidate.status, CandidateStatus::Skipped(_))
        }));
    }

    #[test]
    fn flatten_composites_alpha_over_background() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        let mut rgba = image::RgbaImage::from_pixel(8, 8, image::Rgba([0, 0, 255, 255]));
        rgba.put_pixel(0, 0, image::Rgba([0, 0, 255, 0]));
        rgba.save(&input).unwrap();
        let summary = Optimizer::new(CliConfig {
            inputs: vec![input],
            out_dir: Some(dir.path().join("out")),
            formats: CodecSelection::new([Codec::Png]),
            allow_larger: true,
            alpha: AlphaPolicy::Flatten {
                background: "#ff0000".to_string(),
            },
            gpu: crate::config::GpuConfig {
                enabled: false,
                ..crate::config::GpuConfig::default()
            },
            ..CliConfig::default()
        })
        .run()
        .unwrap();

        assert_eq!(summary.report.failures, 0);
        let output = image::open(dir.path().join("out").join("input.png")).unwrap();
        assert!(!output.color().has_alpha());
        assert_eq!(output.to_rgb8().get_pixel(0, 0), &Rgb([255, 0, 0]));
        assert_eq!(output.to_rgb8().get_pixel(1, 1), &Rgb([0, 0, 255]));
    }

    #[test]
    fn same_named_inputs_publish_distinct_outputs_and_protect_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let mut inputs = Vec::new();
        for (folder, shade) in [("a", 10), ("b", 200)] {
            let folder = dir.path().join(folder);
            fs::create_dir(&folder).unwrap();
            let input = folder.join("x.png");
            DynamicImage::ImageRgb8(RgbImage::from_pixel(8, 8, Rgb([shade, shade, shade])))
                .save_with_format(&input, image::ImageFormat::Png)
                .unwrap();
            inputs.push(input);
        }
        let base = CliConfig {
            formats: CodecSelection::new([Codec::Png]),
            allow_larger: true,
            gpu: crate::config::GpuConfig {
                enabled: false,
                ..crate::config::GpuConfig::default()
            },
            ..CliConfig::default()
        };
        let out_dir = dir.path().join("out");
        let summary = Optimizer::new(CliConfig {
            inputs: inputs.clone(),
            out_dir: Some(out_dir.clone()),
            ..base.clone()
        })
        .run()
        .unwrap();
        assert_eq!(summary.report.failures, 0);
        assert_eq!(fs::read_dir(&out_dir).unwrap().count(), 2);

        // Writing next to the input with overwrite enabled must not replace it.
        let original_bytes = fs::read(&inputs[0]).unwrap();
        let summary = Optimizer::new(CliConfig {
            inputs: vec![inputs[0].clone()],
            out_dir: Some(dir.path().join("a")),
            overwrite: true,
            ..base
        })
        .run()
        .unwrap();
        assert_eq!(summary.report.failures, 1);
        assert_eq!(fs::read(&inputs[0]).unwrap(), original_bytes);
    }

    fn base_config() -> CliConfig {
        CliConfig {
            allow_larger: true,
            gpu: crate::config::GpuConfig {
                enabled: false,
                ..crate::config::GpuConfig::default()
            },
            ..CliConfig::default()
        }
    }

    fn gradient(path: &Path) {
        DynamicImage::ImageRgb8(RgbImage::from_fn(32, 32, |x, y| {
            Rgb([x as u8 * 8, y as u8 * 8, 90])
        }))
        .save(path)
        .unwrap();
    }

    #[test]
    fn a_file_that_was_already_in_the_input_folder_is_not_replaced() {
        let dir = tempfile::tempdir().unwrap();
        gradient(&dir.path().join("a.png"));
        let unrelated = dir.path().join("a.jpg");
        fs::write(&unrelated, b"someone else's picture").unwrap();

        let summary = Optimizer::new(CliConfig {
            inputs: vec![dir.path().join("a.png")],
            out_dir: Some(dir.path().to_path_buf()),
            formats: CodecSelection::new([Codec::Jpeg]),
            min_quality: Some(0.0),
            overwrite: true,
            ..base_config()
        })
        .run()
        .unwrap();

        let report = &summary.report.reports[0];
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("already there")),
            "{:?}",
            report.errors
        );
        assert_eq!(fs::read(&unrelated).unwrap(), b"someone else's picture");

        // Without protect-inputs the same run replaces it, as asked.
        let summary = Optimizer::new(CliConfig {
            inputs: vec![dir.path().join("a.png")],
            out_dir: Some(dir.path().to_path_buf()),
            formats: CodecSelection::new([Codec::Jpeg]),
            min_quality: Some(0.0),
            overwrite: true,
            protect_inputs: false,
            ..base_config()
        })
        .run()
        .unwrap();
        assert_eq!(summary.report.failures, 0);
        assert_ne!(fs::read(&unrelated).unwrap(), b"someone else's picture");
    }

    #[test]
    fn a_panic_in_one_image_is_an_error_entry_and_the_batch_goes_on() {
        let dir = tempfile::tempdir().unwrap();
        let inputs = ["panic.png", "fine.png"].map(|name| dir.path().join(name));
        inputs.iter().for_each(|input| gradient(input));
        let optimizer = Optimizer::new(CliConfig {
            inputs: inputs.to_vec(),
            out_dir: Some(dir.path().join("out")),
            formats: CodecSelection::new([Codec::Png]),
            report_out: Some(dir.path().join("report.json")),
            report: ReportFormat::Json,
            ..base_config()
        });

        let summary = optimizer.run().unwrap();

        let reports = &summary.report.reports;
        assert_eq!(reports.len(), 2);
        assert!(
            reports[0].errors[0].starts_with("internal error: injected panic"),
            "{:?}",
            reports[0].errors
        );
        assert!(reports[1].errors.is_empty() && reports[1].winner_index.is_some());
        assert_eq!(summary.report.failures, 1);
        // What the panicking image held was given back.
        assert_eq!(optimizer.file_memory.used(), 0);
        assert_eq!(optimizer.active_files.load(Ordering::SeqCst), 0);
        assert!(dir.path().join("report.json").exists());
    }

    #[test]
    fn checkpointed_runs_keep_input_order_and_resume_by_path() {
        let dir = tempfile::tempdir().unwrap();
        let inputs = ["c.png", "a.png", "b.png"].map(|name| dir.path().join(name));
        inputs.iter().for_each(|input| gradient(input));
        let first_report = dir.path().join("first.json");
        let config = CliConfig {
            inputs: inputs.to_vec(),
            dry_run: true,
            report: ReportFormat::Json,
            report_out: Some(first_report.clone()),
            ..base_config()
        };
        let first = Optimizer::new(config.clone()).run().unwrap();
        let names = |summary: &BatchSummary| {
            summary
                .report
                .reports
                .iter()
                .map(|report| report.input.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            names(&first),
            inputs.map(|input| input.display().to_string())
        );

        let second = Optimizer::new(CliConfig {
            report_out: Some(dir.path().join("second.json")),
            resume_report: Some(first_report),
            ..config
        })
        .run()
        .unwrap();
        let manifest = second.report.manifest.clone().unwrap();
        assert_eq!(
            manifest.resume_decision.as_deref(),
            Some("resumed 3 compatible completed input(s)")
        );
        assert_eq!(manifest.completed_inputs, 3);
        assert_eq!(names(&second), names(&first));
    }

    fn avif_with_size(width: u32, height: u32) -> Vec<u8> {
        let boxed = |kind: &[u8; 4], body: &[u8]| {
            let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
            out.extend_from_slice(kind);
            out.extend_from_slice(body);
            out
        };
        let ispe = [
            vec![0; 4],
            width.to_be_bytes().into(),
            height.to_be_bytes().into(),
        ]
        .concat();
        let ipma = [
            &[0, 0, 0, 0][..],
            &1_u32.to_be_bytes(),
            &1_u16.to_be_bytes(),
            &[1, 0x81],
        ]
        .concat();
        let iprp = [
            boxed(b"ipco", &boxed(b"ispe", &ispe)),
            boxed(b"ipma", &ipma),
        ]
        .concat();
        let meta = [
            vec![0; 4],
            boxed(b"pitm", &[0, 0, 0, 0, 0, 1]),
            boxed(b"iprp", &iprp),
        ]
        .concat();
        [
            boxed(b"ftyp", b"avif\0\0\0\0avifmif1"),
            boxed(b"meta", &meta),
        ]
        .concat()
    }

    #[test]
    fn an_avif_header_beyond_the_limits_never_reaches_the_decoder_tool() {
        let dir = tempfile::tempdir().unwrap();
        let optimizer = Optimizer::new(base_config());
        for (width, height, expected) in [
            (
                50_000,
                50_000,
                "image has 2500000000 pixels, exceeding the configured 100000000 pixel limit",
            ),
            (
                100_001,
                1,
                "image width 100001 exceeds the configured 100000 pixel width limit",
            ),
        ] {
            let input = dir.path().join("big.avif");
            fs::write(&input, avif_with_size(width, height)).unwrap();
            let error = optimizer.decode_input(&input).err().unwrap();
            assert_eq!(error.to_string(), expected);
        }
        let limited = Optimizer::new(CliConfig {
            max_decoded_bytes: 1000,
            ..base_config()
        });
        let input = dir.path().join("big.avif");
        fs::write(&input, avif_with_size(100, 100)).unwrap();
        let error = limited.decode_input(&input).err().unwrap();
        assert!(
            error
                .to_string()
                .starts_with("decoded image requires 40000 bytes")
        );
    }

    #[test]
    fn gpu_required_reports_late_metric_failure() {
        if gpu_or_skip(GpuWgpuBackend::new()).is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([20, 30, 40])))
            .save_with_format(&input, image::ImageFormat::Png)
            .unwrap();
        let gpu = crate::config::GpuConfig {
            required: true,
            max_vram_bytes: Some(1),
            ..crate::config::GpuConfig::default()
        };
        let summary = Optimizer::new(CliConfig {
            inputs: vec![input],
            out_dir: Some(dir.path().join("out")),
            formats: CodecSelection::new([Codec::Png]),
            allow_larger: true,
            gpu,
            ..CliConfig::default()
        })
        .run()
        .unwrap();

        assert_eq!(summary.report.failures, 1);
        let image = &summary.report.reports[0];
        assert!(
            image
                .candidates
                .iter()
                .any(|candidate| format!("{:?}", candidate.status).contains("GPU is required")),
            "{:?} {:?}",
            image.errors,
            image
                .candidates
                .iter()
                .map(|c| &c.status)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn resume_fingerprint_ignores_report_paths() {
        let base = CliConfig::default();
        let mut with_paths = base.clone();
        with_paths.report_out = Some(PathBuf::from("one.json"));
        with_paths.resume_report = Some(PathBuf::from("old.json"));

        let first = Optimizer::with_progress(base, Arc::new(NoopProgress));
        let second = Optimizer::with_progress(with_paths, Arc::new(NoopProgress));

        assert_eq!(first.config_fingerprint, second.config_fingerprint);
    }

    /// A candidate whose status is a verdict on quality (accepted, or rejected
    /// with a "quality ..." reason) carries the score it was judged by.
    fn quality_candidate(codec: Codec, quality: u8, status: CandidateStatus) -> CandidateReport {
        let scored = match &status {
            CandidateStatus::Accepted => true,
            CandidateStatus::Rejected(reason) => reason.starts_with("quality "),
            _ => false,
        };
        CandidateReport {
            codec,
            settings: EncoderSettings::Quality { quality, effort: 3 },
            path: None,
            size_bytes: Some(100),
            encode_ms: Some(1),
            decode_ms: Some(1),
            decode_process_ms: None,
            decode_io_ms: None,
            preprocess_ms: None,
            score_ms: Some(1),
            cache_ms: None,
            external_metric_ms: None,
            gpu_upload_ms: None,
            gpu_transform_ms: None,
            gpu_compute_ms: None,
            gpu_readback_ms: None,
            gpu_timestamp_ns: None,
            total_ms: Some(3),
            quality: scored.then(|| QualityScore {
                overall: 75.0,
                ..QualityScore::lossless()
            }),
            backend: Some("test".to_string()),
            score_backend: Some(ActualScoreBackend::Cpu),
            quality_source: QualitySource::Fresh,
            cache_hit_kind: None,
            process_failure: None,
            gpu_verification: None,
            influence: None,
            bytes: Default::default(),
            status,
            warnings: Vec::new(),
        }
    }

    /// Images between the planning of their candidates and their end.
    #[derive(Default)]
    struct ImagesInFlight {
        now: AtomicUsize,
        peak: AtomicUsize,
    }

    impl ProgressObserver for ImagesInFlight {
        fn candidates_planned(&self, _path: &Path, _total: usize) {
            let now = self.now.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
        }
        fn file_finished(&self, _path: &Path) {
            // Saturating decrement; `fetch_update` is deprecated on newer toolchains
            // and its replacement is missing on older ones.
            let mut now = self.now.load(Ordering::SeqCst);
            while now > 0 {
                match self
                    .now
                    .compare_exchange(now, now - 1, Ordering::SeqCst, Ordering::SeqCst)
                {
                    Ok(_) => break,
                    Err(actual) => now = actual,
                }
            }
        }
    }

    fn cpu_only() -> crate::config::GpuConfig {
        crate::config::GpuConfig {
            enabled: false,
            ..crate::config::GpuConfig::default()
        }
    }

    #[test]
    fn an_image_larger_than_the_memory_limit_runs_alone_and_completes() {
        let dir = tempfile::tempdir().unwrap();
        let inputs = (0..3)
            .map(|index| noisy_source(dir.path(), &format!("source-{index}.png")))
            .collect::<Vec<_>>();
        let run = |label: &str, memory: u64| {
            let trace = Arc::new(ImagesInFlight::default());
            let summary = Optimizer::with_progress(
                CliConfig {
                    inputs: inputs.clone(),
                    out_dir: Some(dir.path().join(label)),
                    formats: CodecSelection::new([Codec::Jpeg]),
                    allow_larger: true,
                    min_quality: Some(0.0),
                    workers: 4,
                    max_memory_bytes: Some(memory),
                    gpu: cpu_only(),
                    ..CliConfig::default()
                },
                trace.clone(),
            )
            .run()
            .unwrap();
            (summary.report.reports, trace.peak.load(Ordering::SeqCst))
        };

        // Every image is larger than one byte, so each is granted the whole
        // limit, and they take turns.
        let (limited, limited_peak) = run("limited", 1);
        let (open, _) = run("open", u64::MAX);

        assert_eq!(limited_peak, 1);
        // Each image says that it alone needs more than the limit; nothing else does.
        for report in &limited {
            assert!(
                report
                    .warnings
                    .iter()
                    .any(|w| w.contains("more than the whole memory limit")),
                "{:?}",
                report.warnings
            );
        }
        assert!(open.iter().all(|report| report.warnings.is_empty()));
        let sizes = |reports: &[ImageReport]| {
            reports
                .iter()
                .map(|report| {
                    assert!(report.errors.is_empty(), "{:?}", report.errors);
                    report.candidates[report.winner_index.unwrap()].size_bytes
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(sizes(&limited), sizes(&open));
        let waits = |reports: &[ImageReport]| {
            reports
                .iter()
                .map(|report| report.timings.backpressure_wait_ms.unwrap())
                .sum::<u64>()
        };
        assert!(waits(&limited) > 0);
    }

    #[test]
    fn candidates_wait_for_temporary_room_only_while_something_is_running() {
        let optimizer = Optimizer::new(CliConfig {
            max_run_temp_bytes: Some(1_000),
            gpu: cpu_only(),
            ..CliConfig::default()
        });
        let waited = AtomicU64::new(0);
        // The soft limit is three quarters of the run budget.
        optimizer.run_temp_usage.store(900, Ordering::SeqCst);

        // Nothing is running, so nothing will make room: the work goes ahead.
        assert_eq!(optimizer.wait_for_temp_room(None, &waited), Ok(()));

        optimizer.active_candidates.store(1, Ordering::SeqCst);
        let deadline = Instant::now() + std::time::Duration::from_millis(60);
        assert_eq!(
            optimizer.wait_for_temp_room(Some(deadline), &waited),
            Err(WaitEnded::TimedOut)
        );
        assert!(waited.load(Ordering::SeqCst) >= 50);

        std::thread::scope(|scope| {
            let waiting = scope.spawn(|| optimizer.wait_for_temp_room(None, &waited));
            std::thread::sleep(std::time::Duration::from_millis(100));
            optimizer.run_temp_usage.store(100, Ordering::SeqCst);
            optimizer.temp_gate.wake();
            assert_eq!(waiting.join().unwrap(), Ok(()));
        });
    }

    #[test]
    fn the_source_size_comes_from_the_header_not_a_decode() {
        let dir = tempfile::tempdir().unwrap();
        let png = noisy_source(dir.path(), "source.png");
        let optimizer = Optimizer::new(CliConfig {
            gpu: cpu_only(),
            ..CliConfig::default()
        });
        assert_eq!(optimizer.source_pixels(&png, 1), 96 * 96);
        // A file that is no image is sized from its length, at most max_pixels.
        let junk = dir.path().join("junk.png");
        std::fs::write(&junk, [0_u8; 100]).unwrap();
        assert_eq!(optimizer.source_pixels(&junk, 100), 400);
        assert_eq!(
            optimizer.source_pixels(&junk, u64::MAX),
            optimizer.config.max_pixels
        );
    }
}
