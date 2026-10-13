use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use anyhow::Result;
use smart_compress_codecs::{
    Codec, EncoderSettings, MetadataPayload, Permit, WaitEnded, cancellation_requested,
    normalized_png,
};

use super::{
    Optimizer, SCORING_VERSION,
    guards::{ActiveCandidate, FileProgressGuard, TempUsageGuard, directory_bytes},
    inputs::input_identity,
    output::metadata_payload,
    plan::lossless_gate_applies,
    quality_floor,
    score::write_metric_png,
    source::external_tools_accept_input,
    stages::SearchContext,
    wait_stop_reason, work,
};
use crate::{
    candidate::CandidateStatus,
    config::MetadataPolicy,
    decision::DecisionPolicy,
    gpu::{GpuStatus, GpuWorkEstimate},
    image_probe::{DecodedSource, apply_orientation, icc_profile_mismatch, probe_image},
    memory,
    report::{ImageOutcome, ImageReport, ImageStageTimings},
    scoring::{FULL_SIZE_METRIC_PIXELS, GpuPreparedOriginal, OriginalImage},
};

fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

/// What the GPU prepared of the original, and the fallback when it could not.
#[derive(Default)]
struct GpuPreprocess {
    prepared_original: Option<GpuPreparedOriginal>,
    preprocess_ms: Option<u64>,
    fallback_reason: Option<String>,
}

/// The oriented source pixels, the metadata to write with them, and what the
/// probe found out about how to score them.
struct SourceImage {
    original: OriginalImage,
    metadata_payload: MetadataPayload,
    min_quality: f32,
    has_alpha: bool,
}

/// What the candidates of one image read and write. The field order is the
/// drop order and matters: the usage count goes first, then the candidate
/// folder, then the normalized copy.
struct Workspace {
    temp_usage: TempUsageGuard,
    candidate_dir: tempfile::TempDir,
    normalized: Option<(tempfile::TempDir, PathBuf)>,
    absolute_input: PathBuf,
    cache_source_hash: Option<String>,
    external_metric_original: Option<PathBuf>,
}

/// The file the encoders read: the normalized copy when there is one.
fn encoder_source<'a>(
    normalized: &'a Option<(tempfile::TempDir, PathBuf)>,
    absolute_input: &'a Path,
) -> &'a Path {
    normalized
        .as_ref()
        .map_or(absolute_input, |(_, path)| path.as_path())
}

impl Optimizer {
    pub(super) fn process_one(&self, input: &Path) -> ImageReport {
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
    pub(super) fn wait_for_temp_room(
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

        let image_deadline = self.config.max_image_time.map(|limit| image_start + limit);
        let (source_pixels, parallelism, concurrency) =
            self.size_image(input, original_size, &mut report);
        let Some(_reserved) = self.admit_image(source_pixels, image_deadline, waited, &mut report)
        else {
            return report;
        };

        self.active_files.fetch_add(1, Ordering::Relaxed);
        let _active_file = ActiveCandidate(&self.active_files, &self.temp_gate);
        #[cfg(test)]
        assert!(
            input.file_name().is_none_or(|name| name != "panic.png"),
            "injected panic"
        );

        let Some((decoded, alpha_changed)) = self.decode_source(input, &mut report) else {
            return report;
        };
        let orientation_applied = decoded.metadata.orientation_applied();
        let Some(gpu) = self.preprocess_on_gpu(&decoded, &mut report) else {
            return report;
        };
        let source = self.read_source(input, decoded, gpu.prepared_original.as_ref(), &mut report);
        if self.ends_before_search(&mut report) {
            return report;
        }

        let predicted = report
            .probe
            .as_ref()
            .is_some_and(|probe| self.searches_predicted(parallelism, probe));
        let predicted_target = predicted.then_some(source.min_quality - self.config.quality_slack);
        let Some((candidate_plan, gated)) = self.plan_search(input, predicted_target, &mut report)
        else {
            return report;
        };
        let Some(mut workspace) = self.prepare_workspace(
            input,
            (orientation_applied, alpha_changed),
            &source,
            &mut report,
        ) else {
            return report;
        };

        let policy = DecisionPolicy {
            min_quality: source.min_quality,
            quality_slack: self.config.quality_slack,
            allow_larger: self.config.allow_larger,
            original_size,
            prefer_compatibility_delta: self.config.preset.compat_delta(),
            target: self.config.target,
        };

        let search_start = Instant::now();
        let candidate_total = candidate_plan.len();
        self.progress.candidates_planned(input, candidate_total);
        let ctx = SearchContext {
            input,
            image_start,
            image_deadline,
            waited,
            parallelism,
            concurrency,
            gated,
            predicted,
            candidate_total,
            pool: self.pool().expect("run builds the worker pool"),
            encoder_source: encoder_source(&workspace.normalized, &workspace.absolute_input),
            original: &source.original,
            metadata: &source.metadata_payload,
            prepared_original: gpu.prepared_original.as_ref(),
            preprocess_ms: gpu.preprocess_ms,
            gpu_preprocess_fallback_reason: gpu.fallback_reason.as_deref(),
            has_alpha: source.has_alpha,
            candidate_dir: workspace.candidate_dir.path(),
            policy: &policy,
            cache_source_hash: workspace.cache_source_hash.as_deref(),
            external_metric_original: workspace.external_metric_original.as_deref(),
        };
        let search_stop_reason =
            self.search_candidates(&ctx, candidate_plan, &mut workspace.temp_usage, &mut report);
        if !self.settle_search(input, search_start, search_stop_reason, &mut report) {
            return report;
        }
        self.write_outcome(
            input,
            original_size,
            &policy,
            workspace.candidate_dir.path(),
            &mut report,
        );

        report
    }

    /// Counts the candidates the image may run at once, and warns when it alone
    /// is more than the memory limit. Returns the source pixels, the candidate
    /// slots and the concurrency.
    fn size_image(
        &self,
        input: &Path,
        original_size: u64,
        report: &mut ImageReport,
    ) -> (u64, usize, usize) {
        let workers = self.config.workers.max(1);
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
        (source_pixels, parallelism, concurrency)
    }

    /// Reserves the image's memory; None when the wait for it ended.
    fn admit_image(
        &self,
        source_pixels: u64,
        image_deadline: Option<Instant>,
        waited: &AtomicU64,
        report: &mut ImageReport,
    ) -> Option<Permit<'_>> {
        // The image reserves its memory before it is decoded. One larger than
        // the limit is granted the whole of it, and so runs alone.
        let admission_start = Instant::now();
        let reference = if self.ssimulacra2_in_process {
            memory::in_process_reference_bytes(source_pixels.min(FULL_SIZE_METRIC_PIXELS))
        } else {
            0
        };
        let reserved = self.file_memory.acquire(
            memory::file_bytes(source_pixels).saturating_add(reference),
            image_deadline,
        );
        waited.fetch_add(
            admission_start.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        match reserved {
            Ok(permit) => Some(permit),
            Err(ended) => {
                report.errors.push(format!(
                    "{} while waiting for memory",
                    wait_stop_reason(ended, self.config.max_image_time)
                ));
                None
            }
        }
    }

    /// Decodes the source and applies the alpha policy and the colour profile
    /// checks. Returns the decoded source and whether the alpha policy changed
    /// the image; None when the image cannot go on.
    fn decode_source(
        &self,
        input: &Path,
        report: &mut ImageReport,
    ) -> Option<(DecodedSource, bool)> {
        let source_decode_start = Instant::now();
        let decoded = self.decode_input(input);
        report.timings.source_decode_ms = Some(source_decode_start.elapsed().as_millis() as u64);
        let mut decoded = match decoded {
            Ok(value) => value,
            Err(error) => {
                report.errors.push(format!("{error:#}"));
                return None;
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
                return None;
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
        Some((decoded, alpha_changed))
    }

    /// Prepares the original on the GPU when the image is scored there; None
    /// when the image cannot go on.
    fn preprocess_on_gpu(
        &self,
        decoded: &DecodedSource,
        report: &mut ImageReport,
    ) -> Option<GpuPreprocess> {
        let mut preprocess = GpuPreprocess::default();
        if self.should_use_gpu_preprocess(&decoded.image)
            && let Some(gpu) = self.compute.as_ref().as_gpu()
        {
            let preprocess_start = Instant::now();
            let gpu_unit = match work(1) {
                Ok(unit) => unit,
                Err(error) => {
                    report.errors.push(format!("{error:#}"));
                    return None;
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
                    preprocess.prepared_original = Some(prepared);
                }
                Err(error) => {
                    let reason = format!("GPU preprocess failed: {error:#}");
                    if self.config.gpu.required {
                        report.errors.push(reason);
                        return None;
                    }
                    report
                        .warnings
                        .push(format!("CPU scoring fallback: {reason}"));
                    preprocess.fallback_reason = Some(reason);
                }
            }
            preprocess.preprocess_ms = Some(preprocess_start.elapsed().as_millis() as u64);
            report.timings.gpu_preprocess_ms = preprocess.preprocess_ms;
        }
        Some(preprocess)
    }

    /// Orients the image, picks the metric it is scored with, probes it and
    /// records the probe in the report.
    fn read_source(
        &self,
        input: &Path,
        mut decoded: DecodedSource,
        prepared_original: Option<&GpuPreparedOriginal>,
        report: &mut ImageReport,
    ) -> SourceImage {
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
            prepared_original.map(GpuPreparedOriginal::overview),
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
            prepared_original.and_then(GpuPreparedOriginal::band_pixels),
        ) {
            work.tiling_required_for_full_source_preprocess = true;
            work.selected_tile_pixels = Some(band_pixels);
        }
        report.probe = Some(probe);
        SourceImage {
            original,
            metadata_payload,
            min_quality,
            has_alpha,
        }
    }

    /// Ends the image before any search for a dry run, or when a metric tool
    /// it needs is missing. Returns whether the image ends here.
    fn ends_before_search(&self, report: &mut ImageReport) -> bool {
        if self.config.dry_run {
            report.outcome = ImageOutcome::DryRun;
            return true;
        }
        if self.use_ssimulacra2
            && !self.ssimulacra2_in_process
            && !self.backend.tools().ssimulacra2.enabled
        {
            report.errors.push(
                "SSIMULACRA2 metric requested, but ssimulacra2 tool is not available".to_string(),
            );
            return true;
        }
        false
    }

    /// Plans the candidates, and whether the lossless probe gates the other
    /// lossless encoders. None when the image ends here: nothing could be
    /// planned, or its output is already there.
    fn plan_search(
        &self,
        input: &Path,
        predicted_target: Option<f32>,
        report: &mut ImageReport,
    ) -> Option<(Vec<(Codec, EncoderSettings)>, bool)> {
        let probe = report
            .probe
            .as_ref()
            .expect("probe set after successful decode");
        let candidate_plan = self.plan_candidates(probe, predicted_target);
        let gated = !self.config.verbose && lossless_gate_applies(probe, &candidate_plan);
        if candidate_plan.is_empty() {
            report.errors.push(format!(
                "no candidate could be planned for this image: {}",
                self.unplanned_reasons(probe)
            ));
            return None;
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
                        return None;
                    }
                }
            }
            if let Some(path) = existing {
                report.warnings.push(format!(
                    "output already exists, left as it is: {} (pass --overwrite to replace it)",
                    path.display()
                ));
                report.outcome = ImageOutcome::Skipped;
                return None;
            }
        }
        Some((candidate_plan, gated))
    }

    /// Writes the normalized copy of the source when the encoders cannot read
    /// the input itself, opens the folder the candidates are written in and
    /// accounts for its temporary space. None when the image cannot go on.
    fn prepare_workspace(
        &self,
        input: &Path,
        (orientation_applied, alpha_changed): (bool, bool),
        source: &SourceImage,
        report: &mut ImageReport,
    ) -> Option<Workspace> {
        let SourceImage {
            original,
            metadata_payload,
            ..
        } = source;
        // External encoders copy EXIF/XMP from a source file they read
        // directly, so only `keep` may hand them the untouched input.
        let needs_normalized_source = orientation_applied
            || alpha_changed
            || !external_tools_accept_input(input)
            || !metadata_payload.is_empty()
            || !matches!(self.config.metadata, MetadataPolicy::Keep);
        let normalize_start = Instant::now();
        let normalized = if needs_normalized_source {
            match work(1).and_then(|_unit| normalized_png(original, metadata_payload)) {
                Ok(value) => Some(value),
                Err(error) => {
                    report.errors.push(format!("{error:#}"));
                    return None;
                }
            }
        } else {
            None
        };
        report.timings.normalize_ms = Some(normalize_start.elapsed().as_millis() as u64);
        // Tools run in their own folder and read a leading `-` as an option,
        // so the input they are given is absolute.
        let absolute_input = std::path::absolute(input).unwrap_or_else(|_| input.to_path_buf());
        let cache_lookup_start = Instant::now();
        let cache_source_hash = self.cache.as_ref().and_then(|cache| {
            cache
                .source_hash(encoder_source(&normalized, &absolute_input))
                .ok()
        });
        report.timings.cache_lookup_ms = Some(cache_lookup_start.elapsed().as_millis() as u64);
        let candidate_dir = match tempfile::tempdir() {
            Ok(dir) => dir,
            Err(error) => {
                report.errors.push(format!("{error:#}"));
                return None;
            }
        };
        let mut temp_usage = TempUsageGuard::new(
            Arc::clone(&self.run_temp_usage),
            Arc::clone(&self.temp_gate),
        );
        let external_metric_prepare_start = Instant::now();
        let external_metric_original = if self.use_ssimulacra2 && !self.ssimulacra2_in_process {
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
                    match write_metric_png(original, &path) {
                        Ok(()) => Some(path),
                        Err(error) => {
                            report.errors.push(format!(
                                "preparing external metric original failed: {error:#}"
                            ));
                            return None;
                        }
                    }
                }
            }
        } else {
            None
        };
        if self.ssimulacra2_in_process
            && self.scores_ssimulacra2(original.width(), original.height())
            && let Err(error) = self.prepare_ssimulacra2_reference(original)
        {
            report.errors.push(format!(
                "preparing the SSIMULACRA2 reference failed: {error:#}"
            ));
            return None;
        }
        report.timings.external_metric_prepare_ms =
            Some(external_metric_prepare_start.elapsed().as_millis() as u64);
        let initial_temp_bytes = directory_bytes(candidate_dir.path());
        report.temporary_bytes_current = initial_temp_bytes;
        report.temporary_bytes_peak = initial_temp_bytes;
        if let Err(error) = temp_usage.set(initial_temp_bytes, self.config.max_run_temp_bytes) {
            report.errors.push(format!("{error:#}"));
            return None;
        }
        if let Some(limit) = self.config.max_temp_bytes
            && initial_temp_bytes > limit
        {
            report.errors.push(format!(
                "temporary storage budget of {limit} bytes exceeded ({initial_temp_bytes} bytes)"
            ));
            return None;
        }
        Some(Workspace {
            temp_usage,
            candidate_dir,
            normalized,
            absolute_input,
            cache_source_hash,
            external_metric_original,
        })
    }

    /// Records why the search ended and what it cost. False when the run was
    /// cancelled, which ends the image without an output.
    fn settle_search(
        &self,
        input: &Path,
        search_start: Instant,
        search_stop_reason: Option<String>,
        report: &mut ImageReport,
    ) -> bool {
        report.search_stop_reason = search_stop_reason;
        self.progress.candidates_settled(input);
        report.timings.search_ms = Some(search_start.elapsed().as_millis() as u64);
        if cancellation_requested() {
            report.errors.push("cancelled".to_string());
            for candidate in &mut report.candidates {
                candidate.path = None;
            }
            return false;
        }
        if let Some(reason) = &report.search_stop_reason {
            report.warnings.push(format!(
                "the search stopped early ({reason}); a smaller passing candidate may have gone untried"
            ));
        }
        true
    }

    /// Picks the winner, writes it, and settles the outcome and the temporary
    /// space in the report.
    fn write_outcome(
        &self,
        input: &Path,
        original_size: u64,
        policy: &DecisionPolicy,
        candidate_dir: &Path,
        report: &mut ImageReport,
    ) {
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
        report.temporary_bytes_current = directory_bytes(candidate_dir);
        report.temporary_bytes_peak = report
            .temporary_bytes_peak
            .max(report.temporary_bytes_current);
        for candidate in &mut report.candidates {
            candidate.path = None;
        }
    }
}
