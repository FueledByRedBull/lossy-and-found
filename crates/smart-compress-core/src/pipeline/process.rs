use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use anyhow::Result;
use smart_compress_codecs::{
    Codec, EncoderSettings, WaitEnded, cancellation_requested, normalized_png,
};

use super::{
    CandidateEvaluationRequest, Optimizer, QualityRefinementRequest, SCORING_VERSION, StagedPlan,
    evaluate::record_candidate,
    guards::{ActiveCandidate, FileProgressGuard, TempUsageGuard, directory_bytes},
    inputs::input_identity,
    output::metadata_payload,
    plan::{is_lossless_probe, lossless_gate, lossless_gate_applies, stage_plan},
    quality_floor,
    score::write_metric_png,
    search::{attempts, coarse_followup, held_back_to_run},
    source::external_tools_accept_input,
    wait_stop_reason, work,
};
use crate::{
    candidate::{CandidateReport, CandidateStatus},
    config::MetadataPolicy,
    decision::{AcceptedCandidate, DecisionPolicy},
    gpu::{GpuStatus, GpuWorkEstimate},
    image_probe::{apply_orientation, icc_profile_mismatch, probe_image},
    memory,
    report::{ImageOutcome, ImageReport, ImageStageTimings},
    scoring::{ComputeBackend, FULL_SIZE_METRIC_PIXELS, GpuPreparedOriginal, OriginalImage},
};

fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
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
}
