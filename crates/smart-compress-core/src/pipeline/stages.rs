use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::atomic::AtomicU64,
    time::Instant,
};

use smart_compress_codecs::{Codec, EncoderSettings, MetadataPayload, cancellation_requested};

use super::{
    CandidateEvaluationRequest, Optimizer, QualityRefinementRequest, StagedPlan,
    evaluate::record_candidate,
    guards::{TempUsageGuard, directory_bytes},
    plan::{is_lossless_probe, lossless_gate, stage_plan},
    search::{attempts, coarse_followup, held_back_to_run},
    wait_stop_reason,
};
use crate::{
    candidate::{CandidateReport, CandidateStatus},
    decision::{AcceptedCandidate, DecisionPolicy},
    report::ImageReport,
    scoring::{GpuPreparedOriginal, OriginalImage},
};

/// What the search of one image reads, fixed once its inputs are prepared.
pub(super) struct SearchContext<'a> {
    pub(super) input: &'a Path,
    pub(super) image_start: Instant,
    pub(super) image_deadline: Option<Instant>,
    pub(super) waited: &'a AtomicU64,
    pub(super) parallelism: usize,
    pub(super) concurrency: usize,
    pub(super) gated: bool,
    pub(super) candidate_total: usize,
    pub(super) pool: &'a rayon::ThreadPool,
    pub(super) encoder_source: &'a Path,
    pub(super) original: &'a OriginalImage,
    pub(super) metadata: &'a MetadataPayload,
    pub(super) prepared_original: Option<&'a GpuPreparedOriginal>,
    pub(super) preprocess_ms: Option<u64>,
    pub(super) gpu_preprocess_fallback_reason: Option<&'a str>,
    pub(super) has_alpha: bool,
    pub(super) candidate_dir: &'a Path,
    pub(super) policy: &'a DecisionPolicy,
    pub(super) cache_source_hash: Option<&'a str>,
    pub(super) external_metric_original: Option<&'a Path>,
}

impl<'a> SearchContext<'a> {
    fn request(
        &self,
        candidate_index: usize,
        codec: Codec,
        settings: EncoderSettings,
        peers: usize,
    ) -> CandidateEvaluationRequest<'a> {
        CandidateEvaluationRequest {
            input: self.input,
            codec,
            settings,
            encoder_source: self.encoder_source,
            original: self.original,
            metadata: self.metadata,
            prepared_original: self.prepared_original,
            preprocess_ms: self.preprocess_ms,
            gpu_preprocess_fallback_reason: self.gpu_preprocess_fallback_reason,
            has_alpha: self.has_alpha,
            candidate_dir: self.candidate_dir,
            policy: self.policy,
            cache_source_hash: self.cache_source_hash,
            external_metric_original: self.external_metric_original,
            candidate_index,
            progress_index: candidate_index + 1,
            progress_total: self.candidate_total,
            peers,
            deadline: self.image_deadline,
            waited: self.waited,
        }
    }

    fn refinement_request(&self) -> QualityRefinementRequest<'a> {
        QualityRefinementRequest {
            input: self.input,
            parallelism: self.concurrency,
            image_start: self.image_start,
            waited: self.waited,
            encoder_source: self.encoder_source,
            original: self.original,
            metadata: self.metadata,
            prepared_original: self.prepared_original,
            preprocess_ms: self.preprocess_ms,
            gpu_preprocess_fallback_reason: self.gpu_preprocess_fallback_reason,
            has_alpha: self.has_alpha,
            candidate_dir: self.candidate_dir,
            policy: self.policy,
            cache_source_hash: self.cache_source_hash,
            external_metric_original: self.external_metric_original,
        }
    }
}

/// Orders a plan for the one-at-a-time path.
fn order_one_at_a_time(candidate_plan: &mut [(Codec, EncoderSettings)], gated: bool) {
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

impl Optimizer {
    /// Searches the planned candidates stage by stage, then refines the qualities
    /// around the floor. Returns why the search had to stop, if it did.
    pub(super) fn search_candidates(
        &self,
        ctx: &SearchContext<'_>,
        mut candidate_plan: Vec<(Codec, EncoderSettings)>,
        temp_usage: &mut TempUsageGuard,
        report: &mut ImageReport,
    ) -> Option<String> {
        let mut best: Option<AcceptedCandidate> = None;
        // Encoding, decoding and scoring a candidate depend only on the source,
        // so the planned grid is evaluated concurrently and the decisions below
        // still run in plan order. Temporary-disk budgets need the one-at-a-time
        // path to stop before the next file is written.
        if ctx.parallelism == 1 {
            order_one_at_a_time(&mut candidate_plan, ctx.gated);
        }
        // With candidates running in parallel the coarse grid is staged: the middle
        // quality of every lossy format first, then only the side of it the quality
        // floor must lie on. Verbose mode keeps the whole grid in one stage.
        let staged = if ctx.parallelism > 1 && !self.config.verbose {
            stage_plan(candidate_plan, ctx.gated)
        } else {
            StagedPlan {
                first: candidate_plan,
                held: Vec::new(),
                mids: HashMap::new(),
            }
        };
        let mut search_stop_reason = self.run_staged(ctx, staged, report, &mut best, temp_usage);
        if search_stop_reason.is_none() {
            search_stop_reason =
                self.refine_quality_search(ctx.refinement_request(), report, &mut best);
        }
        search_stop_reason
    }

    /// Runs the first stage, then the held-back candidates that the first
    /// stage's results leave worth running.
    fn run_staged(
        &self,
        ctx: &SearchContext<'_>,
        staged: StagedPlan,
        report: &mut ImageReport,
        best: &mut Option<AcceptedCandidate>,
        temp_usage: &mut TempUsageGuard,
    ) -> Option<String> {
        let StagedPlan { first, held, mids } = staged;
        let mut search_stop_reason = self.run_stage(ctx, first, report, best, temp_usage);
        if search_stop_reason.is_none() && !held.is_empty() {
            let mut run = Vec::new();
            let mut skipped = Vec::new();
            for (codec, settings) in held {
                let reason = match (&settings, mids.get(&codec)) {
                    (EncoderSettings::Quality { quality, .. }, Some(mid)) => coarse_followup(
                        &report.candidates,
                        ctx.policy,
                        best.as_ref(),
                        codec,
                        *mid,
                        *quality,
                    ),
                    _ if ctx.gated => lossless_gate(
                        &report.candidates,
                        codec,
                        &settings,
                        ctx.policy,
                        best.as_ref(),
                    ),
                    _ => None,
                };
                match reason {
                    Some(reason) => {
                        skipped.push(CandidateReport::unevaluated(codec, settings, reason));
                    }
                    None => run.push((codec, settings)),
                }
            }
            search_stop_reason = self.run_stage(ctx, run, report, best, temp_usage);
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
                search_stop_reason = self.run_stage(ctx, run, report, best, temp_usage);
            }
            for candidate in skipped {
                record_candidate(ctx.policy, candidate, report, best);
            }
        }
        search_stop_reason
    }

    /// Evaluates one stage of the plan and records it in plan order, whichever
    /// candidate finishes first. Returns why the search had to stop, if it did.
    fn run_stage(
        &self,
        ctx: &SearchContext<'_>,
        stage: Vec<(Codec, EncoderSettings)>,
        report: &mut ImageReport,
        best: &mut Option<AcceptedCandidate>,
        temp_usage: &mut TempUsageGuard,
    ) -> Option<String> {
        let first_index = report.candidates.len();
        let stage_best = best.clone();
        let mut evaluated = if ctx.parallelism > 1 {
            self.evaluate_concurrently(ctx, &stage, first_index, stage_best.as_ref(), report)
        } else {
            Vec::new()
        };
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
                search_stop_reason = Some(format!("candidate attempt budget of {limit} reached"));
                break;
            }
            if let Some(limit) = self.config.max_image_time
                && ctx.image_start.elapsed() >= limit
            {
                search_stop_reason = Some(format!("image wall-time budget of {:?} reached", limit));
                break;
            }
            let lossy = matches!(settings, EncoderSettings::Quality { .. });
            let candidate = match evaluated.get_mut(offset).and_then(Option::take) {
                Some(candidate) => candidate,
                None if ctx.parallelism == 1 && lossy && settled.contains(&codec) => {
                    CandidateReport::unevaluated(
                        codec,
                        settings,
                        "a lower quality of this format already settled it".to_string(),
                    )
                }
                None => match (ctx.parallelism == 1 && ctx.gated)
                    .then(|| {
                        lossless_gate(
                            &report.candidates,
                            codec,
                            &settings,
                            ctx.policy,
                            best.as_ref(),
                        )
                    })
                    .flatten()
                {
                    Some(reason) => CandidateReport::unevaluated(codec, settings, reason),
                    None => {
                        if let Err(ended) = self.wait_for_temp_room(ctx.image_deadline, ctx.waited)
                        {
                            search_stop_reason =
                                Some(wait_stop_reason(ended, self.config.max_image_time));
                            break;
                        }
                        ctx.pool
                            .install(|| {
                                self.score_candidate_report(
                                    ctx.request(first_index + offset, codec, settings.clone(), 1),
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
            record_candidate(ctx.policy, candidate, report, best);
            let temporary_bytes = directory_bytes(ctx.candidate_dir);
            report.temporary_bytes_current = temporary_bytes;
            report.temporary_bytes_peak = report.temporary_bytes_peak.max(temporary_bytes);
            if let Err(error) = temp_usage.set(temporary_bytes, self.config.max_run_temp_bytes) {
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
    }

    /// Evaluates the candidates of a stage in chunks on the worker pool. A
    /// candidate that was not evaluated (the search ended first) is left as None.
    fn evaluate_concurrently(
        &self,
        ctx: &SearchContext<'_>,
        stage: &[(Codec, EncoderSettings)],
        first_index: usize,
        stage_best: Option<&AcceptedCandidate>,
        report: &ImageReport,
    ) -> Vec<Option<CandidateReport>> {
        use rayon::prelude::*;
        let mut evaluated: Vec<Option<CandidateReport>> = Vec::new();
        let budget = self
            .config
            .max_candidate_attempts
            .map_or(usize::MAX, |limit| limit.saturating_sub(attempts(report)));
        let planned = stage.iter().take(budget).enumerate().collect::<Vec<_>>();
        for chunk in planned.chunks(ctx.concurrency) {
            if self
                .wait_for_temp_room(ctx.image_deadline, ctx.waited)
                .is_err()
            {
                break;
            }
            ctx.pool.install(|| {
                evaluated.par_extend(chunk.par_iter().map(|(index, (codec, settings))| {
                    if cancellation_requested()
                        || self
                            .config
                            .max_image_time
                            .is_some_and(|limit| ctx.image_start.elapsed() >= limit)
                    {
                        return None;
                    }
                    // The pass or fail of a quality places the bracket of its
                    // format, so only candidates outside any bracket may be
                    // skipped as dominated.
                    let best = match settings {
                        EncoderSettings::Quality { .. } => None,
                        _ => stage_best,
                    };
                    self.score_candidate_report(
                        ctx.request(first_index + index, *codec, settings.clone(), chunk.len()),
                        best,
                    )
                }));
            });
        }
        evaluated
    }
}
