use std::collections::HashSet;

use smart_compress_codecs::{Codec, EncoderSettings, cancellation_requested};

use super::{
    CandidateEvaluationRequest, Optimizer, QualityRefinementRequest, evaluate::record_candidate,
    wait_stop_reason,
};
use crate::{
    candidate::{CandidateReport, CandidateStatus},
    decision::{AcceptedCandidate, DecisionPolicy},
    report::ImageReport,
};

/// The formats whose quality is searched.
pub(super) const LOSSY: [Codec; 4] = [Codec::Jpeg, Codec::Webp, Codec::Avif, Codec::Jxl];

impl Optimizer {
    /// Probes evenly spaced qualities between each codec's highest rejected
    /// and lowest accepted grid point. The probes are independent, so they run
    /// concurrently instead of as a one-at-a-time bisection.
    pub(super) fn refine_quality_search(
        &self,
        request: QualityRefinementRequest<'_>,
        report: &mut ImageReport,
        best: &mut Option<AcceptedCandidate>,
        codecs: &[Codec],
    ) -> Option<String> {
        for round in 0..self.config.speed.refinement_rounds() {
            let mut points = self.plan_refinement(
                &report.candidates,
                best.as_ref(),
                request.policy,
                round,
                codecs,
            );
            // Next to a point that broke the monotonic assumption the bracket
            // cannot be trusted, so its neighbours are probed too.
            for &codec in codecs {
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
            if let Some(reason) = self.run_points(&request, points, report, best, true) {
                return Some(reason);
            }
        }
        None
    }

    /// Evaluates qualities at each format's final effort, in chunks whose
    /// candidates run concurrently. A candidate no smaller than the best so far
    /// is left unscored when `skip_dominated` is set. Returns why the search had
    /// to stop, if it did.
    pub(super) fn run_points(
        &self,
        request: &QualityRefinementRequest<'_>,
        mut points: Vec<(Codec, u8)>,
        report: &mut ImageReport,
        best: &mut Option<AcceptedCandidate>,
        skip_dominated: bool,
    ) -> Option<String> {
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
            let current_best = best.as_ref().filter(|_| skip_dominated);
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
        stop_reason
    }

    /// Picks the qualities one refinement round should try: for each format
    /// that can still win, a few around where its score should cross the floor.
    fn plan_refinement(
        &self,
        candidates: &[CandidateReport],
        best: Option<&AcceptedCandidate>,
        policy: &DecisionPolicy,
        round: usize,
        codecs: &[Codec],
    ) -> Vec<(Codec, u8)> {
        let offsets = self.config.speed.refinement_offsets(round);
        let floor = policy.min_quality - self.config.quality_slack;
        let mut plans = Vec::new();
        for &codec in codecs {
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
pub(super) fn quality_bracket(
    candidates: &[CandidateReport],
    codec: Codec,
) -> Option<QualityBracket> {
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
pub(super) fn attempts(report: &ImageReport) -> usize {
    report
        .candidates
        .iter()
        .filter(|candidate| candidate.total_ms.is_some())
        .count()
}

/// Why a coarse quality held back at the first stage is not worth running,
/// given how the middle quality of its format came out. Both rest on a lower
/// quality scoring lower and encoding smaller; `held_back_to_run` checks that
/// assumption once more points are known.
pub(super) fn coarse_followup(
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
pub(super) enum InversionKind {
    /// A lower quality passed while a higher one failed.
    Score,
    /// A lower quality produced a larger file than a higher one.
    Size,
}

/// Two evaluated qualities of one format, at the same effort, that break the
/// assumption that a lower quality scores lower and encodes smaller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Inversion {
    pub(super) kind: InversionKind,
    pub(super) lower: u8,
    pub(super) higher: u8,
    pub(super) lower_size: Option<u64>,
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
pub(super) fn inversions(candidates: &[CandidateReport], codec: Codec) -> Vec<Inversion> {
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

pub(super) fn describe_inversions(codec: Codec, inversions: &[Inversion]) -> String {
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
pub(super) fn held_back_to_run(
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
pub(super) fn neighbour_probes(
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
pub(super) struct BracketEnd {
    pub(super) quality: u8,
    pub(super) score: Option<f32>,
    pub(super) size: Option<u64>,
}

pub(super) struct QualityBracket {
    pub(super) low: BracketEnd,
    pub(super) high: BracketEnd,
    /// The next scored failure under `low`, for when `high` has no score.
    below: Option<BracketEnd>,
    pub(super) seen: HashSet<u8>,
}

impl QualityBracket {
    /// Quality at which the score should reach `floor`, assuming it rises
    /// linearly: between the two ends when both were scored, otherwise along
    /// the slope of the two failures below. With nothing to go on, the middle.
    pub(super) fn crossing(&self, floor: f32) -> f32 {
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
    pub(super) fn estimated_size(&self, quality: f32) -> Option<f64> {
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
