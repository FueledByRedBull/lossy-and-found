//! The predicted search. Each lossy format starts at a quality estimated from
//! the target score; the measured error of that start sizes the range the
//! crossing is expected in, and the rounds that follow close in on it. A format
//! whose measurements do not fit that picture is handed to the staged search.

use smart_compress_codecs::{Codec, EncoderSettings};

use super::{
    Optimizer, codec_name,
    guards::TempUsageGuard,
    plan::stage_plan,
    search::{LOSSY, describe_inversions, inversions},
    stages::SearchContext,
};
use crate::{
    candidate::{CandidateReport, CandidateStatus},
    config::SearchMode,
    decision::{AcceptedCandidate, DecisionPolicy},
    image_probe::{ContentClass, ImageProbe},
    report::ImageReport,
};

/// What the search assumes of one format near the quality floors.
struct Model {
    /// Quality at which the score reaches 80.
    at_80: f32,
    /// Quality gained per ten points of target score.
    per_ten_points: f32,
    /// Score points gained per quality step.
    score_per_step: f32,
    /// Growth of the file size per quality step, as a natural logarithm.
    log_size_per_step: f32,
}

/// Fitted by least squares on SSIMULACRA2 curves of the ten photos in
/// `samples/photos`, each encoded at every quality near the crossings of the
/// scores 70, 80 and 90, with the final efforts of the balanced speed
/// (`cjpegli --progressive_level=2`, `cwebp -m 4`, `avifenc --speed 6`,
/// `cjxl -e 5`). A crossing is the lowest quality that scores the target; only
/// those at or under quality 92 enter the fit (20 to 26 per format).
///
/// The error of the start, leave-one-photo-out, is on average 8.3, 7.1, 6.0
/// and 5.2 quality steps for JPEG, WebP, AVIF and JPEG XL; on twelve other
/// stock photos it is 7.4, 4.9, 4.1 and 4.8. Image features were tried and
/// left out: over all 22 photos none of the unique colour ratio, edge density,
/// flat region ratio, megapixels or source bits per pixel lowered the mean error
/// for all four formats. The best, the unique colour ratio, lowers it by 2.1,
/// 0.6 and 0.5 steps for JPEG, AVIF and JPEG XL and raises it by 0.3 for WebP.
/// The slopes are medians of the curves around the crossings.
fn model(codec: Codec) -> Model {
    match codec {
        Codec::Jpeg => Model {
            at_80: 64.9,
            per_ten_points: 20.5,
            score_per_step: 0.63,
            log_size_per_step: 0.017,
        },
        Codec::Webp => Model {
            at_80: 85.5,
            per_ten_points: 13.2,
            score_per_step: 0.96,
            log_size_per_step: 0.048,
        },
        Codec::Avif => Model {
            at_80: 64.2,
            per_ten_points: 15.9,
            score_per_step: 0.60,
            log_size_per_step: 0.032,
        },
        Codec::Jxl => Model {
            at_80: 75.9,
            per_ten_points: 13.7,
            score_per_step: 0.61,
            log_size_per_step: 0.031,
        },
        Codec::Png => unreachable!("PNG has no quality setting"),
    }
}

/// The quality a format is expected to reach `target` at, no higher than `ceiling`.
pub(super) fn predicted_quality(codec: Codec, target: f32, ceiling: u8) -> u8 {
    let model = model(codec);
    let quality = model.at_80 + model.per_ten_points * (target - 80.0) / 10.0;
    quality.round().clamp(1.0, f32::from(ceiling)) as u8
}

/// What an evaluated quality says about the floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Outcome {
    Passed,
    Failed,
    /// Too large to be scored: nothing at or above this quality can win.
    Unscored,
}

/// A quality that was evaluated and what came of it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Reading {
    pub(super) quality: u8,
    pub(super) score: Option<f32>,
    pub(super) size: Option<u64>,
    pub(super) outcome: Outcome,
}

/// The readings of one format's evaluated qualities, lowest first. A candidate
/// that failed to run or was declined has none.
pub(super) fn readings(candidates: &[CandidateReport], codec: Codec) -> Vec<Reading> {
    let mut found = candidates
        .iter()
        .filter(|candidate| candidate.codec == codec)
        .filter_map(|candidate| {
            let EncoderSettings::Quality { quality, .. } = candidate.settings else {
                return None;
            };
            let outcome = match &candidate.status {
                _ if candidate.failed_quality() => Outcome::Failed,
                CandidateStatus::Accepted => Outcome::Passed,
                CandidateStatus::Rejected(_) => Outcome::Unscored,
                CandidateStatus::Skipped(_) | CandidateStatus::Failed(_) => return None,
            };
            Some(Reading {
                quality,
                score: candidate.quality.map(|score| score.overall),
                size: candidate.size_bytes,
                outcome,
            })
        })
        .collect::<Vec<_>>();
    found.sort_by_key(|reading| reading.quality);
    found
}

/// Qualities the crossing is expected within, centred on the start corrected
/// by its measured error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ErrorRange {
    pub(super) centre: f32,
    pub(super) half_width: u32,
}

/// Half-width of the range per whole point of the start's error, in quality steps.
const RANGE_STEPS: u32 = 3;
/// Times the range may double before the prediction is given up.
const MAX_WIDENINGS: u8 = 2;
/// A start further than this many points from the target says the prediction
/// does not apply to this image, or that the metric is not the one it was fitted on.
const MAX_START_ERROR: f32 = 15.0;
/// Error assumed for a start that could not be scored because its file was too large.
const UNSCORED_ERROR: f32 = 4.0;
/// A rise in score per quality step under this is no rise at all.
const MIN_SCORE_PER_STEP: f32 = 0.05;
/// Rounds after the start before a format is handed to the staged search.
const MAX_ROUNDS: usize = 6;
/// Qualities probed per round and format, nearest the estimated crossing.
const PROBES_PER_ROUND: usize = 3;
/// How much larger than the smallest file that could be reached a format's
/// smallest expected file may be and still be probed: room for the allowance
/// that compatibility gets over size, at most 10%.
const SIZE_MARGIN: f64 = 1.1;
/// Half-width of the expected size, as a natural logarithm, per quality step
/// the estimate lies from the nearest measured size. Near the floors the
/// logarithm of a file size grows by about 0.01 to 0.07 per quality step, by
/// different amounts for different formats and images.
const SIZE_BAND_PER_STEP: f64 = 0.1;
/// The same between two measured sizes; interpolating them was within 5% of
/// the measured size at the quality in between for 90% of the pairs of
/// qualities up to 12 steps apart.
const SIZE_BAND_BRACKETED: f64 = 0.05;

impl ErrorRange {
    fn after(start: &Reading, target: f32, score_per_step: f32) -> Self {
        let error = start.score.map_or(UNSCORED_ERROR, |score| score - target);
        Self {
            centre: f32::from(start.quality) - error / score_per_step,
            half_width: error.abs().ceil().max(1.0) as u32 * RANGE_STEPS,
        }
    }

    fn contains(&self, quality: i32) -> bool {
        let distance = (quality as f32 - self.centre).abs();
        distance <= self.half_width as f32
    }
}

/// The bytes a format is expected to reach at its crossing, at the least and at most.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct SizeBand {
    low: f64,
    high: f64,
}

impl SizeBand {
    fn around(size: f64, log_half_width: f64) -> Self {
        Self {
            low: size * (-log_half_width).exp(),
            high: size * log_half_width.exp(),
        }
    }
}

/// What a format's search does next.
#[derive(Debug, PartialEq)]
pub(super) enum Step {
    /// Probe these qualities; the format should reach about this many bytes at
    /// the crossing.
    Probe {
        qualities: Vec<u8>,
        expected_size: Option<SizeBand>,
    },
    /// The crossing is known, or the format cannot win.
    Done,
    /// The staged search takes the format over, for this reason.
    Fallback(String),
}

/// The search of one format.
pub(super) struct Track {
    pub(super) codec: Codec,
    range: Option<ErrorRange>,
    widenings: u8,
}

impl Track {
    pub(super) fn new(codec: Codec) -> Self {
        Self {
            codec,
            range: None,
            widenings: 0,
        }
    }

    #[cfg(test)]
    pub(super) fn range(&self) -> Option<ErrorRange> {
        self.range
    }

    /// Plans the next probes of the format from everything evaluated so far.
    pub(super) fn step(
        &mut self,
        candidates: &[CandidateReport],
        target: f32,
        ceiling: u8,
        best: Option<&AcceptedCandidate>,
        policy: &DecisionPolicy,
    ) -> Step {
        let codec = self.codec;
        if let Some(failed) = candidates.iter().find(|candidate| {
            candidate.codec == codec && matches!(candidate.status, CandidateStatus::Failed(_))
        }) {
            return Step::Fallback(format!(
                "a candidate failed before it could be judged ({})",
                failed.settings.label()
            ));
        }
        // A format whose evaluated points are inconsistent is not searched
        // by prediction, whatever range is left.
        let found = inversions(candidates, codec)
            .into_iter()
            .filter(|inversion| {
                inversion
                    .lower_size
                    .is_none_or(|size| policy.candidate_can_beat(size, best, codec))
            })
            .collect::<Vec<_>>();
        if !found.is_empty() {
            return Step::Fallback(describe_inversions(codec, &found));
        }
        let readings = readings(candidates, codec);
        let Some(start) = readings.first() else {
            return Step::Fallback("no quality was judged".to_string());
        };
        if self.range.is_none() {
            if let Some(score) = start.score
                && (score - target).abs() > MAX_START_ERROR
            {
                return Step::Fallback(format!(
                    "the first probe, q{}, scored {score:.1} against a floor of {target:.1}",
                    start.quality
                ));
            }
            self.range = Some(ErrorRange::after(
                start,
                target,
                model(codec).score_per_step,
            ));
        }
        self.next(&readings, target, ceiling, best, policy)
    }

    fn next(
        &mut self,
        readings: &[Reading],
        target: f32,
        ceiling: u8,
        best: Option<&AcceptedCandidate>,
        policy: &DecisionPolicy,
    ) -> Step {
        let model = model(self.codec);
        let first = |outcome| readings.iter().find(|reading| reading.outcome == outcome);
        let passed = first(Outcome::Passed);
        let failed = readings
            .iter()
            .rev()
            .find(|reading| reading.outcome == Outcome::Failed);
        let capped = first(Outcome::Unscored);
        // File size only grows with quality: a format whose highest failing
        // quality is already no smaller than the best cannot win above it.
        if let Some(size) = failed.and_then(|reading| reading.size)
            && !policy.candidate_can_beat(size, best, self.codec)
        {
            return Step::Done;
        }
        if let (Some(low), Some(high)) = (failed, passed)
            && high.quality <= low.quality
        {
            return Step::Fallback(format!(
                "non-monotonic behaviour detected for {}: q{} passes while q{} fails",
                codec_name(self.codec),
                high.quality,
                low.quality
            ));
        }
        // The qualities still worth a probe: above the highest failure, below
        // the lowest pass and the lowest quality that cannot win.
        let lowest = failed.map_or(1, |reading| i32::from(reading.quality) + 1);
        let top = [
            Some(i32::from(ceiling)),
            passed.map(|reading| i32::from(reading.quality) - 1),
            capped.map(|reading| i32::from(reading.quality) - 1),
        ]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(i32::from(ceiling));
        if lowest > top {
            return Step::Done;
        }
        let estimate = match (failed, passed, capped) {
            (Some(low), Some(high), _) => crossing(low, high, target),
            (Some(low), None, _) => {
                let Some(rise) = rise_of(
                    readings
                        .iter()
                        .filter(|reading| reading.outcome == Outcome::Failed),
                    model.score_per_step,
                ) else {
                    return Step::Fallback("the score does not rise with quality".to_string());
                };
                let deficit = target - low.score.unwrap_or(target);
                (f32::from(low.quality) + deficit / rise).max(f32::from(low.quality) + 2.0)
            }
            (None, Some(high), _) => {
                let Some(rise) = rise_of(
                    readings
                        .iter()
                        .filter(|reading| reading.outcome == Outcome::Passed),
                    model.score_per_step,
                ) else {
                    return Step::Fallback("the score does not rise with quality".to_string());
                };
                let surplus = high.score.map_or(UNSCORED_ERROR, |score| score - target);
                (f32::from(high.quality) - surplus / rise).min(f32::from(high.quality) - 1.0)
            }
            (None, None, Some(capped)) => {
                f32::from(capped.quality) - UNSCORED_ERROR / model.score_per_step
            }
            (None, None, None) => return Step::Fallback("no quality was judged".to_string()),
        };
        let centre = estimate.ceil() as i32;
        // Without a bracket the estimate may leave the range the start allowed.
        if (failed.is_none() || passed.is_none())
            && let Some(range) = self.range.as_mut()
            && !range.contains(centre)
        {
            self.widenings += 1;
            if self.widenings > MAX_WIDENINGS {
                return Step::Fallback(
                    "the crossing lies far from the predicted start".to_string(),
                );
            }
            range.half_width *= 2;
        }
        // The untried qualities nearest the estimate, which are on one side of
        // it when the estimate lies beside what is left to try.
        let mut qualities = (lowest..=top)
            .filter(|quality| readings.iter().all(|r| i32::from(r.quality) != *quality))
            .collect::<Vec<_>>();
        qualities.sort_by(|a, b| {
            (*a as f32 - estimate)
                .abs()
                .total_cmp(&(*b as f32 - estimate).abs())
        });
        qualities.truncate(PROBES_PER_ROUND);
        if qualities.is_empty() {
            return Step::Fallback("no untried quality is left in the range".to_string());
        }
        qualities.sort_unstable();
        let qualities = qualities.into_iter().map(|quality| quality as u8).collect();
        Step::Probe {
            qualities,
            expected_size: expected_size(
                readings,
                estimate,
                failed.zip(passed),
                model.log_size_per_step,
            ),
        }
    }
}

/// Quality at which the score crosses `target`, between a failing and a passing reading.
fn crossing(low: &Reading, high: &Reading, target: f32) -> f32 {
    let (low_q, high_q) = (f32::from(low.quality), f32::from(high.quality));
    match (low.score, high.score) {
        (Some(below), Some(above)) if above > below => {
            (low_q + (target - below) * (high_q - low_q) / (above - below)).clamp(low_q, high_q)
        }
        _ => (low_q + high_q) / 2.0,
    }
}

/// Score points per quality step over the widest span of scored readings on
/// one side of the crossing, else the model's; `None` when they show the score
/// not rising.
fn rise_of<'a>(side: impl Iterator<Item = &'a Reading>, model_rise: f32) -> Option<f32> {
    let scored = side
        .filter_map(|reading| Some((f32::from(reading.quality), reading.score?)))
        .collect::<Vec<_>>();
    let (Some(low), Some(high)) = (
        scored.iter().min_by(|a, b| a.0.total_cmp(&b.0)),
        scored.iter().max_by(|a, b| a.0.total_cmp(&b.0)),
    ) else {
        return Some(model_rise);
    };
    if high.0 <= low.0 {
        return Some(model_rise);
    }
    let rise = (high.1 - low.1) / (high.0 - low.0);
    (rise > MIN_SCORE_PER_STEP).then_some(rise)
}

/// File size at `quality`: interpolated between the failing and the passing
/// reading when both have a size, else carried over from the reading with a
/// size nearest to it.
fn expected_size(
    readings: &[Reading],
    quality: f32,
    bracket: Option<(&Reading, &Reading)>,
    log_size_per_step: f32,
) -> Option<SizeBand> {
    if let Some((low, high)) = bracket
        && let (Some(low_size), Some(high_size)) = (low.size, high.size)
    {
        let (low_log, high_log) = ((low_size as f64).ln(), (high_size as f64).ln());
        let span = f32::from(high.quality) - f32::from(low.quality);
        let position = f64::from((quality - f32::from(low.quality)) / span.max(1.0));
        let size = (low_log + (high_log - low_log) * position).exp();
        return Some(SizeBand::around(size, SIZE_BAND_BRACKETED));
    }
    let nearest = readings
        .iter()
        .filter(|reading| reading.size.is_some())
        .min_by(|a, b| {
            (f32::from(a.quality) - quality)
                .abs()
                .total_cmp(&(f32::from(b.quality) - quality).abs())
        })?;
    let steps = quality - f32::from(nearest.quality);
    let size = nearest.size? as f64 * f64::from(log_size_per_step * steps).exp();
    Some(SizeBand::around(
        size,
        SIZE_BAND_PER_STEP * f64::from(steps.abs()),
    ))
}

impl Optimizer {
    /// Whether the predicted search runs for an image: it needs candidates in
    /// parallel stages, a refinement to stand in for, and the photographs and
    /// SSIMULACRA2 scale its start was fitted on. The score of a graphic often
    /// does not rise steadily with quality, which the search is built on.
    pub(super) fn searches_predicted(&self, parallelism: usize, probe: &ImageProbe) -> bool {
        self.config.search == SearchMode::Predicted
            && parallelism > 1
            && !self.config.verbose
            && self.config.speed.refinement_rounds() > 0
            && probe.content_class == ContentClass::Photo
            && self.scores_ssimulacra2(probe.width, probe.height)
    }

    /// The rounds that follow the predicted starts, then the staged search for
    /// the formats that were handed back.
    pub(super) fn refine_predicted(
        &self,
        ctx: &SearchContext<'_>,
        report: &mut ImageReport,
        best: &mut Option<AcceptedCandidate>,
        temp_usage: &mut TempUsageGuard,
    ) -> Option<String> {
        let target = ctx.policy.min_quality - self.config.quality_slack;
        let ceiling = self.config.speed.qualities()[0];
        let request = ctx.refinement_request();
        let mut tracks = LOSSY
            .into_iter()
            .filter(|codec| {
                report.candidates.iter().any(|candidate| {
                    candidate.codec == *codec
                        && matches!(candidate.settings, EncoderSettings::Quality { .. })
                })
            })
            .map(Track::new)
            .collect::<Vec<_>>();
        let mut handed_back: Vec<(Codec, String)> = Vec::new();
        for round in 0..=MAX_ROUNDS {
            let mut plans = Vec::new();
            for mut track in std::mem::take(&mut tracks) {
                match track.step(
                    &report.candidates,
                    target,
                    ceiling,
                    best.as_ref(),
                    ctx.policy,
                ) {
                    Step::Probe {
                        qualities,
                        expected_size,
                    } => {
                        plans.push((track, qualities, expected_size));
                    }
                    Step::Done => {}
                    Step::Fallback(reason) => handed_back.push((track.codec, reason)),
                }
            }
            // A format whose smallest expected file cannot beat the best so far,
            // or is well above the largest that another format may reach, is
            // left out.
            let front = plans
                .iter()
                .filter_map(|(_, _, size)| size.map(|size| size.high))
                .fold(f64::INFINITY, f64::min);
            plans.retain(|(track, _, size)| {
                size.is_none_or(|size| {
                    size.low <= front * SIZE_MARGIN
                        && ctx.policy.candidate_can_beat(
                            size.low as u64,
                            best.as_ref(),
                            track.codec,
                        )
                })
            });
            if round == MAX_ROUNDS {
                handed_back.extend(plans.iter().map(|(track, _, _)| {
                    (
                        track.codec,
                        format!("the range did not close in {MAX_ROUNDS} rounds"),
                    )
                }));
                break;
            }
            if plans.is_empty() {
                break;
            }
            let points = plans
                .iter()
                .flat_map(|(track, qualities, _)| {
                    qualities.iter().map(|quality| (track.codec, *quality))
                })
                .collect::<Vec<_>>();
            tracks = plans.into_iter().map(|(track, _, _)| track).collect();
            // A probe is scored even if a smaller candidate exists: the readings
            // are what the next round plans from.
            if let Some(reason) = self.run_points(&request, points, report, best, false) {
                return Some(reason);
            }
        }
        if handed_back.is_empty() {
            return None;
        }
        for (codec, reason) in &handed_back {
            report.warnings.push(format!(
                "the predicted search gave up on {}: {reason}; searching this format with the staged search",
                codec_name(*codec)
            ));
        }
        let codecs = handed_back
            .iter()
            .map(|(codec, _)| *codec)
            .collect::<Vec<_>>();
        let speed = self.config.speed;
        let grid = codecs
            .iter()
            .flat_map(|codec| {
                speed.qualities().iter().map(move |quality| {
                    (
                        *codec,
                        EncoderSettings::Quality {
                            quality: *quality,
                            effort: speed.probe_effort(*codec),
                        },
                    )
                })
            })
            .collect::<Vec<_>>();
        let ctx = SearchContext {
            candidate_total: report.candidates.len() + grid.len(),
            ..*ctx
        };
        let stop = self.run_staged(&ctx, stage_plan(grid, false), report, best, temp_usage);
        if stop.is_some() {
            return stop;
        }
        self.refine_quality_search(ctx.refinement_request(), report, best, &codecs)
    }
}
