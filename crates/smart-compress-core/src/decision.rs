use smart_compress_codecs::Codec;
use smart_compress_metrics::QualityScore;

use crate::{candidate::CandidateReport, config::TargetPolicy};

#[derive(Debug, Clone)]
pub struct DecisionPolicy {
    pub min_quality: f32,
    pub quality_slack: f32,
    pub allow_larger: bool,
    pub original_size: u64,
    pub prefer_compatibility_delta: f32,
    pub target: TargetPolicy,
}

#[derive(Debug, Clone)]
pub struct AcceptedCandidate {
    pub codec: Codec,
    pub size_bytes: u64,
    pub quality: QualityScore,
}

impl DecisionPolicy {
    pub fn passes_quality(&self, quality: QualityScore) -> bool {
        quality.overall + self.quality_slack >= self.min_quality
    }

    pub fn can_accept_size(&self, size_bytes: u64) -> bool {
        self.allow_larger || size_bytes < self.original_size
    }

    pub fn candidate_can_beat(
        &self,
        candidate_size: u64,
        best: Option<&AcceptedCandidate>,
        codec: Codec,
    ) -> bool {
        let Some(best) = best else {
            return true;
        };
        // At an equal size `choose` still prefers the better-ranked codec, under
        // every target, so such a candidate must not be pruned.
        candidate_size < best.size_bytes
            || (codec.compatibility_rank() > best.codec.compatibility_rank()
                && self.within_allowance(candidate_size, best.size_bytes))
    }

    /// Compatibility allowance above the smallest candidate; none when only size counts.
    fn allowance(&self) -> f32 {
        if matches!(self.target, TargetPolicy::Smallest) {
            0.0
        } else {
            self.prefer_compatibility_delta
        }
    }

    /// Whether `size` is within the allowance of `smallest`. `choose` and
    /// `candidate_can_beat` share this one comparison, so a candidate exactly at
    /// the allowance is never pruned and then chosen anyway.
    fn within_allowance(&self, size: u64, smallest: u64) -> bool {
        size.abs_diff(smallest) as f32 / smallest.max(1) as f32 <= self.allowance()
    }

    /// About the largest file that could still win: smaller than the original
    /// unless larger outputs are allowed, and within reach of the best accepted
    /// candidate. `None` while nothing bounds it.
    pub fn size_ceiling(&self, best: Option<&AcceptedCandidate>) -> Option<u64> {
        let from_original = (!self.allow_larger).then_some(self.original_size);
        let from_best = best.map(|best| {
            (best.size_bytes as f64 * (1.0 + f64::from(self.allowance()))).round() as u64
        });
        from_original.into_iter().chain(from_best).min()
    }

    pub fn choose<'a>(&self, candidates: &'a [CandidateReport]) -> Option<&'a CandidateReport> {
        let acceptable = candidates
            .iter()
            .filter_map(|candidate| {
                let size = candidate.size_bytes?;
                let quality = candidate.quality?;
                (self.can_accept_size(size) && self.passes_quality(quality))
                    .then_some((size, candidate))
            })
            .collect::<Vec<_>>();
        let smallest = acceptable.iter().map(|(size, _)| *size).min()?;
        // The compatibility allowance is measured against the smallest
        // candidate only, so it cannot chain into a much larger winner.
        acceptable
            .into_iter()
            .filter(|(size, _)| self.within_allowance(*size, smallest))
            .min_by(|(left_size, left), (right_size, right)| {
                right
                    .codec
                    .compatibility_rank()
                    .cmp(&left.codec.compatibility_rank())
                    .then(left_size.cmp(right_size))
            })
            .map(|(_, candidate)| candidate)
    }
}

#[cfg(test)]
mod tests {
    use smart_compress_codecs::{Codec, EncoderSettings};

    use super::*;
    use crate::candidate::{CandidateReport, CandidateStatus};

    fn accepted(codec: Codec, size: u64) -> CandidateReport {
        CandidateReport {
            codec,
            settings: EncoderSettings::Lossless,
            path: None,
            size_bytes: Some(size),
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
            total_ms: None,
            quality: Some(QualityScore::lossless()),
            backend: None,
            score_backend: None,
            quality_source: Default::default(),
            cache_hit_kind: None,
            process_failure: None,
            gpu_verification: None,
            influence: None,
            bytes: Default::default(),
            status: CandidateStatus::Accepted,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn floors_and_size_limits_are_exact_at_their_boundaries() {
        let mut policy = DecisionPolicy {
            min_quality: 80.0,
            quality_slack: 0.0,
            allow_larger: false,
            original_size: 1_000,
            prefer_compatibility_delta: 0.05,
            target: TargetPolicy::Web,
        };
        let scored = |overall: f32| QualityScore {
            overall,
            ..QualityScore::lossless()
        };
        assert!(policy.passes_quality(scored(80.0)));
        assert!(!policy.passes_quality(scored(79.99)));
        policy.quality_slack = 0.5;
        assert!(policy.passes_quality(scored(79.5)));
        assert!(!policy.passes_quality(scored(79.49)));

        assert!(policy.can_accept_size(999));
        assert!(!policy.can_accept_size(1_000));
        policy.allow_larger = true;
        assert!(policy.can_accept_size(5_000));

        // The compatibility allowance is inclusive at exactly five percent.
        let at_delta = vec![accepted(Codec::Avif, 1_000), accepted(Codec::Jpeg, 1_050)];
        assert_eq!(policy.choose(&at_delta).unwrap().codec, Codec::Jpeg);
        let past_delta = vec![accepted(Codec::Avif, 1_000), accepted(Codec::Jpeg, 1_051)];
        assert_eq!(policy.choose(&past_delta).unwrap().codec, Codec::Avif);
        policy.target = TargetPolicy::Smallest;
        assert_eq!(policy.choose(&at_delta).unwrap().codec, Codec::Avif);
    }

    fn policy_for(target: TargetPolicy, delta: f32) -> DecisionPolicy {
        DecisionPolicy {
            min_quality: 80.0,
            quality_slack: 0.0,
            allow_larger: true,
            original_size: 1_000_000,
            prefer_compatibility_delta: delta,
            target,
        }
    }

    fn best_of(codec: Codec, size_bytes: u64) -> AcceptedCandidate {
        AcceptedCandidate {
            codec,
            size_bytes,
            quality: QualityScore::lossless(),
        }
    }

    /// Every ordering of 0..n.
    fn orderings(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![Vec::new()];
        }
        let mut all = Vec::new();
        for rest in orderings(n - 1) {
            for at in 0..=rest.len() {
                let mut ordering = rest.clone();
                ordering.insert(at, n - 1);
                all.push(ordering);
            }
        }
        all
    }

    /// The search skips a candidate that cannot beat the best accepted so far
    /// and records the rest in whatever order they finish. Whatever the order,
    /// and whether the pruning sees the live best (one at a time) or the best at
    /// the start of its stage (concurrent), `choose` must pick what it would
    /// from the whole set.
    fn pruning_never_changes_the_choice(target: TargetPolicy) {
        let sets: [&[(Codec, u64)]; 7] = [
            // Exactly at the allowance, in both directions.
            &[(Codec::Avif, 1_000), (Codec::Jpeg, 1_050)],
            &[(Codec::Avif, 1_000), (Codec::Jpeg, 1_100)],
            // The allowance must not chain from one runner-up to the next.
            &[(Codec::Jxl, 100), (Codec::Webp, 104), (Codec::Jpeg, 108)],
            // Equal sizes: the better-ranked codec wins.
            &[(Codec::Jxl, 1_000), (Codec::Avif, 1_000)],
            &[
                (Codec::Avif, 1_000),
                (Codec::Webp, 1_000),
                (Codec::Jpeg, 1_000),
            ],
            &[
                (Codec::Jxl, 990),
                (Codec::Avif, 1_000),
                (Codec::Webp, 1_049),
                (Codec::Jpeg, 1_100),
            ],
            &[
                (Codec::Jxl, 500),
                (Codec::Avif, 500),
                (Codec::Webp, 520),
                (Codec::Jpeg, 525),
            ],
        ];
        let rank = |codec: Codec, size| (codec.compatibility_rank(), size);
        for delta in [0.0, 0.05, 0.10] {
            let policy = policy_for(target, delta);
            for set in sets {
                let all = set
                    .iter()
                    .map(|&(codec, size)| accepted(codec, size))
                    .collect::<Vec<_>>();
                let expected = policy
                    .choose(&all)
                    .map(|winner| rank(winner.codec, winner.size_bytes.unwrap()));
                for ordering in orderings(set.len()) {
                    // `split` candidates make up the first stage and see no best;
                    // `None` is the one-at-a-time path.
                    for split in std::iter::once(None).chain((0..=set.len()).map(Some)) {
                        let mut best: Option<AcceptedCandidate> = None;
                        let mut stage_best = None;
                        let mut kept = Vec::new();
                        for (position, &index) in ordering.iter().enumerate() {
                            let (codec, size) = set[index];
                            if split == Some(position) {
                                stage_best = best.clone();
                            }
                            let seen = match split {
                                None => best.as_ref(),
                                Some(split) if position < split => None,
                                Some(_) => stage_best.as_ref(),
                            };
                            if !policy.candidate_can_beat(size, seen, codec) {
                                continue;
                            }
                            if policy.candidate_can_beat(size, best.as_ref(), codec) {
                                best = Some(best_of(codec, size));
                            }
                            kept.push(accepted(codec, size));
                        }
                        let chosen = policy
                            .choose(&kept)
                            .map(|winner| rank(winner.codec, winner.size_bytes.unwrap()));
                        assert_eq!(
                            chosen, expected,
                            "{target:?} delta {delta}: {set:?} recorded as {ordering:?}, split {split:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pruning_keeps_the_choice_under_web() {
        pruning_never_changes_the_choice(TargetPolicy::Web);
    }

    #[test]
    fn pruning_keeps_the_choice_under_archive() {
        pruning_never_changes_the_choice(TargetPolicy::Archive);
    }

    #[test]
    fn pruning_keeps_the_choice_under_smallest() {
        pruning_never_changes_the_choice(TargetPolicy::Smallest);
    }

    #[test]
    fn pruning_keeps_the_choice_under_max_compat() {
        pruning_never_changes_the_choice(TargetPolicy::MaxCompat);
    }

    #[test]
    fn an_equal_size_with_a_better_rank_is_not_pruned_under_any_target() {
        let best = best_of(Codec::Avif, 1_000);
        for target in [
            TargetPolicy::Web,
            TargetPolicy::Archive,
            TargetPolicy::Smallest,
            TargetPolicy::MaxCompat,
        ] {
            let policy = policy_for(target, 0.05);
            assert!(
                policy.candidate_can_beat(1_000, Some(&best), Codec::Jpeg),
                "{target:?}"
            );
            assert!(
                !policy.candidate_can_beat(1_000, Some(&best), Codec::Avif),
                "{target:?}"
            );
            assert!(
                !policy.candidate_can_beat(1_000, Some(&best), Codec::Jxl),
                "{target:?}"
            );
            // Only a target that trades size for compatibility lets a larger file through.
            let allowed = !matches!(target, TargetPolicy::Smallest);
            assert_eq!(
                policy.candidate_can_beat(1_050, Some(&best), Codec::Jpeg),
                allowed
            );
            assert!(!policy.candidate_can_beat(1_051, Some(&best), Codec::Jpeg));
        }
    }

    #[test]
    fn the_size_ceiling_follows_the_original_and_the_best() {
        let mut policy = policy_for(TargetPolicy::Web, 0.05);
        assert_eq!(policy.size_ceiling(None), None);
        policy.allow_larger = false;
        assert_eq!(policy.size_ceiling(None), Some(1_000_000));
        let best = best_of(Codec::Avif, 1_000);
        assert_eq!(policy.size_ceiling(Some(&best)), Some(1_050));
        policy.target = TargetPolicy::Smallest;
        assert_eq!(policy.size_ceiling(Some(&best)), Some(1_000));
        policy.allow_larger = true;
        assert_eq!(policy.size_ceiling(Some(&best)), Some(1_000));
    }

    #[test]
    fn prefers_compatibility_inside_delta() {
        let policy = DecisionPolicy {
            min_quality: 90.0,
            quality_slack: 0.0,
            allow_larger: true,
            original_size: 1_000,
            prefer_compatibility_delta: 0.05,
            target: TargetPolicy::Web,
        };
        let candidates = vec![accepted(Codec::Avif, 100), accepted(Codec::Webp, 103)];
        let winner = policy.choose(&candidates).unwrap();
        assert_eq!(winner.codec, Codec::Webp);

        // The allowance must not chain: JPEG is 8% above the smallest.
        let chained = vec![
            accepted(Codec::Avif, 100),
            accepted(Codec::Webp, 104),
            accepted(Codec::Jpeg, 108),
        ];
        assert_eq!(policy.choose(&chained).unwrap().codec, Codec::Webp);
    }
}
