use std::{
    path::{Path, PathBuf},
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

use anyhow::Result;
use smart_compress_codecs::{
    Codec, EncodeError, EncodeJob, EncodedArtifact, EncoderSettings, MetadataPayload,
    ProcessFailure,
};
use smart_compress_metrics::QualityScore;

use super::{
    CandidateEvaluationRequest, Optimizer, ScoreDecodedRequest,
    guards::{ActiveCandidate, CandidateProgressGuard},
};
use crate::{
    cache::{CachedCandidateMetadata, file_hash},
    candidate::{
        ActualScoreBackend, CacheHitKind, CandidateInfluence, CandidateReport, CandidateStatus,
        QualitySource,
    },
    decision::{AcceptedCandidate, DecisionPolicy},
    memory,
    report::ImageReport,
};

/// Pixels one thread of an external encoder is worth giving; an image with
/// fewer pixels than this gets one thread however many are free. Measured on a
/// 7.7 MP photo, one candidate at a time, with every encoder given 1, 2, 4 and
/// 7 threads: mean encode time 1109, 903, 717 and 653 ms, so the extra threads
/// are worth 61%, 39% and 24% of a thread each. On a 0.26 MP image it made no
/// difference (162, 151, 146 and 146 ms). Three megapixels a thread stops
/// short of the point where a thread is worth less than half a thread.
const PIXELS_PER_ENCODER_THREAD: u64 = 3_000_000;

fn process_failure_summary(
    error: &anyhow::Error,
) -> Option<crate::candidate::ProcessFailureSummary> {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<ProcessFailure>())
        .map(crate::candidate::ProcessFailureSummary::from)
}

impl Optimizer {
    /// Reserves the memory a candidate needs, waiting while the candidates in
    /// flight hold it all, and then evaluates it. `None` when the image ran
    /// out of time or the run was cancelled during the wait. A candidate
    /// larger than the limit is granted the whole of it and runs alone.
    pub(super) fn score_candidate_report(
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
}

pub(super) fn candidate_path(dir: &Path, codec: Codec, settings: &EncoderSettings) -> PathBuf {
    let label = settings.label().replace(['/', ':'], "-");
    dir.join(format!("{codec:?}-{label}.{}", codec.extension()).to_ascii_lowercase())
}

/// How a candidate that the encoder did not produce is recorded: one it
/// declined to make has no verdict and is merely skipped; any other error is
/// a failure.
pub(super) fn encode_failure(error: &anyhow::Error) -> (CandidateStatus, CandidateInfluence) {
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

/// Appends an evaluated candidate and promotes it when it beats the current best.
pub(super) fn record_candidate(
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

struct StoreQualityCacheRequest<'a> {
    cache_source_hash: Option<&'a str>,
    codec: Codec,
    settings: &'a EncoderSettings,
    has_alpha: bool,
    size_bytes: u64,
    quality: QualityScore,
    candidate: &'a CandidateReport,
}
