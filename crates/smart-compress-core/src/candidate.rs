use std::{collections::BTreeMap, path::PathBuf};

use serde::{Deserialize, Deserializer, Serialize};
use smart_compress_codecs::{Codec, EncodedArtifact, EncoderSettings, ProcessFailure};
use smart_compress_metrics::QualityScore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CandidateStatus {
    Accepted,
    /// Judged and turned down: scored below the quality floor, or too large to
    /// be worth scoring. Only a rejection with a `quality` is a verdict on quality.
    Rejected(String),
    /// Not evaluated, or no longer of interest.
    Skipped(String),
    /// An encoder, decoder or metric failed (a tool error, a timeout, a bad
    /// output), so the candidate has no verdict at all.
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ActualScoreBackend {
    Cpu,
    Gpu,
    Ssimulacra2Inprocess,
    CpuFallback { reason: String },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QualitySource {
    #[default]
    Fresh,
    ScoreCache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheHitKind {
    EncodedArtifact,
    ScoreMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandidateInfluence {
    FirstPassing,
    ImprovedWinner,
    Rejected,
    Skipped,
    Failed,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CandidateByteCounters {
    pub source_upload_bytes: Option<u64>,
    pub candidate_upload_bytes: Option<u64>,
    pub temporary_output_bytes: Option<u64>,
    pub decoded_buffer_bytes: Option<u64>,
    pub peak_estimated_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessFailureSummary {
    pub executable: PathBuf,
    pub kind: String,
    pub exit_code: Option<i32>,
    pub message: String,
    pub elapsed_ms: u64,
    pub termination_requested: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuVerificationReport {
    pub tolerance: f32,
    pub cpu_overall: f32,
    pub gpu_overall: f32,
    pub overall_delta: f32,
    /// The per-component differences, by component name. Older reports wrote
    /// them as a list of name and value pairs.
    #[serde(deserialize_with = "deserialize_component_deltas")]
    pub component_deltas: BTreeMap<String, f32>,
    pub passed: bool,
}

fn deserialize_component_deltas<'de, D>(deserializer: D) -> Result<BTreeMap<String, f32>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Deltas {
        Map(BTreeMap<String, f32>),
        Pairs(Vec<(String, f32)>),
    }

    Ok(match Deltas::deserialize(deserializer)? {
        Deltas::Map(map) => map,
        Deltas::Pairs(pairs) => pairs.into_iter().collect(),
    })
}

impl From<&ProcessFailure> for ProcessFailureSummary {
    fn from(failure: &ProcessFailure) -> Self {
        Self {
            executable: failure
                .executable
                .file_name()
                .map(PathBuf::from)
                .unwrap_or_default(),
            kind: format!("{:?}", failure.kind),
            exit_code: failure.exit_code,
            message: failure.message.clone(),
            elapsed_ms: failure.elapsed_ms,
            termination_requested: failure.termination_requested,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateReport {
    pub codec: Codec,
    pub settings: EncoderSettings,
    pub path: Option<PathBuf>,
    pub size_bytes: Option<u64>,
    pub encode_ms: Option<u64>,
    pub decode_ms: Option<u64>,
    #[serde(default)]
    pub decode_process_ms: Option<u64>,
    #[serde(default)]
    pub decode_io_ms: Option<u64>,
    #[serde(default)]
    pub preprocess_ms: Option<u64>,
    #[serde(default)]
    pub score_ms: Option<u64>,
    #[serde(default)]
    pub cache_ms: Option<u64>,
    #[serde(default)]
    pub external_metric_ms: Option<u64>,
    #[serde(default)]
    pub gpu_upload_ms: Option<u64>,
    #[serde(default)]
    pub gpu_transform_ms: Option<u64>,
    #[serde(default)]
    pub gpu_compute_ms: Option<u64>,
    #[serde(default)]
    pub gpu_readback_ms: Option<u64>,
    /// GPU-side pass time from timestamp queries, in nanoseconds; unlike the
    /// host-wall `gpu_*_ms` values it is absent when the adapter lacks them.
    #[serde(default)]
    pub gpu_timestamp_ns: Option<u64>,
    #[serde(default)]
    pub total_ms: Option<u64>,
    pub quality: Option<QualityScore>,
    pub backend: Option<String>,
    #[serde(default)]
    pub score_backend: Option<ActualScoreBackend>,
    #[serde(default)]
    pub quality_source: QualitySource,
    #[serde(default)]
    pub cache_hit_kind: Option<CacheHitKind>,
    #[serde(default)]
    pub process_failure: Option<ProcessFailureSummary>,
    #[serde(default)]
    pub gpu_verification: Option<GpuVerificationReport>,
    #[serde(default)]
    pub influence: Option<CandidateInfluence>,
    #[serde(default)]
    pub bytes: CandidateByteCounters,
    pub status: CandidateStatus,
    pub warnings: Vec<String>,
}

impl CandidateReport {
    /// Whether the candidate was scored and fell below the quality floor.
    pub fn failed_quality(&self) -> bool {
        matches!(self.status, CandidateStatus::Rejected(_)) && self.quality.is_some()
    }

    /// A planned candidate that was skipped before any encoding happened.
    pub fn unevaluated(codec: Codec, settings: EncoderSettings, reason: String) -> Self {
        Self {
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
            total_ms: None,
            quality: None,
            backend: None,
            score_backend: None,
            quality_source: QualitySource::Fresh,
            cache_hit_kind: None,
            process_failure: None,
            gpu_verification: None,
            influence: Some(CandidateInfluence::Skipped),
            bytes: CandidateByteCounters::default(),
            status: CandidateStatus::Skipped(reason),
            warnings: Vec::new(),
        }
    }

    pub fn from_artifact(artifact: EncodedArtifact) -> Self {
        Self {
            codec: artifact.codec,
            settings: artifact.settings,
            path: Some(artifact.path),
            size_bytes: Some(artifact.size_bytes),
            encode_ms: Some(artifact.encode_ms),
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
            quality: None,
            backend: Some(artifact.backend),
            score_backend: None,
            quality_source: QualitySource::Fresh,
            cache_hit_kind: None,
            process_failure: None,
            gpu_verification: None,
            influence: None,
            bytes: CandidateByteCounters {
                temporary_output_bytes: Some(artifact.size_bytes),
                ..CandidateByteCounters::default()
            },
            status: CandidateStatus::Skipped("not evaluated yet".to_string()),
            warnings: artifact.warnings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_fields_default_when_deserializing_older_reports() {
        let report = CandidateReport {
            codec: Codec::Png,
            settings: EncoderSettings::Lossless,
            path: None,
            size_bytes: Some(12),
            encode_ms: Some(3),
            decode_ms: Some(2),
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
            backend: Some("test".to_string()),
            score_backend: None,
            quality_source: QualitySource::Fresh,
            cache_hit_kind: None,
            process_failure: None,
            gpu_verification: None,
            influence: None,
            bytes: CandidateByteCounters::default(),
            status: CandidateStatus::Accepted,
            warnings: Vec::new(),
        };
        let mut value = serde_json::to_value(report).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("preprocess_ms");
        object.remove("score_ms");
        object.remove("total_ms");
        object.remove("score_backend");
        object.remove("decode_process_ms");
        object.remove("decode_io_ms");
        object.remove("cache_ms");
        object.remove("external_metric_ms");
        object.remove("gpu_upload_ms");
        object.remove("gpu_transform_ms");
        object.remove("gpu_compute_ms");
        object.remove("gpu_readback_ms");
        object.remove("gpu_timestamp_ns");
        object.remove("quality_source");
        object.remove("cache_hit_kind");
        object.remove("process_failure");
        object.remove("gpu_verification");
        object.remove("influence");
        object.remove("bytes");

        let decoded: CandidateReport = serde_json::from_value(value).unwrap();

        assert_eq!(decoded.preprocess_ms, None);
        assert_eq!(decoded.score_ms, None);
        assert_eq!(decoded.total_ms, None);
        assert_eq!(decoded.score_backend, None);
        assert_eq!(decoded.quality_source, QualitySource::Fresh);
        assert_eq!(decoded.cache_hit_kind, None);
    }

    #[test]
    fn gpu_verification_deltas_are_a_map_and_older_pairs_still_read() {
        let report = GpuVerificationReport {
            tolerance: 0.25,
            cpu_overall: 90.0,
            gpu_overall: 90.1,
            overall_delta: 0.1,
            component_deltas: BTreeMap::from([("edge".to_string(), 0.2), ("rgb".to_string(), 0.1)]),
            passed: true,
        };
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["component_deltas"]["rgb"], 0.1_f32);
        assert_eq!(value["component_deltas"]["edge"], 0.2_f32);

        let mut older = value;
        older["component_deltas"] = serde_json::json!([["rgb", 0.1], ["edge", 0.2]]);
        let read: GpuVerificationReport = serde_json::from_value(older).unwrap();
        assert_eq!(read.component_deltas, report.component_deltas);
    }

    #[test]
    fn a_score_read_from_an_older_report_calls_the_rgb_agreement_ssim() {
        let score: QualityScore =
            serde_json::from_str(r#"{"ssim": 91.5, "edge": 88.0, "overall": 88.0}"#).unwrap();
        assert_eq!(score.rgb, 91.5);
        let written = serde_json::to_value(score).unwrap();
        assert_eq!(written["rgb"], 91.5_f32);
        assert!(written.get("ssim").is_none());
    }

    #[test]
    fn cpu_fallback_serializes_with_its_reason() {
        let backend = ActualScoreBackend::CpuFallback {
            reason: "GPU readback failed".to_string(),
        };

        let value = serde_json::to_value(backend).unwrap();

        assert_eq!(value["cpu-fallback"]["reason"], "GPU readback failed");
    }
}
