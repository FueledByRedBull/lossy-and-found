use std::fmt::Write as _;

use anyhow::Context;
use comfy_table::{Cell, Table, presets::UTF8_FULL};
use serde::{Deserialize, Serialize};
use smart_compress_codecs::ToolInventory;

use crate::{
    candidate::{ActualScoreBackend, CandidateReport},
    config::ReportFormat,
    gpu::{GpuReport, GpuWorkEstimate},
    image_probe::ImageProbe,
};

/// The version of the report format this build writes and reads. A report with
/// a newer version is rejected; one without a version is version 1.
pub const REPORT_SCHEMA_VERSION: u32 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunStatus {
    Partial,
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub status: RunStatus,
    pub started_unix_ms: u64,
    pub completed_unix_ms: Option<u64>,
    pub expected_inputs: usize,
    pub completed_inputs: usize,
    pub config_fingerprint: String,
    #[serde(default)]
    pub resume_decision: Option<String>,
}

/// How an image ended. Only `Failed` counts as a failure of the run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ImageOutcome {
    /// A smaller output passed the quality floor and was written.
    Optimized,
    /// Nothing smaller passed, so the original was left as it is.
    Kept,
    /// The output already exists and overwriting is off; nothing was done.
    Skipped,
    /// A dry run: the image was probed and nothing was written.
    DryRun,
    /// The image could not be processed; `errors` says why.
    #[default]
    Failed,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageStageTimings {
    #[serde(default)]
    pub source_open_ms: Option<u64>,
    pub source_decode_ms: Option<u64>,
    pub gpu_preprocess_ms: Option<u64>,
    pub orientation_ms: Option<u64>,
    pub probe_ms: Option<u64>,
    pub normalize_ms: Option<u64>,
    pub cache_lookup_ms: Option<u64>,
    pub external_metric_prepare_ms: Option<u64>,
    pub search_ms: Option<u64>,
    pub output_write_ms: Option<u64>,
    pub total_ms: Option<u64>,
    /// Time this image's work spent waiting for memory or temporary-disk room
    /// held by other work. Summed over its waits, so it can exceed the wall time
    /// the image waited if several candidates waited together.
    pub backpressure_wait_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReportStageTotals {
    pub source_decode_ms: u64,
    pub normalize_ms: u64,
    pub search_ms: u64,
    pub encode_ms: u64,
    pub candidate_decode_ms: u64,
    pub score_ms: u64,
    pub cache_ms: u64,
    pub external_metric_ms: u64,
    pub output_write_ms: u64,
    pub total_ms: u64,
    /// Sum of the candidates' GPU timestamp-query time in nanoseconds; `None`
    /// when no candidate had one.
    #[serde(default)]
    pub gpu_timestamp_ns: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageReport {
    pub input: String,
    #[serde(default)]
    pub input_identity: Option<String>,
    pub output: Option<String>,
    pub original_size: u64,
    pub saved_bytes: Option<i64>,
    pub probe: Option<ImageProbe>,
    pub gpu_work: Option<GpuWorkEstimate>,
    #[serde(default)]
    pub gpu_preprocess_backend: Option<String>,
    pub compute_backend: String,
    #[serde(default)]
    pub scoring_version: String,
    /// Score that gated acceptance: `ssimulacra2` or `builtin` (`psnr` in
    /// reports before schema version 6).
    #[serde(default)]
    pub quality_metric: String,
    #[serde(default)]
    pub min_quality: Option<f32>,
    #[serde(default)]
    pub timings: ImageStageTimings,
    pub candidates: Vec<CandidateReport>,
    pub winner_index: Option<usize>,
    #[serde(default)]
    pub outcome: ImageOutcome,
    pub errors: Vec<String>,
    /// Things done differently from what was asked that did not fail the image.
    #[serde(default)]
    pub warnings: Vec<String>,
    pub dry_run: bool,
    #[serde(default)]
    pub search_stop_reason: Option<String>,
    #[serde(default)]
    pub temporary_bytes_current: u64,
    #[serde(default)]
    pub temporary_bytes_peak: u64,
    #[serde(default)]
    pub temporary_budget_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportOutput {
    #[serde(default = "unversioned_report_schema_version")]
    pub schema_version: u32,
    /// The color space the built-in metric compares in.
    #[serde(default = "default_metric_color_space")]
    pub metric_color_space: String,
    /// The tools found at the start of the run. Absent from reports before
    /// schema version 6, which carried them in every image.
    #[serde(default)]
    pub tools: Option<ToolInventory>,
    /// The GPU found at the start of the run, with the warnings of detecting it.
    #[serde(default)]
    pub gpu: Option<GpuReport>,
    pub reports: Vec<ImageReport>,
    pub failures: usize,
    /// Things about the run as a whole, such as folders that could not be read.
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub stage_totals: ReportStageTotals,
    #[serde(default)]
    pub manifest: Option<RunManifest>,
}

/// What a report without a version is: the format before versions were written.
fn unversioned_report_schema_version() -> u32 {
    1
}

/// The color space the built-in metric compares in.
pub const METRIC_COLOR_SPACE: &str = "srgb-relative-v1";

fn default_metric_color_space() -> String {
    METRIC_COLOR_SPACE.to_string()
}

fn check_schema_version(version: u64) -> anyhow::Result<()> {
    anyhow::ensure!(version > 0, "report schema version is invalid");
    anyhow::ensure!(
        version <= u64::from(REPORT_SCHEMA_VERSION),
        "report schema version {version} is newer than this build understands (it reads versions up to {REPORT_SCHEMA_VERSION}); use a newer smart-compress"
    );
    Ok(())
}

impl ReportOutput {
    /// Reads a report from JSON. The version is checked before the rest is
    /// read, so a report from a newer build is refused with a clear message
    /// rather than a complaint about a field this build does not know.
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(text).context("report is not valid JSON")?;
        check_schema_version(
            value
                .get("schema_version")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(1),
        )?;
        serde_json::from_value(value).context("report does not match the report format")
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        check_schema_version(u64::from(self.schema_version))?;
        let manifest = self
            .manifest
            .as_ref()
            .context("report has no run manifest")?;
        anyhow::ensure!(
            manifest.completed_inputs == self.reports.len(),
            "manifest completed input count does not match report entries"
        );
        if matches!(manifest.status, RunStatus::Complete) {
            anyhow::ensure!(
                manifest.completed_inputs == manifest.expected_inputs,
                "complete report has incomplete input count"
            );
        }
        for report in &self.reports {
            if let Some(index) = report.winner_index {
                anyhow::ensure!(
                    index < report.candidates.len(),
                    "winner index is outside the candidate list for {}",
                    report.input
                );
            }
        }
        Ok(())
    }

    /// How many images ended in each way, as a line of text.
    pub fn outcome_summary(&self) -> String {
        let count = |outcome| {
            self.reports
                .iter()
                .filter(|report| report.outcome == outcome)
                .count()
        };
        let mut summary = format!(
            "{} improved, {} kept as they were, {} skipped, {} failed",
            count(ImageOutcome::Optimized),
            count(ImageOutcome::Kept),
            count(ImageOutcome::Skipped),
            count(ImageOutcome::Failed),
        );
        if let dry_runs @ 1.. = count(ImageOutcome::DryRun) {
            let _ = write!(summary, ", {dry_runs} analyzed without writing");
        }
        summary
    }

    pub fn render(&self, format: ReportFormat) -> anyhow::Result<String> {
        match format {
            ReportFormat::Text => Ok(self.to_text()),
            ReportFormat::Json => serde_json::to_string_pretty(self).map_err(anyhow::Error::from),
            ReportFormat::Markdown => Ok(self.to_markdown()),
        }
    }

    pub fn calculate_stage_totals(reports: &[ImageReport]) -> ReportStageTotals {
        let mut totals = ReportStageTotals::default();
        for report in reports {
            totals.source_decode_ms += report.timings.source_decode_ms.unwrap_or(0);
            totals.normalize_ms += report.timings.normalize_ms.unwrap_or(0);
            totals.search_ms += report.timings.search_ms.unwrap_or(0);
            totals.output_write_ms += report.timings.output_write_ms.unwrap_or(0);
            totals.total_ms += report.timings.total_ms.unwrap_or(0);
            for candidate in &report.candidates {
                totals.encode_ms += candidate.encode_ms.unwrap_or(0);
                totals.candidate_decode_ms += candidate.decode_ms.unwrap_or(0);
                totals.score_ms += candidate.score_ms.unwrap_or(0);
                totals.cache_ms += candidate.cache_ms.unwrap_or(0);
                totals.external_metric_ms += candidate.external_metric_ms.unwrap_or(0);
                if let Some(nanos) = candidate.gpu_timestamp_ns {
                    *totals.gpu_timestamp_ns.get_or_insert(0) += nanos;
                }
            }
        }
        totals
    }

    pub fn to_text(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(output, "Metric color space: {}", self.metric_color_space);
        if let Some(gpu) = &self.gpu {
            let _ = writeln!(
                output,
                "GPU: {:?}{}",
                gpu.status,
                gpu.selected_adapter
                    .as_ref()
                    .map_or(String::new(), |adapter| format!(" ({})", adapter.name)),
            );
            for warning in &gpu.warnings {
                let _ = writeln!(output, "  GPU warning: {warning}");
            }
        }
        let tool_hints = self
            .tools
            .as_ref()
            .map(missing_tool_hints)
            .unwrap_or_default();
        if !tool_hints.is_empty() {
            let _ = writeln!(output, "Missing optional tools:");
            for (name, hint) in tool_hints {
                let _ = writeln!(output, "  - {name}: {hint}");
            }
        }
        output.push('\n');
        for report in &self.reports {
            let _ = writeln!(output, "Input: {}", report.input);
            let _ = writeln!(
                output,
                "Report schema: {}, scoring: {}",
                self.schema_version, report.scoring_version
            );
            if let Some(backend) = &report.gpu_preprocess_backend {
                let _ = writeln!(output, "GPU preprocess: {backend}");
            }
            if let Some(probe) = &report.probe {
                let _ = writeln!(
                    output,
                    "Original: {} bytes, {}x{}, {:?}, decoded {}, source {}{}",
                    report.original_size,
                    probe.width,
                    probe.height,
                    probe.content_class,
                    probe.color_type,
                    probe.original_color_type,
                    probe
                        .icc_profile_size
                        .map_or(String::new(), |size| format!(", ICC {size} bytes"))
                );
            } else {
                let _ = writeln!(output, "Original: {} bytes", report.original_size);
            }
            if report.dry_run {
                let _ = writeln!(output, "Dry run: no outputs written");
            }
            let _ = writeln!(output, "Compute: {}", report.compute_backend);
            if let Some(work) = &report.gpu_work {
                let _ = writeln!(
                    output,
                    "GPU work estimate: source RGBA {} bytes; metric {}x{} ({} bytes); peak metric/candidate {} bytes; full-source tiling required: {}",
                    work.source_rgba_bytes,
                    work.metric_width,
                    work.metric_height,
                    work.metric_rgba_bytes,
                    work.estimated_peak_metric_bytes_per_candidate,
                    work.tiling_required_for_full_source_preprocess
                );
                let _ = writeln!(
                    output,
                    "GPU metric tiling: required: {}; selected metric tile pixels: {}",
                    work.metric_tiling_required,
                    work.selected_metric_tile_pixels
                        .map_or("-".to_string(), |pixels| pixels.to_string())
                );
            }
            if !report.errors.is_empty() {
                let _ = writeln!(output, "Errors:");
                for error in &report.errors {
                    let _ = writeln!(output, "  - {error}");
                }
            }
            if !report.warnings.is_empty() {
                let _ = writeln!(output, "Warnings:");
                for warning in &report.warnings {
                    let _ = writeln!(output, "  - {warning}");
                }
            }
            if let Some(reason) = &report.search_stop_reason {
                let _ = writeln!(output, "Search stopped early: {reason}");
            }

            if !report.candidates.is_empty() {
                let mut table = Table::new();
                table.load_preset(UTF8_FULL);
                table.set_header(vec![
                    Cell::new("Codec"),
                    Cell::new("Settings"),
                    Cell::new("Size"),
                    Cell::new("Quality"),
                    Cell::new("SSIM2"),
                    Cell::new("Butter"),
                    Cell::new("Score backend"),
                    Cell::new("Cache source"),
                    Cell::new("Preprocess ms"),
                    Cell::new("Score ms"),
                    Cell::new("Total ms"),
                    Cell::new("Status"),
                ]);
                for candidate in &report.candidates {
                    table.add_row(vec![
                        Cell::new(format!("{:?}", candidate.codec)),
                        Cell::new(candidate.settings.label()),
                        Cell::new(
                            candidate
                                .size_bytes
                                .map_or("-".to_string(), |size| size.to_string()),
                        ),
                        Cell::new(
                            candidate.quality.map_or("-".to_string(), |quality| {
                                format!("{:.1}", quality.overall)
                            }),
                        ),
                        Cell::new(
                            candidate
                                .quality
                                .and_then(|quality| quality.ssimulacra2)
                                .map_or("-".to_string(), |quality| format!("{quality:.1}")),
                        ),
                        Cell::new(
                            candidate
                                .quality
                                .and_then(|quality| quality.butteraugli)
                                .map_or("-".to_string(), |distance| format!("{distance:.3}")),
                        ),
                        Cell::new(score_backend_label(candidate.score_backend.as_ref())),
                        Cell::new(cache_hit_label(candidate.cache_hit_kind)),
                        Cell::new(timing_label(candidate.preprocess_ms)),
                        Cell::new(timing_label(candidate.score_ms)),
                        Cell::new(timing_label(candidate.total_ms)),
                        Cell::new(format!("{:?}", candidate.status)),
                    ]);
                }
                let _ = writeln!(output, "{table}");
            }

            if let Some(index) = report.winner_index {
                if let Some(winner) = report.candidates.get(index) {
                    let _ = writeln!(
                        output,
                        "Winner: {:?} {} -> {} bytes",
                        winner.codec,
                        winner.settings.label(),
                        winner.size_bytes.unwrap_or(0)
                    );
                }
            } else if report.errors.is_empty() && !report.dry_run {
                let _ = writeln!(output, "Winner: none");
            }
            if let Some(output_path) = &report.output {
                let _ = writeln!(output, "Output: {output_path}");
            }
            let _ = writeln!(
                output,
                "Timings (ms): open={} decode={} preprocess={} orientation={} probe={} normalize={} cache={} metric-prepare={} search={} write={} total={} backpressure-wait={}",
                timing_label(report.timings.source_open_ms),
                timing_label(report.timings.source_decode_ms),
                timing_label(report.timings.gpu_preprocess_ms),
                timing_label(report.timings.orientation_ms),
                timing_label(report.timings.probe_ms),
                timing_label(report.timings.normalize_ms),
                timing_label(report.timings.cache_lookup_ms),
                timing_label(report.timings.external_metric_prepare_ms),
                timing_label(report.timings.search_ms),
                timing_label(report.timings.output_write_ms),
                timing_label(report.timings.total_ms),
                timing_label(report.timings.backpressure_wait_ms),
            );
            output.push('\n');
        }
        for warning in &self.warnings {
            let _ = writeln!(output, "Warning: {warning}");
        }
        let _ = writeln!(output, "Stage totals (ms): {:?}", self.stage_totals);
        let _ = writeln!(output, "Failures: {}", self.failures);
        let _ = writeln!(output, "Summary: {}", self.outcome_summary());
        output
    }

    pub fn to_markdown(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(output, "# Run");
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "- Metric color space: `{}`",
            self.metric_color_space
        );
        if let Some(gpu) = &self.gpu {
            let _ = writeln!(output, "- GPU: `{:?}`", gpu.status);
            if let Some(adapter) = &gpu.selected_adapter {
                let _ = writeln!(output, "- GPU Adapter: `{}`", adapter.name);
            }
            for warning in &gpu.warnings {
                let _ = writeln!(output, "- GPU Warning: {warning}");
            }
        }
        let tool_hints = self
            .tools
            .as_ref()
            .map(missing_tool_hints)
            .unwrap_or_default();
        if !tool_hints.is_empty() {
            let _ = writeln!(output, "- Missing Optional Tools:");
            for (name, hint) in tool_hints {
                let _ = writeln!(output, "  - `{name}`: {hint}");
            }
        }
        output.push('\n');
        for report in &self.reports {
            let _ = writeln!(output, "## {}", report.input);
            let _ = writeln!(output);
            let _ = writeln!(
                output,
                "- Report schema: {}; scoring: `{}`",
                self.schema_version, report.scoring_version
            );
            if let Some(backend) = &report.gpu_preprocess_backend {
                let _ = writeln!(output, "- GPU Preprocess: `{backend}`");
            }
            let _ = writeln!(output, "- Original: {} bytes", report.original_size);
            if let Some(probe) = &report.probe {
                let _ = writeln!(output, "- Dimensions: {}x{}", probe.width, probe.height);
                let _ = writeln!(output, "- Classification: `{:?}`", probe.content_class);
                let _ = writeln!(output, "- Decoded Color: `{}`", probe.color_type);
                let _ = writeln!(output, "- Source Color: `{}`", probe.original_color_type);
                if let Some(size) = probe.icc_profile_size {
                    let _ = writeln!(output, "- ICC Profile: `{size}` bytes");
                }
            }
            let _ = writeln!(output, "- Compute: `{}`", report.compute_backend);
            if let Some(work) = &report.gpu_work {
                let _ = writeln!(
                    output,
                    "- GPU Work Estimate: source RGBA `{}` bytes; metric `{}x{}`; peak metric/candidate `{}` bytes; metric tiling required `{}`; full-source tiling required `{}`",
                    work.source_rgba_bytes,
                    work.metric_width,
                    work.metric_height,
                    work.estimated_peak_metric_bytes_per_candidate,
                    work.metric_tiling_required,
                    work.tiling_required_for_full_source_preprocess
                );
                if let Some(tile_pixels) = work.selected_metric_tile_pixels {
                    let _ = writeln!(output, "- GPU Metric Tile Pixels: `{tile_pixels}`");
                }
            }
            if let Some(output_path) = &report.output {
                let _ = writeln!(output, "- Output: `{output_path}`");
            }
            let _ = writeln!(
                output,
                "- Timings (ms): open {}, decode {}, preprocess {}, orientation {}, probe {}, normalize {}, cache {}, metric-prepare {}, search {}, write {}, total {}, backpressure wait {}",
                timing_label(report.timings.source_open_ms),
                timing_label(report.timings.source_decode_ms),
                timing_label(report.timings.gpu_preprocess_ms),
                timing_label(report.timings.orientation_ms),
                timing_label(report.timings.probe_ms),
                timing_label(report.timings.normalize_ms),
                timing_label(report.timings.cache_lookup_ms),
                timing_label(report.timings.external_metric_prepare_ms),
                timing_label(report.timings.search_ms),
                timing_label(report.timings.output_write_ms),
                timing_label(report.timings.total_ms),
                timing_label(report.timings.backpressure_wait_ms),
            );
            if !report.errors.is_empty() {
                let _ = writeln!(output, "- Errors: {}", report.errors.join("; "));
            }
            if !report.warnings.is_empty() {
                let _ = writeln!(output, "- Warnings: {}", report.warnings.join("; "));
            }
            if let Some(reason) = &report.search_stop_reason {
                let _ = writeln!(output, "- Search stopped early: {reason}");
            }
            if !report.candidates.is_empty() {
                let _ = writeln!(output);
                let _ = writeln!(
                    output,
                    "| Codec | Settings | Size | Quality | SSIM2 | Butter | Score backend | Cache source | Preprocess ms | Score ms | Total ms | Status |"
                );
                let _ = writeln!(
                    output,
                    "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|"
                );
                for candidate in &report.candidates {
                    let _ = writeln!(
                        output,
                        "| {:?} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:?} |",
                        candidate.codec,
                        candidate.settings.label(),
                        candidate
                            .size_bytes
                            .map_or("-".to_string(), |size| size.to_string()),
                        candidate
                            .quality
                            .map_or("-".to_string(), |quality| format!("{:.1}", quality.overall)),
                        candidate
                            .quality
                            .and_then(|quality| quality.ssimulacra2)
                            .map_or("-".to_string(), |quality| format!("{quality:.1}")),
                        candidate
                            .quality
                            .and_then(|quality| quality.butteraugli)
                            .map_or("-".to_string(), |distance| format!("{distance:.3}")),
                        score_backend_label(candidate.score_backend.as_ref()),
                        cache_hit_label(candidate.cache_hit_kind),
                        timing_label(candidate.preprocess_ms),
                        timing_label(candidate.score_ms),
                        timing_label(candidate.total_ms),
                        candidate.status
                    );
                }
            }
            output.push('\n');
        }
        for warning in &self.warnings {
            let _ = writeln!(output, "**Warning:** {warning}");
        }
        let _ = writeln!(output, "**Stage totals (ms):** `{:?}`", self.stage_totals);
        let _ = writeln!(output, "**Failures:** {}", self.failures);
        let _ = writeln!(output, "**Summary:** {}", self.outcome_summary());
        output
    }
}

fn score_backend_label(backend: Option<&ActualScoreBackend>) -> String {
    match backend {
        Some(ActualScoreBackend::Cpu) => "cpu".to_string(),
        Some(ActualScoreBackend::Gpu) => "gpu".to_string(),
        Some(ActualScoreBackend::CpuFallback { reason }) => {
            format!("cpu-fallback ({reason})")
        }
        None => "-".to_string(),
    }
}

fn timing_label(timing_ms: Option<u64>) -> String {
    timing_ms.map_or("-".to_string(), |value| value.to_string())
}

fn cache_hit_label(kind: Option<crate::candidate::CacheHitKind>) -> String {
    match kind {
        Some(crate::candidate::CacheHitKind::EncodedArtifact) => "encoded-artifact".to_string(),
        Some(crate::candidate::CacheHitKind::ScoreMetadata) => "score-metadata".to_string(),
        None => "fresh".to_string(),
    }
}

fn missing_tool_hints(tools: &ToolInventory) -> Vec<(&str, &str)> {
    [
        &tools.cwebp,
        &tools.webpmux,
        &tools.avifenc,
        &tools.avifdec,
        &tools.cjxl,
        &tools.djxl,
        &tools.oxipng,
        &tools.pngquant,
        &tools.cjpegli,
        &tools.cjpeg,
        &tools.ssimulacra2,
        &tools.butteraugli,
    ]
    .into_iter()
    .filter_map(|tool| {
        (!tool.enabled).then_some((tool.name.as_str(), tool.install_hint.as_deref()?))
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialized_tools_do_not_carry_where_they_were_found() {
        let tool = smart_compress_codecs::ToolInfo {
            name: "cwebp".to_string(),
            path: Some("/home/someone/tools/cwebp.exe".into()),
            version: Some("1.5.0".to_string()),
            enabled: true,
            install_hint: None,
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("cwebp") && json.contains("1.5.0"));
        assert!(!json.contains("someone"), "{json}");
        let read: smart_compress_codecs::ToolInfo = serde_json::from_str(&json).unwrap();
        assert!(read.enabled && read.path.is_none());
    }

    #[test]
    fn fallback_backend_label_includes_reason() {
        let backend = ActualScoreBackend::CpuFallback {
            reason: "GPU dispatch failed".to_string(),
        };

        assert_eq!(
            score_backend_label(Some(&backend)),
            "cpu-fallback (GPU dispatch failed)"
        );
    }

    #[test]
    fn older_report_defaults_new_manifest_and_timing_fields() {
        let value = serde_json::json!({
            "reports": [],
            "failures": 0
        });
        let report: ReportOutput = serde_json::from_value(value).unwrap();

        assert_eq!(report.schema_version, 1);
        assert!(report.manifest.is_none());
        assert_eq!(report.stage_totals.total_ms, 0);
        assert!(report.tools.is_none() && report.gpu.is_none());
    }

    #[test]
    fn a_report_from_a_newer_build_is_refused_with_a_clear_message() {
        let future = format!(
            r#"{{"schema_version": {}, "reports": [], "failures": 0, "unknown_new_field": {{}}}}"#,
            REPORT_SCHEMA_VERSION + 1
        );
        let error = ReportOutput::from_json(&future).unwrap_err().to_string();
        assert!(
            error.contains("newer than this build understands"),
            "{error}"
        );
        assert!(
            error.contains(&(REPORT_SCHEMA_VERSION + 1).to_string()),
            "{error}"
        );

        let mut report = ReportOutput::from_json(r#"{"reports": [], "failures": 0}"#).unwrap();
        assert_eq!(report.schema_version, 1);
        report.schema_version = REPORT_SCHEMA_VERSION + 1;
        assert!(report.validate().is_err());
        report.schema_version = 0;
        assert!(report.validate().is_err());
    }

    #[test]
    fn a_report_before_the_run_level_fields_still_loads() {
        // Schema 5 carried the tools, the GPU and the metric color space in every
        // image, and `ssim` and `psnr` for what are now `rgb` and `builtin`.
        let report = ReportOutput::from_json(
            r#"{"schema_version": 5, "failures": 0, "reports": [{
                "input": "a.jpg", "original_size": 10, "probe": null,
                "tools": {}, "gpu": {}, "gpu_work": null, "compute_backend": "cpu-rayon",
                "metric_color_space": "srgb-relative-v1", "quality_metric": "psnr",
                "candidates": [{
                    "codec": "jpeg", "settings": "Lossless", "path": null, "size_bytes": 5,
                    "encode_ms": 1, "decode_ms": 1, "backend": null,
                    "quality": {"ssim": 90.0, "edge": 91.0, "overall": 90.0},
                    "status": "Accepted", "warnings": []
                }],
                "winner_index": null, "errors": [], "dry_run": false,
                "saved_bytes": null, "output": null
            }]}"#,
        )
        .unwrap();
        assert_eq!(report.reports[0].candidates[0].quality.unwrap().rgb, 90.0);
    }
}
