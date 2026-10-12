//! Snapshots of the three report renderings for one synthetic run that
//! contains every candidate outcome: accepted, rejected for quality, rejected
//! for size, dominated, served from cache, timed out, scored after a GPU
//! fallback, plus an unavailable tool and an image of each outcome.
//!
//! Run with `UPDATE_SNAPSHOTS=1` to rewrite the files after an intended change.

use std::path::{Path, PathBuf};

use smart_compress_codecs::{Codec, EncoderSettings, ToolInfo, ToolInventory};
use smart_compress_core::{
    ImageReport, ReportOutput,
    candidate::{
        ActualScoreBackend, CacheHitKind, CandidateInfluence, CandidateReport, CandidateStatus,
        ProcessFailureSummary, QualitySource,
    },
    gpu::{GpuAdapterReport, GpuReport, GpuStatus},
    image_probe::{ContentClass, ImageProbe},
    report::{
        ImageOutcome, ImageStageTimings, METRIC_COLOR_SPACE, REPORT_SCHEMA_VERSION, RunManifest,
        RunStatus,
    },
};
use smart_compress_metrics::QualityScore;

fn tool(name: &str, enabled: bool) -> ToolInfo {
    ToolInfo {
        name: name.to_string(),
        path: enabled.then(|| PathBuf::from(format!("tools/{name}.exe"))),
        version: enabled.then(|| format!("{name} 1.0")),
        enabled,
        install_hint: (!enabled).then(|| format!("Install {name} to enable this format.")),
    }
}

fn tools() -> ToolInventory {
    ToolInventory {
        cwebp: tool("cwebp", true),
        webpmux: tool("webpmux", true),
        avifenc: tool("avifenc", true),
        avifdec: tool("avifdec", true),
        // An unavailable tool: its format is never planned.
        cjxl: tool("cjxl", false),
        djxl: tool("djxl", false),
        oxipng: tool("oxipng", true),
        pngquant: tool("pngquant", true),
        cjpegli: tool("cjpegli", true),
        cjpeg: tool("cjpeg", false),
        ssimulacra2: tool("ssimulacra2", true),
        butteraugli: tool("butteraugli_main", true),
    }
}

fn gpu() -> GpuReport {
    let adapter = GpuAdapterReport {
        name: "Example GPU".to_string(),
        vendor: 4318,
        device: 9479,
        device_type: "DiscreteGpu".to_string(),
        backend: "Vulkan".to_string(),
        max_texture_dimension_2d: 16_384,
        max_buffer_size: 2_147_483_648,
        max_storage_buffer_binding_size: 2_147_483_648,
        timestamp_queries: true,
    };
    GpuReport {
        status: GpuStatus::Enabled,
        requested_backend: "wgpu".to_string(),
        requested_adapter_name: None,
        requested_adapter_vendor: None,
        requested_adapter_device: None,
        selected_adapter: Some(adapter.clone()),
        available_adapters: vec![adapter],
        warnings: Vec::new(),
    }
}

fn scored(overall: f32) -> QualityScore {
    QualityScore {
        rgb: overall + 4.0,
        ssimulacra2: Some(overall),
        butteraugli: None,
        alpha: None,
        composite_min: None,
        alpha_edge: None,
        edge: overall - 6.0,
        overall,
    }
}

fn lossy(codec: Codec, quality: u8, effort: u8, size: u64) -> CandidateReport {
    let mut candidate = CandidateReport::unevaluated(
        codec,
        EncoderSettings::Quality { quality, effort },
        String::new(),
    );
    candidate.size_bytes = Some(size);
    candidate.encode_ms = Some(120);
    candidate.total_ms = Some(400);
    candidate.backend = Some(
        match codec {
            Codec::Jpeg => "cjpegli",
            Codec::Webp => "cwebp",
            Codec::Avif => "avifenc",
            Codec::Jxl => "cjxl",
            Codec::Png => "native-png+optional-oxipng",
        }
        .to_string(),
    );
    candidate
}

fn candidates() -> Vec<CandidateReport> {
    let mut accepted = lossy(Codec::Avif, 72, 6, 310_000);
    accepted.status = CandidateStatus::Accepted;
    accepted.quality = Some(scored(80.6));
    accepted.decode_ms = Some(90);
    accepted.score_ms = Some(210);
    accepted.external_metric_ms = Some(200);
    accepted.gpu_timestamp_ns = Some(1_250_000);
    accepted.score_backend = Some(ActualScoreBackend::Gpu);
    accepted.influence = Some(CandidateInfluence::ImprovedWinner);

    let mut rejected_quality = lossy(Codec::Webp, 60, 4, 240_000);
    rejected_quality.status = CandidateStatus::Rejected("quality 74.2 below 80.0".to_string());
    rejected_quality.quality = Some(scored(74.2));
    rejected_quality.score_backend = Some(ActualScoreBackend::Gpu);
    rejected_quality.influence = Some(CandidateInfluence::Rejected);

    let mut rejected_size =
        CandidateReport::unevaluated(Codec::Png, EncoderSettings::Lossless, String::new());
    rejected_size.status = CandidateStatus::Rejected("not smaller than original".to_string());
    rejected_size.size_bytes = Some(4_800_000);
    rejected_size.encode_ms = Some(60);
    rejected_size.total_ms = Some(60);
    rejected_size.backend = Some("native-png+optional-oxipng".to_string());
    rejected_size.influence = Some(CandidateInfluence::Rejected);

    let mut dominated = lossy(Codec::Jpeg, 92, 0, 890_000);
    dominated.status =
        CandidateStatus::Skipped("dominated by current accepted candidate".to_string());

    let mut cached = lossy(Codec::Webp, 78, 4, 420_000);
    cached.status = CandidateStatus::Accepted;
    cached.quality = Some(scored(83.1));
    cached.backend = Some("cache".to_string());
    cached.encode_ms = Some(0);
    cached.cache_ms = Some(3);
    cached.total_ms = Some(3);
    cached.quality_source = QualitySource::ScoreCache;
    cached.cache_hit_kind = Some(CacheHitKind::ScoreMetadata);

    let mut timed_out =
        CandidateReport::unevaluated(Codec::Avif, EncoderSettings::Lossless, String::new());
    timed_out.status = CandidateStatus::Failed(
        "tools/avifenc.exe Timeout after 120000 ms: terminated".to_string(),
    );
    timed_out.total_ms = Some(120_050);
    timed_out.influence = Some(CandidateInfluence::Failed);
    timed_out.process_failure = Some(ProcessFailureSummary {
        executable: PathBuf::from("tools/avifenc.exe"),
        kind: "Timeout".to_string(),
        exit_code: None,
        message: "terminated".to_string(),
        elapsed_ms: 120_000,
        termination_requested: true,
    });

    let mut fallback = lossy(Codec::Jpeg, 78, 0, 365_000);
    fallback.status = CandidateStatus::Accepted;
    fallback.quality = Some(scored(81.4));
    fallback.score_backend = Some(ActualScoreBackend::CpuFallback {
        reason: "GPU metric scoring failed: device lost".to_string(),
    });
    fallback
        .warnings
        .push("CPU scoring fallback: GPU metric scoring failed: device lost".to_string());
    fallback.influence = Some(CandidateInfluence::FirstPassing);

    vec![
        fallback,
        dominated,
        rejected_quality,
        cached,
        timed_out,
        accepted,
        rejected_size,
    ]
}

fn image(input: &str) -> ImageReport {
    ImageReport {
        input: input.to_string(),
        input_identity: Some("0".repeat(64)),
        output: None,
        original_size: 1_000_000,
        saved_bytes: None,
        probe: Some(ImageProbe {
            path: input.to_string(),
            format: Some("Jpeg".to_string()),
            width: 3000,
            height: 2000,
            pixels: 6_000_000,
            color_type: "Rgb8".to_string(),
            original_color_type: "Rgb8".to_string(),
            has_alpha: false,
            metadata_size: 3144,
            icc_profile_size: Some(3144),
            unique_color_ratio: 0.42,
            edge_density: 0.11,
            flat_region_ratio: 0.3,
            alpha_coverage: 0.0,
            orientation_applied: true,
            content_class: ContentClass::Photo,
        }),
        gpu_work: None,
        gpu_preprocess_backend: None,
        compute_backend: "gpu-wgpu".to_string(),
        scoring_version: "score-snapshot".to_string(),
        quality_metric: "ssimulacra2".to_string(),
        min_quality: Some(80.0),
        timings: ImageStageTimings {
            source_open_ms: Some(2),
            source_decode_ms: Some(40),
            gpu_preprocess_ms: None,
            orientation_ms: Some(12),
            probe_ms: Some(8),
            normalize_ms: Some(55),
            cache_lookup_ms: Some(1),
            external_metric_prepare_ms: Some(30),
            search_ms: Some(9_000),
            output_write_ms: Some(4),
            total_ms: Some(9_152),
            backpressure_wait_ms: Some(310),
        },
        candidates: Vec::new(),
        winner_index: None,
        outcome: ImageOutcome::Failed,
        errors: Vec::new(),
        warnings: Vec::new(),
        dry_run: false,
        search_stop_reason: None,
        temporary_bytes_current: 0,
        temporary_bytes_peak: 0,
        temporary_budget_bytes: None,
    }
}

fn report() -> ReportOutput {
    let mut optimized = image("photos/harbour.jpg");
    optimized.candidates = candidates();
    optimized.winner_index = Some(5);
    optimized.output = Some("optimized/harbour.avif".to_string());
    optimized.saved_bytes = Some(690_000);
    optimized.outcome = ImageOutcome::Optimized;
    optimized.temporary_bytes_peak = 7_105_000;
    optimized.search_stop_reason = Some("candidate attempt budget of 7 reached".to_string());
    optimized.warnings.push(
        "embedded ICC profile dropped: it describes GRAY data but the image decodes to RGB"
            .to_string(),
    );

    let mut kept = image("photos/receipt.jpg");
    kept.outcome = ImageOutcome::Kept;
    kept.original_size = 48_000;
    let mut too_low = lossy(Codec::Jpeg, 92, 0, 31_000);
    too_low.status = CandidateStatus::Rejected("quality 77.9 below 80.0".to_string());
    too_low.quality = Some(scored(77.9));
    too_low.score_backend = Some(ActualScoreBackend::Cpu);
    too_low.influence = Some(CandidateInfluence::Rejected);
    kept.candidates = vec![too_low];
    kept.warnings.push(
        "nothing scored at least 80.0 while staying smaller than the original; the original is left as it is"
            .to_string(),
    );

    let mut skipped = image("photos/harbour-2.jpg");
    skipped.outcome = ImageOutcome::Skipped;
    skipped.warnings.push(
        "output already exists, left as it is: optimized/harbour-2.avif (pass --overwrite to replace it)"
            .to_string(),
    );

    let mut failed = image("photos/broken.jpg");
    failed.probe = None;
    failed
        .errors
        .push("decoding photos/broken.jpg: the image format could not be determined".to_string());

    let reports = vec![optimized, kept, skipped, failed];
    ReportOutput {
        schema_version: REPORT_SCHEMA_VERSION,
        metric_color_space: METRIC_COLOR_SPACE.to_string(),
        tools: Some(tools()),
        gpu: Some(gpu()),
        stage_totals: ReportOutput::calculate_stage_totals(&reports),
        failures: 1,
        warnings: vec!["could not read part of photos/private: Access is denied.".to_string()],
        manifest: Some(RunManifest {
            run_id: "f".repeat(64),
            status: RunStatus::Complete,
            started_unix_ms: 1_791_000_000_000,
            completed_unix_ms: Some(1_791_000_009_500),
            expected_inputs: 4,
            completed_inputs: 4,
            config_fingerprint: "c".repeat(64),
            resume_decision: None,
        }),
        reports,
    }
}

fn assert_snapshot(name: &str, actual: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/snapshots")
        .join(name);
    let actual = actual.replace("\r\n", "\n");
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &actual).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing snapshot {}: {error}", path.display()))
        .replace("\r\n", "\n");
    assert!(
        expected == actual,
        "{name} changed; rerun with UPDATE_SNAPSHOTS=1 if this is intended.\n--- expected\n{expected}\n--- actual\n{actual}"
    );
}

#[test]
fn text_report_matches_its_snapshot() {
    assert_snapshot("report.txt", &report().to_text());
}

#[test]
fn markdown_report_matches_its_snapshot() {
    assert_snapshot("report.md", &report().to_markdown());
}

#[test]
fn json_report_matches_its_snapshot_and_validates() {
    let report = report();
    report.validate().unwrap();
    let json = serde_json::to_string_pretty(&report).unwrap();
    assert_snapshot("report.json", &json);
    // The snapshot is also a compatibility check: it must still parse.
    let parsed: ReportOutput = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.reports.len(), 4);
    assert_eq!(parsed.reports[0].winner_index, Some(5));
}
