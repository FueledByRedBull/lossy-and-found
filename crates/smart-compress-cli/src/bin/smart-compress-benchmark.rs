mod corpus;

use std::{
    collections::BTreeMap,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{Value, json};
use smart_compress_codecs::ToolInventory;
use smart_compress_core::{
    CliConfig, ImageReport, Optimizer, Preset, ReportFormat, SpeedPreset, TargetPolicy,
    atomic_file::atomic_write,
    candidate::CandidateStatus,
    config::{ParallelismOverrides, PrimaryMetric, parse_codec_list},
};

/// The same bound as the command line puts on `--workers`.
const MAX_WORKERS: usize = 256;

#[derive(Debug, Parser)]
#[command(name = "smart-compress-benchmark")]
#[command(about = "Measure the end-to-end smart-compress pipeline")]
struct Args {
    /// Images or folders to measure.
    #[arg(required_unless_present = "generate_corpus")]
    input: Vec<PathBuf>,

    /// Write the deterministic synthetic corpus and corpus.json to DIR, then exit.
    #[arg(long, value_name = "DIR")]
    generate_corpus: Option<PathBuf>,

    /// Corpus size classes; 12mp, 50mp and 199mp are large and opt-in.
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = corpus::SizeClass::DEFAULT)]
    corpus_sizes: Vec<corpus::SizeClass>,

    /// Seed of the generated corpus.
    #[arg(long, default_value_t = corpus::DEFAULT_SEED)]
    corpus_seed: u64,

    /// File the results are written to, as JSON.
    #[arg(long, default_value = "benchmark-results.json")]
    out: PathBuf,

    /// Times the inputs are run in each mode.
    #[arg(long, default_value_t = 1)]
    runs: usize,

    /// Output formats to try, comma separated (default: the command line's).
    #[arg(long)]
    formats: Option<String>,

    /// Quality preset, as for smart-compress.
    #[arg(long, value_enum, default_value_t = PresetArg::Balanced)]
    preset: PresetArg,

    /// Search effort, as for smart-compress.
    #[arg(long, value_enum, default_value_t = SpeedArg::Balanced)]
    speed: SpeedArg,

    /// Target, as for smart-compress.
    #[arg(long, value_enum, default_value_t = TargetArg::Web)]
    target: TargetArg,

    /// Cache folder; turns the cache on.
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Make the GPU mode keep the original on the GPU (gpu-prepared) instead of
    /// uploading it for every candidate (gpu-metrics).
    #[arg(long)]
    prepared_gpu: bool,

    /// Make the GPU mode check every score against the CPU (gpu-verify).
    #[arg(long)]
    verify_gpu: bool,

    /// Execution modes to compare; overrides --prepared-gpu and --verify-gpu.
    #[arg(long, value_enum, value_delimiter = ',')]
    modes: Vec<Mode>,

    /// Score that decides acceptance. The GPU only computes the built-in
    /// score, so backend comparisons are most telling with `builtin`.
    #[arg(long, value_enum, default_value_t = MetricArg::Auto)]
    metric: MetricArg,

    /// Smallest image, in pixels, scored on the GPU (default: the CLI default).
    #[arg(long)]
    gpu_min_pixels: Option<u64>,

    /// Workers, as for smart-compress (default: the number of logical processors).
    #[arg(long)]
    workers: Option<usize>,

    /// Overrides for the separate uses of --workers, to vary one dimension
    /// alone: comma-separated `files`, `candidates`, `encoder-threads` and
    /// `metric-threads` values, for example `files=2,candidates=1`.
    #[arg(long, value_name = "LIST")]
    shape: Option<String>,

    /// Largest accepted input, in pixels (default: the CLI default).
    #[arg(long)]
    max_pixels: Option<u64>,

    /// Embed every image report instead of the per-image summary rows.
    #[arg(long)]
    full_reports: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Mode {
    Cpu,
    GpuMetrics,
    GpuPrepared,
    GpuVerify,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::GpuMetrics => "gpu-metrics",
            Self::GpuPrepared => "gpu-prepared",
            Self::GpuVerify => "gpu-verify",
        }
    }
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum MetricArg {
    Auto,
    Ssimulacra2,
    Ssimulacra2Inprocess,
    #[value(alias = "psnr")]
    Builtin,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum PresetArg {
    Strict,
    Balanced,
    Aggressive,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum TargetArg {
    #[value(alias = "modern-web")]
    Web,
    Archive,
    Smallest,
    MaxCompat,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum SpeedArg {
    Fast,
    Balanced,
    Exhaustive,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(dir) = &args.generate_corpus {
        corpus::generate(dir, &args.corpus_sizes, args.corpus_seed)?;
        println!("{}", dir.join(corpus::MANIFEST_NAME).display());
        return Ok(());
    }
    anyhow::ensure!(args.runs > 0, "--runs must be greater than zero");
    anyhow::ensure!(
        args.input.iter().all(|path| path.exists()),
        "every benchmark input must exist"
    );

    let started_unix_ms = unix_time_ms();
    let command = std::env::args().collect::<Vec<_>>();
    let modes = if args.modes.is_empty() {
        vec![
            Mode::Cpu,
            if args.verify_gpu {
                Mode::GpuVerify
            } else if args.prepared_gpu {
                Mode::GpuPrepared
            } else {
                Mode::GpuMetrics
            },
        ]
    } else {
        args.modes.clone()
    };
    let mut results = Vec::new();
    let mut cases = Vec::new();
    // Modes alternate within each repetition so drift affects them equally.
    for run_index in 0..args.runs {
        for mode in &modes {
            let (result, case) = run_case(&args, *mode, run_index, started_unix_ms)?;
            results.push(result);
            cases.push(case);
        }
    }

    let output = json!({
        "schema_version": 1,
        "benchmark": "smart-compress-end-to-end-v2",
        "started_unix_ms": started_unix_ms,
        "command": command,
        "hardware": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "family": std::env::consts::FAMILY,
            "tools": ToolInventory::detect(),
        },
        "inputs": args.input.iter().map(input_identity).collect::<Result<Vec<_>>>()?,
        "runs_per_mode": args.runs,
        "summary": summarize(&modes, &cases),
        "harness_peak_working_set_bytes": peak_working_set_bytes(),
        "results": results,
    });
    let text = serde_json::to_string_pretty(&output).context("serializing benchmark report")?;
    atomic_write(&args.out, text.as_bytes(), false)
        .with_context(|| format!("writing benchmark report {}", args.out.display()))?;
    println!("{}", args.out.display());
    Ok(())
}

struct Case {
    mode: Mode,
    wall_ms: u64,
    reports: Vec<ImageReport>,
}

fn run_case(
    args: &Args,
    mode: Mode,
    run_index: usize,
    started_unix_ms: u64,
) -> Result<(Value, Case)> {
    let case_root = std::env::temp_dir().join(format!(
        "smart-compress-benchmark-{started_unix_ms}-{}-{run_index}",
        mode.name()
    ));
    std::fs::create_dir_all(&case_root)
        .with_context(|| format!("creating benchmark output {}", case_root.display()))?;

    let mut config = CliConfig {
        inputs: args.input.clone(),
        recursive: true,
        out_dir: Some(case_root.join("output")),
        preset: map_preset(args.preset),
        speed: map_speed(args.speed),
        target: map_target(args.target),
        report: ReportFormat::Json,
        ..CliConfig::default()
    };
    if let Some(formats) = &args.formats {
        config.formats = parse_codec_list(formats)?;
    }
    config.metrics.primary = match args.metric {
        MetricArg::Auto => PrimaryMetric::Auto,
        MetricArg::Ssimulacra2 => PrimaryMetric::Ssimulacra2,
        MetricArg::Ssimulacra2Inprocess => PrimaryMetric::Ssimulacra2Inprocess,
        MetricArg::Builtin => PrimaryMetric::Builtin,
    };
    if let Some(workers) = args.workers {
        config.workers = workers.clamp(1, MAX_WORKERS);
    }
    if let Some(shape) = &args.shape {
        config.parallelism = parse_shape(shape)?;
    }
    if let Some(max_pixels) = args.max_pixels {
        config.max_pixels = max_pixels;
    }
    if let Some(min_pixels) = args.gpu_min_pixels {
        config.gpu.min_pixels_for_gpu_metrics = min_pixels;
    }
    config.gpu.enabled = mode != Mode::Cpu;
    config.gpu.prefer_gpu_metrics = true;
    config.gpu.prefer_gpu_preprocess = mode == Mode::GpuPrepared;
    config.gpu.verify_gpu = mode == Mode::GpuVerify;
    if let Some(cache_dir) = &args.cache_dir {
        config.cache.enabled = true;
        config.cache.dir = Some(cache_dir.join(mode.name()));
    }

    let cache_state = if args.cache_dir.is_some() && run_index == 0 {
        "cold"
    } else if args.cache_dir.is_some() {
        "warm"
    } else {
        "disabled"
    };
    let start = Instant::now();
    let summary = Optimizer::new(config).run()?;
    let wall_ms = start.elapsed().as_millis() as u64;
    let _ = std::fs::remove_dir_all(&case_root);

    let images = if args.full_reports {
        serde_json::to_value(&summary.report.reports).context("serializing case reports")?
    } else {
        Value::Array(summary.report.reports.iter().map(image_row).collect())
    };
    let result = json!({
        "mode": mode.name(),
        "run_index": run_index,
        "cache": cache_state,
        "wall_ms": wall_ms,
        "failures": summary.report.failures,
        "stage_totals": summary.report.stage_totals,
        "reports": images,
    });
    Ok((
        result,
        Case {
            mode,
            wall_ms,
            reports: summary.report.reports,
        },
    ))
}

fn candidate_key(report: &ImageReport, index: usize) -> String {
    let candidate = &report.candidates[index];
    format!(
        "{} {}",
        format!("{:?}", candidate.codec).to_ascii_lowercase(),
        candidate.settings.label()
    )
}

fn winner_key(report: &ImageReport) -> String {
    report.winner_index.map_or_else(
        || "none".to_string(),
        |index| {
            format!(
                "{} {} bytes",
                candidate_key(report, index),
                report.candidates[index].size_bytes.unwrap_or(0)
            )
        },
    )
}

fn stage_sum(
    report: &ImageReport,
    stage: impl Fn(&smart_compress_core::candidate::CandidateReport) -> Option<u64>,
) -> u64 {
    report.candidates.iter().filter_map(stage).sum()
}

/// One compact row per image: enough to compare modes without the full report.
fn image_row(report: &ImageReport) -> Value {
    let backends = report
        .candidates
        .iter()
        .filter_map(|candidate| candidate.score_backend.as_ref())
        .map(|backend| {
            serde_json::to_value(backend)
                .ok()
                .and_then(|value| value.as_str().map(str::to_string))
                .unwrap_or_else(|| "cpu-fallback".to_string())
        })
        .fold(BTreeMap::<String, usize>::new(), |mut counts, backend| {
            *counts.entry(backend).or_default() += 1;
            counts
        });
    json!({
        "input": report.input,
        "pixels": report.probe.as_ref().map(|probe| probe.pixels),
        "size_class": size_class(report),
        "original_bytes": report.original_size,
        "winner": winner_key(report),
        "winner_quality": report
            .winner_index
            .and_then(|index| report.candidates[index].quality)
            .map(|quality| quality.overall),
        "errors": report.errors,
        "candidates": report.candidates.len(),
        "score_backends": backends,
        "gpu_preprocess": report.gpu_preprocess_backend,
        "total_ms": report.timings.total_ms,
        "source_decode_ms": report.timings.source_decode_ms,
        "gpu_preprocess_ms": report.timings.gpu_preprocess_ms,
        "normalize_ms": report.timings.normalize_ms,
        "search_ms": report.timings.search_ms,
        "encode_ms": stage_sum(report, |candidate| candidate.encode_ms),
        "candidate_decode_ms": stage_sum(report, |candidate| candidate.decode_ms),
        "score_ms": stage_sum(report, |candidate| candidate.score_ms),
        "external_metric_ms": stage_sum(report, |candidate| candidate.external_metric_ms),
        "gpu_upload_ms": stage_sum(report, |candidate| candidate.gpu_upload_ms),
        "gpu_compute_ms": stage_sum(report, |candidate| candidate.gpu_compute_ms),
        "gpu_readback_ms": stage_sum(report, |candidate| candidate.gpu_readback_ms),
        "gpu_timestamp_ns": report
            .candidates
            .iter()
            .filter_map(|candidate| candidate.gpu_timestamp_ns)
            .reduce(|total, nanos| total + nanos),
        "temporary_bytes_peak": report.temporary_bytes_peak,
        "estimated_peak_metric_bytes": report
            .gpu_work
            .as_ref()
            .map(|work| work.estimated_peak_metric_bytes_per_candidate),
    })
}

fn size_class(report: &ImageReport) -> &'static str {
    match report.probe.as_ref().map_or(0, |probe| probe.pixels) {
        0..=262_144 => "tiny",
        262_145..=4_500_000 => "normal",
        4_500_001..=13_000_000 => "12mp",
        13_000_001..=52_000_000 => "50mp",
        _ => "199mp",
    }
}

fn median(values: &mut [u64]) -> Option<u64> {
    values.sort_unstable();
    values.get(values.len() / 2).copied()
}

/// Cross-mode comparison: speed per mode and size class, and whether the
/// backends reached the same decisions.
fn summarize(modes: &[Mode], cases: &[Case]) -> Value {
    let mut per_mode = BTreeMap::new();
    let mut per_class: BTreeMap<&str, BTreeMap<&str, Value>> = BTreeMap::new();
    for mode in modes {
        let mine = cases.iter().filter(|case| case.mode == *mode);
        let mut walls = mine.clone().map(|case| case.wall_ms).collect::<Vec<_>>();
        walls.sort_unstable();
        per_mode.insert(
            mode.name(),
            json!({
                "wall_ms": walls,
                "median_wall_ms": walls.get(walls.len() / 2),
                "p95_wall_ms": walls.get((walls.len() * 95).div_ceil(100).saturating_sub(1)),
                "failed_images": mine
                    .clone()
                    .flat_map(|case| &case.reports)
                    .filter(|report| !report.errors.is_empty())
                    .count(),
            }),
        );
        let mut classes: BTreeMap<&str, Vec<&ImageReport>> = BTreeMap::new();
        for report in mine.flat_map(|case| &case.reports) {
            classes.entry(size_class(report)).or_default().push(report);
        }
        for (class, reports) in classes {
            let stat = |value: &dyn Fn(&ImageReport) -> u64| {
                median(
                    &mut reports
                        .iter()
                        .map(|report| value(report))
                        .collect::<Vec<_>>(),
                )
            };
            per_class.entry(class).or_default().insert(
                mode.name(),
                json!({
                    "images": reports.len(),
                    "median_total_ms": stat(&|report| report.timings.total_ms.unwrap_or(0)),
                    "median_score_ms": stat(&|report| stage_sum(report, |c| c.score_ms)),
                    "median_encode_ms": stat(&|report| stage_sum(report, |c| c.encode_ms)),
                    "median_candidate_decode_ms": stat(&|report| stage_sum(report, |c| c.decode_ms)),
                    "median_gpu_preprocess_ms": stat(&|report| report.timings.gpu_preprocess_ms.unwrap_or(0)),
                    "max_temporary_bytes": reports.iter().map(|report| report.temporary_bytes_peak).max(),
                }),
            );
        }
    }

    // Decisions: every run of every mode must pick the same winner and give
    // each candidate the same verdict.
    let mut winners: BTreeMap<&str, BTreeMap<String, Vec<&str>>> = BTreeMap::new();
    let mut verdicts: BTreeMap<(String, String), BTreeMap<&'static str, Vec<&str>>> =
        BTreeMap::new();
    for case in cases {
        for report in &case.reports {
            let modes = winners
                .entry(report.input.as_str())
                .or_default()
                .entry(winner_key(report))
                .or_default();
            if !modes.contains(&case.mode.name()) {
                modes.push(case.mode.name());
            }
            for (index, candidate) in report.candidates.iter().enumerate() {
                let verdict = match &candidate.status {
                    CandidateStatus::Accepted => "passed",
                    CandidateStatus::Rejected(_) if candidate.quality.is_some() => "failed",
                    _ => continue,
                };
                let modes = verdicts
                    .entry((report.input.clone(), candidate_key(report, index)))
                    .or_default()
                    .entry(verdict)
                    .or_default();
                if !modes.contains(&case.mode.name()) {
                    modes.push(case.mode.name());
                }
            }
        }
    }
    let winner_mismatches = winners
        .iter()
        .filter(|(_, by_winner)| by_winner.len() > 1)
        .map(|(input, by_winner)| json!({ "input": input, "winners": by_winner }))
        .collect::<Vec<_>>();
    let floor_crossings = verdicts
        .iter()
        .filter(|(_, by_verdict)| by_verdict.len() > 1)
        .map(|((input, candidate), by_verdict)| {
            json!({ "input": input, "candidate": candidate, "verdicts": by_verdict })
        })
        .collect::<Vec<_>>();
    json!({
        "modes": per_mode,
        "size_classes": per_class,
        "images_compared": winners.len(),
        "winner_mismatches": winner_mismatches,
        "quality_floor_crossings": floor_crossings,
    })
}

/// Peak resident memory of this process; encoder child processes are separate.
#[cfg(windows)]
fn peak_working_set_bytes() -> Option<u64> {
    #[repr(C)]
    struct ProcessMemoryCounters {
        size: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        rest: [usize; 7],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }
    let mut counters = ProcessMemoryCounters {
        size: size_of::<ProcessMemoryCounters>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        rest: [0; 7],
    };
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.size) };
    (ok != 0).then_some(counters.peak_working_set_size as u64)
}

#[cfg(not(windows))]
fn peak_working_set_bytes() -> Option<u64> {
    None
}

fn input_identity(path: &PathBuf) -> Result<Value> {
    if path.is_dir() {
        return Ok(json!({ "path": path, "directory": true }));
    }
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading benchmark input {}", path.display()))?;
    Ok(json!({
        "path": path,
        "bytes": bytes.len(),
        "blake3": blake3::hash(&bytes).to_hex().to_string(),
    }))
}

fn parse_shape(text: &str) -> Result<ParallelismOverrides> {
    let mut shape = ParallelismOverrides::default();
    for pair in text.split(',') {
        let (key, value) = pair
            .split_once('=')
            .with_context(|| format!("--shape entry '{pair}' is not key=value"))?;
        let value = Some(
            value
                .parse::<usize>()
                .with_context(|| format!("--shape value '{value}' is not a number"))?,
        );
        match key {
            "files" => shape.files = value,
            "candidates" => shape.candidates = value,
            "encoder-threads" => shape.encoder_threads = value,
            "metric-threads" => shape.metric_threads = value,
            other => anyhow::bail!("unknown --shape key '{other}'"),
        }
    }
    Ok(shape)
}

fn map_preset(value: PresetArg) -> Preset {
    match value {
        PresetArg::Strict => Preset::Strict,
        PresetArg::Balanced => Preset::Balanced,
        PresetArg::Aggressive => Preset::Aggressive,
    }
}

fn map_target(value: TargetArg) -> TargetPolicy {
    match value {
        TargetArg::Web => TargetPolicy::Web,
        TargetArg::Archive => TargetPolicy::Archive,
        TargetArg::Smallest => TargetPolicy::Smallest,
        TargetArg::MaxCompat => TargetPolicy::MaxCompat,
    }
}

fn map_speed(value: SpeedArg) -> SpeedPreset {
    match value {
        SpeedArg::Fast => SpeedPreset::Fast,
        SpeedArg::Balanced => SpeedPreset::Balanced,
        SpeedArg::Exhaustive => SpeedPreset::Exhaustive,
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shape_names_the_dimensions_it_overrides() {
        let shape = parse_shape("files=2,candidates=1,encoder-threads=3,metric-threads=4").unwrap();
        assert_eq!(
            shape,
            ParallelismOverrides {
                files: Some(2),
                candidates: Some(1),
                encoder_threads: Some(3),
                metric_threads: Some(4),
            }
        );
        assert_eq!(
            parse_shape("candidates=7").unwrap(),
            ParallelismOverrides {
                candidates: Some(7),
                ..ParallelismOverrides::default()
            }
        );
        assert!(parse_shape("files").is_err());
        assert!(parse_shape("files=many").is_err());
        assert!(parse_shape("fibres=2").is_err());
    }

    #[test]
    fn a_format_list_is_read_whole() {
        let args = Args::try_parse_from([
            "smart-compress-benchmark",
            "input.png",
            "--formats",
            "jpeg,webp,avif",
        ])
        .unwrap();
        assert_eq!(args.formats.as_deref(), Some("jpeg,webp,avif"));
    }
}
