use std::{
    collections::HashMap,
    io::IsTerminal,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use smart_compress_codecs::{Codec, EncoderSettings};
use smart_compress_core::{
    AlphaPolicy, CliConfig, MetadataPolicy, Optimizer, Preset, ProgressObserver, ReportFormat,
    SearchMode, SpeedPreset, TargetPolicy,
    atomic_file::atomic_write,
    cache::CandidateCache,
    config::{CacheConfig, FileConfig, PrimaryMetric, parse_codec_list, parse_file_config},
};

mod gui;

/// Far above any machine's core count; a larger request would only create
/// threads.
const MAX_WORKERS: usize = 256;

fn clamp_workers(workers: usize) -> usize {
    workers.clamp(1, MAX_WORKERS)
}

#[cfg(windows)]
unsafe extern "system" fn console_control_handler(control_type: u32) -> i32 {
    if control_type <= 5 {
        smart_compress_codecs::request_cancel();
        1
    } else {
        0
    }
}

#[cfg(windows)]
#[link(name = "Kernel32")]
unsafe extern "system" {
    fn SetConsoleCtrlHandler(
        handler: Option<unsafe extern "system" fn(u32) -> i32>,
        add: i32,
    ) -> i32;
}

fn install_cancellation_handler() {
    #[cfg(windows)]
    unsafe {
        let _ = SetConsoleCtrlHandler(Some(console_control_handler), 1);
    }
}

const INPUT_OUTPUT: &str = "Input and output";
const QUALITY: &str = "Quality";
const FORMATS: &str = "Formats";
const METADATA_ALPHA: &str = "Metadata and alpha";
const LIMITS: &str = "Limits and budgets";
const CACHE: &str = "Cache";
const GPU: &str = "GPU";
const REPORTS: &str = "Reports";
const CONTROL_PANEL: &str = "Control panel";

#[derive(Debug, Parser)]
#[command(name = "smart-compress", version)]
#[command(about = "Choose the smallest acceptable optimized image from multiple codecs")]
struct Args {
    /// Images to optimize: files, or folders with --recursive.
    #[arg(required = false, help_heading = INPUT_OUTPUT)]
    input: Vec<PathBuf>,

    /// Write the result to exactly this file. For a single input image only; the
    /// extension picks the format, and only that format is tried.
    #[arg(long, value_name = "FILE", help_heading = INPUT_OUTPUT)]
    out: Option<PathBuf>,

    /// Folder for the results. Default: an "optimized" folder next to each input.
    #[arg(long, value_name = "DIR", help_heading = INPUT_OUTPUT)]
    out_dir: Option<PathBuf>,

    /// Process the images in the folders given as input, subfolders included.
    #[arg(long, help_heading = INPUT_OUTPUT)]
    recursive: bool,

    /// Analyze the images and report without encoding or writing any file.
    #[arg(long, help_heading = INPUT_OUTPUT)]
    dry_run: bool,

    /// Replace output files that already exist. Default: an image whose output
    /// exists is skipped. `--overwrite=false` overrides `overwrite = true` in a
    /// config file.
    #[arg(
        long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        value_name = "BOOL",
        help_heading = INPUT_OUTPUT
    )]
    overwrite: Option<bool>,

    /// With --overwrite, refuse to replace an input image or any file in a folder
    /// that holds input images. On by default; `--protect-inputs=false` allows it.
    #[arg(
        long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        value_name = "BOOL",
        help_heading = INPUT_OUTPUT
    )]
    protect_inputs: Option<bool>,

    /// Read settings from this TOML file. Options given here override the file.
    #[arg(long, value_name = "FILE", help_heading = INPUT_OUTPUT)]
    config: Option<PathBuf>,

    /// How demanding the quality floor is. The floor is a score on the scale of
    /// the metric in use; --min-quality replaces it.
    #[arg(long, value_enum, help_heading = QUALITY)]
    preset: Option<PresetArg>,

    /// Lowest score, from 0 to 100, a candidate may have. Replaces the preset's
    /// floor.
    #[arg(long, value_name = "SCORE", help_heading = QUALITY)]
    min_quality: Option<f32>,

    /// Points a score may fall short of the floor and still pass. Default: 0.
    #[arg(long, value_name = "POINTS", help_heading = QUALITY)]
    quality_slack: Option<f32>,

    /// Score that decides acceptance. Default: auto.
    #[arg(long, value_enum, help_heading = QUALITY)]
    metric: Option<MetricArg>,

    /// Score candidates with SSIMULACRA2 (the same as --metric ssimulacra2); an
    /// image fails if the ssimulacra2 tool is missing.
    #[arg(long, help_heading = QUALITY)]
    ssimulacra2: bool,

    /// Also require a Butteraugli distance (the butteraugli_main tool) to pass.
    /// Very slow: over a minute per candidate on a large photo.
    #[arg(long, help_heading = QUALITY)]
    butteraugli: bool,

    /// How many settings are tried per format. More effort finds smaller files
    /// and takes longer. Default: balanced.
    #[arg(long, value_enum, help_heading = QUALITY)]
    speed: Option<SpeedArg>,

    /// Experimental: how the quality floor is searched. staged tries a coarse
    /// grid of qualities and refines around the crossing; predicted starts from
    /// an estimated quality and searches the range its measured error allows,
    /// for photographs scored with SSIMULACRA2 (other images use staged).
    /// Default: staged.
    #[arg(long, value_enum, help_heading = QUALITY)]
    search: Option<SearchArg>,

    /// Accept an output larger than the original. Default: false.
    #[arg(
        long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        value_name = "BOOL",
        help_heading = QUALITY
    )]
    allow_larger: Option<bool>,

    /// What the output is for, which decides which formats may be used.
    /// Default: web.
    #[arg(long, value_enum, help_heading = FORMATS)]
    target: Option<TargetArg>,

    /// Output formats to try, comma separated: jpeg, webp, avif, jxl, png.
    /// Default: jpeg,webp,avif,png. The target may add or remove some.
    #[arg(long, value_name = "LIST", help_heading = FORMATS)]
    formats: Option<String>,

    /// Leave AVIF out of the formats.
    #[arg(long, help_heading = FORMATS)]
    no_avif: bool,

    /// Add JPEG XL to the formats.
    #[arg(long, help_heading = FORMATS)]
    allow_jxl: bool,

    /// What to keep of the image's metadata. Default: safe.
    #[arg(long, value_enum, help_heading = METADATA_ALPHA)]
    metadata: Option<MetadataArg>,

    /// What to do with transparency. Default: preserve.
    #[arg(long, value_enum, help_heading = METADATA_ALPHA)]
    alpha: Option<AlphaArg>,

    /// Colour that flattened transparency is drawn over, as #rrggbb, for
    /// --alpha flatten. Default: #ffffff.
    #[arg(long, value_name = "COLOR", help_heading = METADATA_ALPHA)]
    background: Option<String>,

    /// Images and candidates worked on at once. Default: the number of logical
    /// processors. The limit is 256.
    #[arg(long, value_name = "N", help_heading = LIMITS)]
    workers: Option<usize>,

    /// Longest an encoder or metric process may run, in seconds. Default: 30
    /// with --speed fast, 120 with balanced, 300 with exhaustive.
    #[arg(long, value_name = "SECONDS", help_heading = LIMITS)]
    candidate_timeout_seconds: Option<u64>,

    /// Stop searching an image after this many seconds and keep the best result
    /// found. Default: no limit.
    #[arg(long, value_name = "SECONDS", help_heading = LIMITS)]
    max_image_time_seconds: Option<u64>,

    /// Most candidates tried for one image. Default: no limit.
    #[arg(long, value_name = "N", help_heading = LIMITS)]
    max_candidate_attempts: Option<usize>,

    /// Most temporary-file space one image may use, in bytes; with it,
    /// candidates run one at a time. Default: no limit.
    #[arg(long, value_name = "BYTES", help_heading = LIMITS)]
    max_temp_bytes: Option<u64>,

    /// Soft limit on the temporary files of the whole run, in bytes: work waits
    /// for room above three quarters of it. Default: no limit.
    #[arg(long, value_name = "BYTES", help_heading = LIMITS)]
    max_run_temp_bytes: Option<u64>,

    /// Memory the run plans on using, in bytes, shared between the images, the
    /// candidates and the SSIMULACRA2 processes. Default: two thirds of the
    /// physical memory.
    #[arg(long, value_name = "BYTES", help_heading = LIMITS)]
    max_memory_bytes: Option<u64>,

    /// Largest image accepted, in pixels (width times height). Default: 100000000.
    #[arg(long, value_name = "PIXELS", help_heading = LIMITS)]
    max_pixels: Option<u64>,

    /// Largest input file accepted, in bytes. Default: 2147483648 (2 GiB).
    #[arg(long, value_name = "BYTES", help_heading = LIMITS)]
    max_input_bytes: Option<u64>,

    /// Widest image accepted, in pixels. Default: 100000.
    #[arg(long, value_name = "PIXELS", help_heading = LIMITS)]
    max_width: Option<u32>,

    /// Tallest image accepted, in pixels. Default: 100000.
    #[arg(long, value_name = "PIXELS", help_heading = LIMITS)]
    max_height: Option<u32>,

    /// Most memory the decoded image may take, in bytes. Default: 2147483648
    /// (2 GiB).
    #[arg(long, value_name = "BYTES", help_heading = LIMITS)]
    max_decoded_bytes: Option<u64>,

    /// Most metadata (EXIF, XMP, colour profile) an image may carry, in bytes.
    /// Default: 16777216 (16 MiB).
    #[arg(long, value_name = "BYTES", help_heading = LIMITS)]
    max_metadata_bytes: Option<u64>,

    /// Most bits per pixel a source image may have. Default: 64.
    #[arg(long, value_name = "BITS", help_heading = LIMITS)]
    max_bits_per_pixel: Option<u16>,

    /// Reuse the encoded files and scores of earlier runs, and keep this run's.
    #[arg(long, help_heading = CACHE)]
    cache: bool,

    /// Folder of the cache; turns the cache on. Default: smart-compress/cache in
    /// LOCALAPPDATA on Windows, `~/.cache/smart-compress` elsewhere.
    #[arg(long, value_name = "DIR", help_heading = CACHE)]
    cache_dir: Option<PathBuf>,

    /// Keep the cache under this many bytes by deleting the oldest entries when a
    /// run starts; turns the cache on.
    #[arg(long, value_name = "BYTES", help_heading = CACHE)]
    cache_max_bytes: Option<u64>,

    /// Delete cache entries older than this many days when a run starts; turns
    /// the cache on.
    #[arg(long, value_name = "DAYS", help_heading = CACHE)]
    cache_max_age_days: Option<u64>,

    /// Apply --cache-max-bytes and --cache-max-age-days to the cache now and
    /// exit, without processing any image.
    #[arg(long, help_heading = CACHE)]
    cache_prune: bool,

    /// Score on the CPU only.
    #[arg(long, conflicts_with = "gpu_required", help_heading = GPU)]
    cpu_only: bool,

    /// Fail instead of falling back to the CPU when the GPU cannot be used.
    #[arg(long, help_heading = GPU)]
    gpu_required: bool,

    /// Use the GPU whose name contains this text, ignoring case. Default: the
    /// best hardware GPU. Naming a software adapter selects it.
    #[arg(long, value_name = "TEXT", help_heading = GPU)]
    gpu_adapter_name: Option<String>,

    /// Use the GPU with this PCI vendor id (decimal).
    #[arg(long, value_name = "ID", help_heading = GPU)]
    gpu_adapter_vendor: Option<u32>,

    /// Use the GPU with this device id (decimal).
    #[arg(long, value_name = "ID", help_heading = GPU)]
    gpu_adapter_device: Option<u32>,

    /// Smallest image, in pixels, scored on the GPU; smaller ones are scored on
    /// the CPU. Default: 0, any size.
    #[arg(long, value_name = "PIXELS", help_heading = GPU)]
    gpu_min_pixels: Option<u64>,

    /// Score each candidate on the CPU as well and report where the two differ.
    /// Slow; for checking the GPU.
    #[arg(long, help_heading = GPU)]
    verify_gpu: bool,

    /// How the report is written. Default: text.
    #[arg(long, value_enum, help_heading = REPORTS)]
    report: Option<ReportArg>,

    /// Write the report to this file instead of the terminal. It is rewritten
    /// while the run goes on, so an interrupted run leaves what it finished.
    #[arg(long, value_name = "FILE", help_heading = REPORTS)]
    report_out: Option<PathBuf>,

    /// Skip the images already finished in this JSON report from an earlier run
    /// with the same settings, and carry them into the new report. Needs
    /// --report-out.
    #[arg(long, value_name = "FILE", help_heading = REPORTS)]
    resume_report: Option<PathBuf>,

    /// Check that a JSON report is readable and consistent, then exit.
    #[arg(long, value_name = "FILE", help_heading = REPORTS)]
    validate_report: Option<PathBuf>,

    /// Try every planned candidate: turns off the shortcuts that skip
    /// candidates that cannot win, so the report shows every outcome. Slower.
    #[arg(long, help_heading = REPORTS)]
    verbose: bool,

    /// Do not show the progress bars.
    #[arg(long, help_heading = REPORTS)]
    no_progress: bool,

    /// Open the control panel in the browser instead of running once.
    #[arg(long, help_heading = CONTROL_PANEL)]
    gui: bool,

    /// Show the control panel in its own window and exit when it closes.
    #[arg(long, help_heading = CONTROL_PANEL)]
    gui_window: bool,

    /// Serve the control panel without opening a window or browser tab.
    #[arg(long, help_heading = CONTROL_PANEL)]
    gui_no_open: bool,

    /// Port of the control panel on this computer. Default: 7878.
    #[arg(long, default_value_t = 7878, value_name = "PORT", help_heading = CONTROL_PANEL)]
    gui_port: u16,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PresetArg {
    /// Floor 90 on SSIMULACRA2 (88 on the built-in score): visually lossless.
    Strict,
    /// Floor 80 (73): no difference side by side. The default.
    Balanced,
    /// Floor 70 (68): high quality, smallest files.
    Aggressive,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TargetArg {
    /// JPEG, WebP, AVIF and PNG. A more widely supported format wins over one up
    /// to 10% smaller (strict), 5% (balanced) or 0% (aggressive).
    Web,
    /// For keeping: JPEG XL replaces JPEG.
    Archive,
    /// Adds JPEG XL and takes the smallest file regardless of support.
    Smallest,
    /// Only JPEG and PNG, the formats every program opens.
    MaxCompat,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MetricArg {
    /// SSIMULACRA2 when its tool is installed, otherwise the built-in score.
    Auto,
    /// The perceptual SSIMULACRA2 score; needs the ssimulacra2 tool.
    Ssimulacra2,
    /// The same SSIMULACRA2 score computed inside the program; needs no tool.
    Ssimulacra2Inprocess,
    /// The built-in score (faster, less accurate): the lowest of the RGB, edge
    /// and, with transparency, alpha agreement.
    Builtin,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ReportArg {
    Text,
    Json,
    Markdown,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MetadataArg {
    /// Keep the colour profile and EXIF.
    Keep,
    /// Remove everything, except a colour profile that is not plain sRGB.
    Strip,
    /// Keep the colour profile, drop EXIF and the rest.
    Safe,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AlphaArg {
    /// Keep the image as it is, alpha channel included, even if no pixel is
    /// transparent. An image with an alpha channel is never offered as JPEG.
    Preserve,
    /// Draw the image over --background and drop the alpha channel.
    Flatten,
    /// Drop the alpha channel when every pixel is fully opaque, so JPEG can be
    /// offered; otherwise the same as preserve.
    Auto,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SearchArg {
    Staged,
    Predicted,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SpeedArg {
    Fast,
    Balanced,
    Exhaustive,
}

fn main() -> Result<()> {
    let args = Args::parse();
    smart_compress_codecs::clear_cancel();
    if args.gui || args.gui_window || args.gui_no_open {
        // The panel has its own Cancel; Ctrl+C keeps its default of stopping the server.
        return gui::serve(args.gui_port, args.gui_window, !args.gui_no_open);
    }
    install_cancellation_handler();
    if let Some(message) = run_utility(&args)? {
        println!("{message}");
        return Ok(());
    }
    anyhow::ensure!(!args.input.is_empty(), "at least one input is required");
    let show_progress = !args.no_progress && std::io::stderr().is_terminal();
    let config = build_config(args)?;
    let report_format = config.report;
    let report_out = config.report_out.clone();
    let summary = if show_progress {
        let progress: Arc<dyn ProgressObserver> = Arc::new(TerminalProgress::new());
        Optimizer::with_progress(config, progress).run()?
    } else {
        Optimizer::new(config).run()?
    };
    publish_report(&summary.report, report_format, report_out.as_deref())?;

    if summary.report.failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Runs the maintenance modes that need no inputs and returns their message.
fn run_utility(args: &Args) -> Result<Option<String>> {
    if let Some(path) = &args.validate_report {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading report {}", path.display()))?;
        let report = smart_compress_core::ReportOutput::from_json(&text)
            .with_context(|| format!("parsing report {}", path.display()))?;
        report.validate()?;
        return Ok(Some(format!("valid report: {}", path.display())));
    }
    if args.cache_prune {
        let cache_dir = args
            .cache_dir
            .clone()
            .unwrap_or_else(smart_compress_core::cache::default_cache_dir);
        let cache_config = CacheConfig {
            enabled: true,
            dir: Some(cache_dir),
            max_bytes: args.cache_max_bytes,
            max_age_days: args.cache_max_age_days,
        };
        let tools = smart_compress_codecs::ToolInventory::detect();
        let cache = CandidateCache::from_config(&cache_config, &tools)
            .expect("cache pruning config must enable the cache");
        let removed = cache.prune(cache_config.max_bytes, cache_config.max_age_days)?;
        return Ok(Some(format!("removed {removed} cache files")));
    }
    Ok(None)
}

fn publish_report(
    report: &smart_compress_core::ReportOutput,
    format: ReportFormat,
    report_out: Option<&Path>,
) -> Result<()> {
    let rendered = report.render(format)?;
    if let Some(path) = report_out {
        atomic_write(path, rendered.as_bytes())
            .with_context(|| format!("writing report {}", path.display()))?;
    } else {
        print!("{rendered}");
    }
    Ok(())
}

fn build_config(args: Args) -> Result<CliConfig> {
    let file_config = if let Some(path) = &args.config {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let file_config = parse_file_config(&text)
            .with_context(|| format!("parsing config {}", path.display()))?;
        Some(file_config)
    } else {
        None
    };
    build_config_with_file(args, file_config)
}

fn build_config_with_file(args: Args, file_config: Option<FileConfig>) -> Result<CliConfig> {
    let mut config = CliConfig::default();
    if let Some(file_config) = file_config {
        apply_file_config(&mut config, file_config)?;
    }

    config.inputs = args.input;
    config.out = args.out;
    config.out_dir = args.out_dir.or(config.out_dir);
    config.recursive |= args.recursive;
    config.dry_run = args.dry_run;
    if let Some(overwrite) = args.overwrite {
        config.overwrite = overwrite;
    }
    if let Some(protect_inputs) = args.protect_inputs {
        config.protect_inputs = protect_inputs;
    }
    if let Some(preset) = args.preset {
        config.preset = map_preset(preset);
        // A preset chosen on the command line replaces a floor from the config file.
        config.min_quality = None;
    }
    if let Some(target) = args.target {
        config.target = map_target(target);
    }
    if let Some(formats) = args.formats {
        config.formats = parse_codec_list(&formats)?;
    }
    if args.no_avif {
        config.formats = smart_compress_core::CodecSelection::new(
            config.formats.iter().filter(|codec| *codec != Codec::Avif),
        );
    }
    if args.allow_jxl {
        let mut list = config.formats.iter().collect::<Vec<_>>();
        list.push(smart_compress_codecs::Codec::Jxl);
        config.formats = smart_compress_core::CodecSelection::new(list);
    }
    let requested = config.formats.clone();
    config.formats.apply_target(config.target);
    if config.formats.iter().next().is_none() {
        if requested.iter().next().is_none() {
            anyhow::bail!("no output format is enabled; name at least one with --formats");
        }
        anyhow::bail!(
            "--formats {} leaves no output format under --target {}; choose other formats or another target",
            requested.names(),
            config.target.name()
        );
    }
    if let Some(report) = args.report {
        config.report = map_report(report);
    }
    if let Some(report_out) = args.report_out {
        config.report_out = Some(report_out);
    }
    if let Some(resume_report) = args.resume_report {
        config.resume_report = Some(resume_report);
    }
    if let Some(min_quality) = args.min_quality {
        config.min_quality = Some(min_quality);
    }
    if let Some(quality_slack) = args.quality_slack {
        config.quality_slack = quality_slack;
    }
    if let Some(metadata) = args.metadata {
        config.metadata = map_metadata(metadata);
    }
    if let Some(alpha) = args.alpha {
        config.alpha = match alpha {
            AlphaArg::Preserve => AlphaPolicy::Preserve,
            AlphaArg::Flatten => AlphaPolicy::Flatten {
                background: args.background.unwrap_or_else(|| "#ffffff".to_string()),
            },
            AlphaArg::Auto => AlphaPolicy::Auto,
        };
    } else if let Some(background) = args.background
        && matches!(&config.alpha, AlphaPolicy::Flatten { .. })
    {
        config.alpha = AlphaPolicy::Flatten { background };
    }
    if let Some(speed) = args.speed {
        config.speed = map_speed(speed);
    }
    if let Some(search) = args.search {
        config.search = map_search(search);
    }
    if let Some(workers) = args.workers {
        config.workers = clamp_workers(workers);
    }
    if let Some(seconds) = args.candidate_timeout_seconds {
        anyhow::ensure!(seconds > 0, "candidate timeout must be greater than zero");
        config.candidate_timeout = Some(std::time::Duration::from_secs(seconds));
    }
    if let Some(seconds) = args.max_image_time_seconds {
        anyhow::ensure!(seconds > 0, "max image time must be greater than zero");
        config.max_image_time = Some(std::time::Duration::from_secs(seconds));
    }
    if let Some(attempts) = args.max_candidate_attempts {
        anyhow::ensure!(
            attempts > 0,
            "max candidate attempts must be greater than zero"
        );
        config.max_candidate_attempts = Some(attempts);
    }
    if let Some(bytes) = args.max_temp_bytes {
        anyhow::ensure!(bytes > 0, "max temporary bytes must be greater than zero");
        config.max_temp_bytes = Some(bytes);
    }
    if let Some(bytes) = args.max_run_temp_bytes {
        anyhow::ensure!(
            bytes > 0,
            "maximum run temporary bytes must be greater than zero"
        );
        config.max_run_temp_bytes = Some(bytes);
    }
    if let Some(bytes) = args.max_memory_bytes {
        anyhow::ensure!(bytes > 0, "maximum memory bytes must be greater than zero");
        config.max_memory_bytes = Some(bytes);
    }
    if let Some(allow_larger) = args.allow_larger {
        config.allow_larger = allow_larger;
    }
    if let Some(max_pixels) = args.max_pixels {
        config.max_pixels = max_pixels;
    }
    if let Some(max_input_bytes) = args.max_input_bytes {
        config.max_input_bytes = max_input_bytes;
    }
    if let Some(max_width) = args.max_width {
        config.max_width = max_width;
    }
    if let Some(max_height) = args.max_height {
        config.max_height = max_height;
    }
    if let Some(max_decoded_bytes) = args.max_decoded_bytes {
        config.max_decoded_bytes = max_decoded_bytes;
    }
    if let Some(max_metadata_bytes) = args.max_metadata_bytes {
        config.max_metadata_bytes = max_metadata_bytes;
    }
    if let Some(max_bits_per_pixel) = args.max_bits_per_pixel {
        anyhow::ensure!(
            max_bits_per_pixel > 0,
            "maximum bits per pixel must be greater than zero"
        );
        config.max_bits_per_pixel = max_bits_per_pixel;
    }
    if args.cpu_only {
        config.gpu.enabled = false;
        config.gpu.required = false;
    }
    if args.gpu_required {
        config.gpu.enabled = true;
        config.gpu.required = true;
    }
    if let Some(name) = args.gpu_adapter_name {
        config.gpu.adapter_name = Some(name);
    }
    if let Some(vendor) = args.gpu_adapter_vendor {
        config.gpu.adapter_vendor = Some(vendor);
    }
    if let Some(device) = args.gpu_adapter_device {
        config.gpu.adapter_device = Some(device);
    }
    if let Some(min_pixels) = args.gpu_min_pixels {
        config.gpu.min_pixels_for_gpu_metrics = min_pixels;
    }
    if args.verify_gpu {
        config.gpu.verify_gpu = true;
    }
    if let Some(metric) = args.metric {
        config.metrics.primary = match metric {
            MetricArg::Auto => PrimaryMetric::Auto,
            MetricArg::Ssimulacra2 => PrimaryMetric::Ssimulacra2,
            MetricArg::Ssimulacra2Inprocess => PrimaryMetric::Ssimulacra2Inprocess,
            MetricArg::Builtin => PrimaryMetric::Builtin,
        };
    }
    if args.ssimulacra2 {
        config.metrics.ssimulacra2 = true;
    }
    if args.butteraugli {
        config.metrics.butteraugli = true;
    }
    if args.cache {
        config.cache.enabled = true;
    }
    if let Some(cache_dir) = args.cache_dir {
        config.cache.enabled = true;
        config.cache.dir = Some(cache_dir);
    }
    if let Some(max_bytes) = args.cache_max_bytes {
        anyhow::ensure!(
            max_bytes > 0,
            "cache maximum bytes must be greater than zero"
        );
        config.cache.enabled = true;
        config.cache.max_bytes = Some(max_bytes);
    }
    if let Some(max_age_days) = args.cache_max_age_days {
        anyhow::ensure!(
            max_age_days > 0,
            "cache maximum age must be greater than zero"
        );
        config.cache.enabled = true;
        config.cache.max_age_days = Some(max_age_days);
    }
    config.verbose = args.verbose;
    Ok(config)
}

fn apply_file_config(config: &mut CliConfig, file_config: FileConfig) -> Result<()> {
    if let Some(defaults) = file_config.defaults {
        if let Some(preset) = defaults.preset {
            config.preset = preset;
        }
        if let Some(target) = defaults.target {
            config.target = target;
        }
        if let Some(min_quality) = defaults.min_quality {
            config.min_quality = Some(min_quality);
        }
        if let Some(quality_slack) = defaults.quality_slack {
            config.quality_slack = quality_slack;
        }
        if let Some(report) = defaults.report {
            config.report = report;
        }
        if let Some(report_out) = defaults.report_out {
            config.report_out = Some(report_out);
        }
        if let Some(metadata) = defaults.metadata {
            config.metadata = metadata;
        }
        if let Some(alpha) = defaults.alpha {
            config.alpha = match alpha.as_str() {
                "preserve" => AlphaPolicy::Preserve,
                "flatten" => AlphaPolicy::Flatten {
                    background: "#ffffff".to_string(),
                },
                "auto" => AlphaPolicy::Auto,
                _ => anyhow::bail!("unknown alpha policy '{alpha}'"),
            };
        }
        if let Some(background) = defaults.background
            && matches!(config.alpha, AlphaPolicy::Flatten { .. })
        {
            config.alpha = AlphaPolicy::Flatten { background };
        }
        if let Some(speed) = defaults.speed {
            config.speed = speed;
        }
        if let Some(search) = defaults.search {
            config.search = search;
        }
        if let Some(overwrite) = defaults.overwrite {
            config.overwrite = overwrite;
        }
        if let Some(protect_inputs) = defaults.protect_inputs {
            config.protect_inputs = protect_inputs;
        }
        if let Some(out_dir) = defaults.out_dir {
            config.out_dir = Some(out_dir);
        }
        if let Some(recursive) = defaults.recursive {
            config.recursive = recursive;
        }
        if let Some(allow_larger) = defaults.allow_larger {
            config.allow_larger = allow_larger;
        }
        if let Some(max_pixels) = defaults.max_pixels {
            config.max_pixels = max_pixels;
        }
        if let Some(max_input_bytes) = defaults.max_input_bytes {
            config.max_input_bytes = max_input_bytes;
        }
        if let Some(max_width) = defaults.max_width {
            config.max_width = max_width;
        }
        if let Some(max_height) = defaults.max_height {
            config.max_height = max_height;
        }
        if let Some(max_decoded_bytes) = defaults.max_decoded_bytes {
            config.max_decoded_bytes = max_decoded_bytes;
        }
        if let Some(max_metadata_bytes) = defaults.max_metadata_bytes {
            config.max_metadata_bytes = max_metadata_bytes;
        }
        if let Some(max_bits_per_pixel) = defaults.max_bits_per_pixel {
            anyhow::ensure!(
                max_bits_per_pixel > 0,
                "maximum bits per pixel must be greater than zero"
            );
            config.max_bits_per_pixel = max_bits_per_pixel;
        }
    }
    if let Some(performance) = file_config.performance {
        if let Some(workers) = performance.workers {
            config.workers = clamp_workers(workers);
        }
        if let Some(seconds) = performance.candidate_timeout_seconds {
            anyhow::ensure!(seconds > 0, "candidate timeout must be greater than zero");
            config.candidate_timeout = Some(std::time::Duration::from_secs(seconds));
        }
        if let Some(seconds) = performance.max_image_time_seconds {
            anyhow::ensure!(seconds > 0, "max image time must be greater than zero");
            config.max_image_time = Some(std::time::Duration::from_secs(seconds));
        }
        if let Some(attempts) = performance.max_candidate_attempts {
            anyhow::ensure!(
                attempts > 0,
                "max candidate attempts must be greater than zero"
            );
            config.max_candidate_attempts = Some(attempts);
        }
        if let Some(bytes) = performance.max_temp_bytes {
            anyhow::ensure!(bytes > 0, "max temporary bytes must be greater than zero");
            config.max_temp_bytes = Some(bytes);
        }
        if let Some(bytes) = performance.max_run_temp_bytes {
            anyhow::ensure!(
                bytes > 0,
                "maximum run temporary bytes must be greater than zero"
            );
            config.max_run_temp_bytes = Some(bytes);
        }
        if let Some(bytes) = performance.max_memory_bytes {
            anyhow::ensure!(bytes > 0, "maximum memory bytes must be greater than zero");
            config.max_memory_bytes = Some(bytes);
        }
    }
    if let Some(gpu) = file_config.gpu {
        config.gpu = gpu;
    }
    if let Some(cache) = file_config.cache {
        config.cache = cache;
    }
    if let Some(metrics) = file_config.metrics {
        config.metrics = metrics;
    }
    if let Some(codecs) = file_config.codecs {
        let mut enabled = Vec::new();
        if codecs.png.unwrap_or(true) {
            enabled.push(smart_compress_codecs::Codec::Png);
        }
        if codecs.jpeg.unwrap_or(true) {
            enabled.push(smart_compress_codecs::Codec::Jpeg);
        }
        if codecs.webp.unwrap_or(true) {
            enabled.push(smart_compress_codecs::Codec::Webp);
        }
        if codecs.avif.unwrap_or(true) {
            enabled.push(smart_compress_codecs::Codec::Avif);
        }
        if codecs.jxl.unwrap_or(false) {
            enabled.push(smart_compress_codecs::Codec::Jxl);
        }
        config.formats = smart_compress_core::CodecSelection::new(enabled);
    }
    Ok(())
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

fn map_report(value: ReportArg) -> ReportFormat {
    match value {
        ReportArg::Text => ReportFormat::Text,
        ReportArg::Json => ReportFormat::Json,
        ReportArg::Markdown => ReportFormat::Markdown,
    }
}

fn map_metadata(value: MetadataArg) -> MetadataPolicy {
    match value {
        MetadataArg::Keep => MetadataPolicy::Keep,
        MetadataArg::Strip => MetadataPolicy::Strip,
        MetadataArg::Safe => MetadataPolicy::Safe,
    }
}

fn map_search(value: SearchArg) -> SearchMode {
    match value {
        SearchArg::Staged => SearchMode::Staged,
        SearchArg::Predicted => SearchMode::Predicted,
    }
}

fn map_speed(value: SpeedArg) -> SpeedPreset {
    match value {
        SpeedArg::Fast => SpeedPreset::Fast,
        SpeedArg::Balanced => SpeedPreset::Balanced,
        SpeedArg::Exhaustive => SpeedPreset::Exhaustive,
    }
}

struct TerminalProgress {
    multi: MultiProgress,
    batch: Mutex<Option<ProgressBar>>,
    files: Mutex<HashMap<PathBuf, ProgressBar>>,
}

impl TerminalProgress {
    fn new() -> Self {
        Self {
            multi: MultiProgress::new(),
            batch: Mutex::new(None),
            files: Mutex::new(HashMap::new()),
        }
    }

    fn file_label(path: &Path) -> String {
        path.file_name()
            .and_then(|name| name.to_str())
            .map_or_else(|| path.display().to_string(), str::to_string)
    }

    fn bar_style() -> ProgressStyle {
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {bar:32.cyan/blue} {pos}/{len} {wide_msg}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=> ")
    }

    fn spinner_style() -> ProgressStyle {
        ProgressStyle::with_template("{spinner:.green} {wide_msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
    }
}

impl ProgressObserver for TerminalProgress {
    fn batch_started(&self, total_files: usize) {
        let bar = self.multi.add(ProgressBar::new(total_files as u64));
        bar.set_style(Self::bar_style());
        bar.set_message("files");
        *self.batch.lock().expect("batch progress lock poisoned") = Some(bar);
    }

    fn batch_finished(&self) {
        if let Some(bar) = self
            .batch
            .lock()
            .expect("batch progress lock poisoned")
            .take()
        {
            bar.finish_and_clear();
        }
    }

    fn file_started(&self, path: &Path) {
        let bar = self.multi.add(ProgressBar::new_spinner());
        bar.set_style(Self::spinner_style());
        bar.enable_steady_tick(std::time::Duration::from_millis(120));
        bar.set_message(format!("probing {}", Self::file_label(path)));
        self.files
            .lock()
            .expect("file progress lock poisoned")
            .insert(path.to_path_buf(), bar);
    }

    fn file_finished(&self, path: &Path) {
        if let Some(bar) = self
            .files
            .lock()
            .expect("file progress lock poisoned")
            .remove(path)
        {
            bar.finish_and_clear();
        }
        if let Some(batch) = self
            .batch
            .lock()
            .expect("batch progress lock poisoned")
            .as_ref()
        {
            batch.inc(1);
        }
    }

    fn candidates_planned(&self, path: &Path, total_candidates: usize) {
        if let Some(bar) = self
            .files
            .lock()
            .expect("file progress lock poisoned")
            .get(path)
        {
            bar.set_style(Self::bar_style());
            bar.set_length(total_candidates as u64);
            bar.set_position(0);
            bar.set_message(format!("{} candidates", Self::file_label(path)));
        }
    }

    fn candidate_started(
        &self,
        path: &Path,
        index: usize,
        total: usize,
        codec: Codec,
        settings: &EncoderSettings,
    ) {
        if let Some(bar) = self
            .files
            .lock()
            .expect("file progress lock poisoned")
            .get(path)
        {
            bar.set_message(format!(
                "{} {index}/{total} {} {}",
                Self::file_label(path),
                codec.name(),
                settings.label()
            ));
        }
    }

    fn candidate_finished(&self, path: &Path, _index: usize, _total: usize) {
        if let Some(bar) = self
            .files
            .lock()
            .expect("file progress lock poisoned")
            .get(path)
        {
            bar.inc(1);
        }
    }

    fn candidates_settled(&self, path: &Path) {
        if let Some(bar) = self
            .files
            .lock()
            .expect("file progress lock poisoned")
            .get(path)
        {
            bar.set_length(bar.position());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workers_are_clamped_to_a_sane_range() {
        assert_eq!(clamp_workers(0), 1);
        assert_eq!(clamp_workers(6), 6);
        assert_eq!(clamp_workers(usize::MAX), MAX_WORKERS);
    }

    struct ExpectedConfig {
        preset: Preset,
        target: TargetPolicy,
        formats: Vec<Codec>,
        report: ReportFormat,
        min_quality: Option<f32>,
        quality_slack: f32,
        metadata: MetadataPolicy,
        alpha: AlphaPolicy,
        speed: SpeedPreset,
        overwrite: bool,
        max_pixels: u64,
        workers: usize,
        max_memory_bytes: Option<u64>,
        gpu_enabled: bool,
        gpu_required: bool,
        verify_gpu: bool,
        cache_enabled: bool,
        ssimulacra2: bool,
        butteraugli: bool,
    }

    struct PrecedenceCase {
        name: &'static str,
        argv: &'static [&'static str],
        file_config: Option<&'static str>,
        expected: ExpectedConfig,
    }

    #[test]
    fn configuration_precedence_is_layered() {
        const FILE_CONFIG: &str = r##"
            [defaults]
            preset = "strict"
            target = "web"
            min_quality = 93.0
            metadata = "strip"
            alpha = "auto"
            overwrite = true

            [codecs]
            png = true
            jpeg = false
            webp = false
            avif = false
            jxl = true

            [performance]
            workers = 3
            max_memory_bytes = 4000000000

            [gpu]
            enabled = false
            verify_gpu = true

            [cache]
            enabled = true
            dir = "cache"

            [metrics]
            ssimulacra2 = true
            butteraugli = false
        "##;

        let defaults = CliConfig::default();
        let cases = [
            PrecedenceCase {
                name: "default-only",
                argv: &["smart-compress", "input.png"],
                file_config: None,
                expected: ExpectedConfig {
                    preset: Preset::Balanced,
                    target: TargetPolicy::Web,
                    formats: vec![Codec::Jpeg, Codec::Webp, Codec::Avif, Codec::Png],
                    report: ReportFormat::Text,
                    min_quality: None,
                    quality_slack: 0.0,
                    metadata: MetadataPolicy::Safe,
                    alpha: AlphaPolicy::Preserve,
                    speed: SpeedPreset::Balanced,
                    overwrite: false,
                    max_pixels: 100_000_000,
                    workers: defaults.workers,
                    max_memory_bytes: None,
                    gpu_enabled: true,
                    gpu_required: false,
                    verify_gpu: false,
                    cache_enabled: false,
                    ssimulacra2: false,
                    butteraugli: false,
                },
            },
            PrecedenceCase {
                name: "config-only",
                argv: &["smart-compress", "input.png"],
                file_config: Some(FILE_CONFIG),
                expected: ExpectedConfig {
                    preset: Preset::Strict,
                    target: TargetPolicy::Web,
                    formats: vec![Codec::Jxl, Codec::Png],
                    report: ReportFormat::Text,
                    min_quality: Some(93.0),
                    quality_slack: 0.0,
                    metadata: MetadataPolicy::Strip,
                    alpha: AlphaPolicy::Auto,
                    speed: SpeedPreset::Balanced,
                    overwrite: true,
                    max_pixels: 100_000_000,
                    workers: 3,
                    max_memory_bytes: Some(4_000_000_000),
                    gpu_enabled: false,
                    gpu_required: false,
                    verify_gpu: true,
                    cache_enabled: true,
                    ssimulacra2: true,
                    butteraugli: false,
                },
            },
            PrecedenceCase {
                name: "cli-only",
                argv: &[
                    "smart-compress",
                    "input.png",
                    "--preset",
                    "aggressive",
                    "--target",
                    "web",
                    "--formats",
                    "jpeg,webp",
                    "--report",
                    "json",
                    "--min-quality",
                    "86",
                    "--quality-slack",
                    "1.25",
                    "--metadata",
                    "keep",
                    "--alpha",
                    "flatten",
                    "--background",
                    "#123456",
                    "--speed",
                    "fast",
                    "--overwrite",
                    "--max-pixels",
                    "42",
                    "--workers",
                    "5",
                    "--max-memory-bytes",
                    "2000000000",
                    "--cpu-only",
                    "--cache",
                    "--ssimulacra2",
                    "--butteraugli",
                ],
                file_config: None,
                expected: ExpectedConfig {
                    preset: Preset::Aggressive,
                    target: TargetPolicy::Web,
                    formats: vec![Codec::Jpeg, Codec::Webp],
                    report: ReportFormat::Json,
                    min_quality: Some(86.0),
                    quality_slack: 1.25,
                    metadata: MetadataPolicy::Keep,
                    alpha: AlphaPolicy::Flatten {
                        background: "#123456".to_string(),
                    },
                    speed: SpeedPreset::Fast,
                    overwrite: true,
                    max_pixels: 42,
                    workers: 5,
                    max_memory_bytes: Some(2_000_000_000),
                    gpu_enabled: false,
                    gpu_required: false,
                    verify_gpu: false,
                    cache_enabled: true,
                    ssimulacra2: true,
                    butteraugli: true,
                },
            },
            PrecedenceCase {
                name: "config-plus-cli",
                argv: &[
                    "smart-compress",
                    "input.png",
                    "--preset",
                    "aggressive",
                    "--target",
                    "max-compat",
                    "--formats",
                    "jpeg,webp",
                    "--report",
                    "markdown",
                    "--min-quality",
                    "87",
                    "--quality-slack",
                    "1.5",
                    "--metadata",
                    "keep",
                    "--alpha",
                    "flatten",
                    "--background",
                    "#abcdef",
                    "--speed",
                    "exhaustive",
                    "--overwrite=false",
                    "--max-pixels",
                    "84",
                    "--workers",
                    "7",
                    "--max-memory-bytes",
                    "3000000000",
                    "--gpu-required",
                    "--butteraugli",
                ],
                file_config: Some(FILE_CONFIG),
                expected: ExpectedConfig {
                    preset: Preset::Aggressive,
                    target: TargetPolicy::MaxCompat,
                    formats: vec![Codec::Jpeg],
                    report: ReportFormat::Markdown,
                    min_quality: Some(87.0),
                    quality_slack: 1.5,
                    metadata: MetadataPolicy::Keep,
                    alpha: AlphaPolicy::Flatten {
                        background: "#abcdef".to_string(),
                    },
                    speed: SpeedPreset::Exhaustive,
                    overwrite: false,
                    max_pixels: 84,
                    workers: 7,
                    max_memory_bytes: Some(3_000_000_000),
                    gpu_enabled: true,
                    gpu_required: true,
                    verify_gpu: true,
                    cache_enabled: true,
                    ssimulacra2: true,
                    butteraugli: true,
                },
            },
        ];

        for case in cases {
            let args = Args::try_parse_from(case.argv).unwrap_or_else(|error| {
                panic!("{}: failed to parse arguments: {error}", case.name)
            });
            let file_config = case
                .file_config
                .map(|text| parse_file_config(text).expect("valid test config"));
            let actual = build_config_with_file(args, file_config)
                .unwrap_or_else(|error| panic!("{}: failed to build config: {error}", case.name));
            let expected = case.expected;

            assert_eq!(actual.preset, expected.preset, "{}: preset", case.name);
            assert_eq!(actual.target, expected.target, "{}: target", case.name);
            assert_eq!(
                actual.formats.iter().collect::<Vec<_>>(),
                expected.formats,
                "{}: formats",
                case.name
            );
            assert_eq!(actual.report, expected.report, "{}: report", case.name);
            assert_eq!(
                actual.min_quality, expected.min_quality,
                "{}: min quality",
                case.name
            );
            assert_eq!(
                actual.quality_slack, expected.quality_slack,
                "{}: quality slack",
                case.name
            );
            assert_eq!(
                actual.metadata, expected.metadata,
                "{}: metadata",
                case.name
            );
            assert_eq!(actual.alpha, expected.alpha, "{}: alpha", case.name);
            assert_eq!(actual.speed, expected.speed, "{}: speed", case.name);
            assert_eq!(
                actual.overwrite, expected.overwrite,
                "{}: overwrite",
                case.name
            );
            assert_eq!(
                actual.max_pixels, expected.max_pixels,
                "{}: max pixels",
                case.name
            );
            assert_eq!(actual.workers, expected.workers, "{}: workers", case.name);
            assert_eq!(
                actual.max_memory_bytes, expected.max_memory_bytes,
                "{}: max memory bytes",
                case.name
            );
            assert_eq!(
                actual.gpu.enabled, expected.gpu_enabled,
                "{}: GPU enabled",
                case.name
            );
            assert_eq!(
                actual.gpu.required, expected.gpu_required,
                "{}: GPU required",
                case.name
            );
            assert_eq!(
                actual.gpu.verify_gpu, expected.verify_gpu,
                "{}: GPU verification",
                case.name
            );
            assert_eq!(
                actual.cache.enabled, expected.cache_enabled,
                "{}: cache",
                case.name
            );
            assert_eq!(
                actual.metrics.ssimulacra2, expected.ssimulacra2,
                "{}: SSIMULACRA2",
                case.name
            );
            assert_eq!(
                actual.metrics.butteraugli, expected.butteraugli,
                "{}: Butteraugli",
                case.name
            );
        }
    }

    #[test]
    fn the_search_mode_is_staged_unless_the_flag_or_the_file_says_otherwise() {
        let search = |argv: &[&str], file: Option<&str>| {
            let args = Args::try_parse_from(
                std::iter::once("smart-compress")
                    .chain(["input.png"])
                    .chain(argv.iter().copied()),
            )
            .unwrap();
            let file = file.map(|text| parse_file_config(text).unwrap());
            build_config_with_file(args, file).unwrap().search
        };
        let predicted = "[defaults]\nsearch = \"predicted\"";
        let staged = "[defaults]\nsearch = \"staged\"";
        assert_eq!(search(&[], None), SearchMode::Staged);
        assert_eq!(
            search(&["--search", "predicted"], None),
            SearchMode::Predicted
        );
        assert_eq!(search(&[], Some(predicted)), SearchMode::Predicted);
        // The command line wins over the file.
        assert_eq!(
            search(&["--search", "staged"], Some(predicted)),
            SearchMode::Staged
        );
        assert_eq!(
            search(&["--search", "predicted"], Some(staged)),
            SearchMode::Predicted
        );
        assert!(
            Args::try_parse_from(["smart-compress", "input.png", "--search", "binary"]).is_err()
        );
        assert!(parse_file_config("[defaults]\nsearch = \"binary\"").is_err());
    }

    #[test]
    fn a_format_set_the_target_empties_is_rejected_by_name() {
        let build = |argv: &[&str]| {
            let args =
                Args::try_parse_from(std::iter::once("smart-compress").chain(argv.iter().copied()))
                    .unwrap();
            build_config_with_file(args, None)
        };
        for (argv, expected) in [
            (
                &["--formats", "webp", "--target", "max-compat"][..],
                "--formats webp leaves no output format under --target max-compat",
            ),
            (
                &["--formats", "avif", "--no-avif"][..],
                "no output format is enabled",
            ),
        ] {
            let error = build(&[&["input.png"], argv].concat()).unwrap_err();
            assert!(error.to_string().starts_with(expected), "{error}");
        }
        assert!(
            build(&[
                "input.png",
                "--formats",
                "webp,jpeg",
                "--target",
                "max-compat"
            ])
            .is_ok()
        );
        // Archive swaps JPEG for JPEG XL, which leaves a format rather than none.
        let archive = build(&["input.png", "--formats", "jpeg", "--target", "archive"]).unwrap();
        assert_eq!(archive.formats.names(), "jxl");
    }

    fn build(argv: &[&str], file_config: Option<&str>) -> Result<CliConfig> {
        let args =
            Args::try_parse_from(std::iter::once("smart-compress").chain(argv.iter().copied()))
                .unwrap();
        build_config_with_file(
            args,
            file_config.map(|text| parse_file_config(text).unwrap()),
        )
    }

    #[test]
    fn overwrite_and_protect_inputs_are_switches() {
        const OVERWRITE_ON: &str = "[defaults]\noverwrite = true";
        const PROTECT_OFF: &str = "[defaults]\nprotect_inputs = false";
        let config = |argv: &[&str], file: Option<&str>| {
            build(&[&["input.png"], argv].concat(), file).unwrap()
        };
        // Defaults: skip existing outputs, protect inputs.
        let base = config(&[], None);
        assert!(!base.overwrite && base.protect_inputs);
        // A switch, which does not swallow the input that follows it.
        let switched = build(&["--overwrite", "input.png"], None).unwrap();
        assert!(switched.overwrite);
        assert_eq!(switched.inputs, [PathBuf::from("input.png")]);
        assert!(config(&["--overwrite=true"], None).overwrite);
        assert!(!config(&["--overwrite=false"], Some(OVERWRITE_ON)).overwrite);
        assert!(!config(&["--protect-inputs=false"], None).protect_inputs);
        assert!(config(&["--protect-inputs"], Some(PROTECT_OFF)).protect_inputs);
        // --allow-larger is a switch the same way.
        let larger = build(&["--allow-larger", "input.png"], None).unwrap();
        assert!(larger.allow_larger && larger.inputs.len() == 1);
        assert!(config(&["--allow-larger=true"], None).allow_larger);
        assert!(!config(&["--allow-larger=false"], None).allow_larger);
    }

    #[test]
    fn the_in_process_ssimulacra2_metric_is_selected_by_flag_and_by_file() {
        let config = build(&["input.png", "--metric", "ssimulacra2-inprocess"], None).unwrap();
        assert_eq!(config.metrics.primary, PrimaryMetric::Ssimulacra2Inprocess);
        let file = "[metrics]\nprimary = \"ssimulacra2-inprocess\"";
        let config = build(&["input.png"], Some(file)).unwrap();
        assert_eq!(config.metrics.primary, PrimaryMetric::Ssimulacra2Inprocess);
    }

    #[test]
    fn the_builtin_metric_is_selected_by_flag_and_by_file() {
        let config = build(&["input.png", "--metric", "builtin"], None).unwrap();
        assert_eq!(config.metrics.primary, PrimaryMetric::Builtin);
        let file = "[metrics]
primary = \"builtin\"";
        let config = build(&["input.png"], Some(file)).unwrap();
        assert_eq!(config.metrics.primary, PrimaryMetric::Builtin);
    }

    #[test]
    fn the_config_file_can_name_the_output_folder_and_recursion() {
        let file = "[defaults]\nout_dir = \"results\"\nrecursive = true";
        let config = build(&["folder"], Some(file)).unwrap();
        assert_eq!(config.out_dir, Some(PathBuf::from("results")));
        assert!(config.recursive);
        let overridden = build(&["folder", "--out-dir", "elsewhere"], Some(file)).unwrap();
        assert_eq!(overridden.out_dir, Some(PathBuf::from("elsewhere")));
    }

    #[test]
    fn the_version_flag_is_accepted() {
        let error = Args::try_parse_from(["smart-compress", "--version"]).unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::DisplayVersion);
    }

    #[test]
    fn cpu_only_conflicts_with_gpu_required() {
        let error = Args::try_parse_from([
            "smart-compress",
            "input.png",
            "--cpu-only",
            "--gpu-required",
        ])
        .expect_err("contradictory GPU options should be rejected");

        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn candidate_timeout_uses_config_and_explicit_cli_override() {
        let file_config: FileConfig = parse_file_config(
            r#"
                [performance]
                candidate_timeout_seconds = 45
            "#,
        )
        .unwrap();
        let config_args = Args::try_parse_from(["smart-compress", "input.png"]).unwrap();
        let config = build_config_with_file(config_args, Some(file_config)).unwrap();
        assert_eq!(config.candidate_timeout().as_secs(), 45);

        let cli_args = Args::try_parse_from([
            "smart-compress",
            "input.png",
            "--candidate-timeout-seconds",
            "12",
        ])
        .unwrap();
        let file_config: FileConfig = parse_file_config(
            r#"
                [performance]
                candidate_timeout_seconds = 45
            "#,
        )
        .unwrap();
        let config = build_config_with_file(cli_args, Some(file_config)).unwrap();
        assert_eq!(config.candidate_timeout().as_secs(), 12);
    }

    #[test]
    fn candidate_timeout_rejects_zero() {
        let args = Args::try_parse_from([
            "smart-compress",
            "input.png",
            "--candidate-timeout-seconds",
            "0",
        ])
        .unwrap();

        let error = build_config_with_file(args, None).unwrap_err();
        assert_eq!(
            error.to_string(),
            "candidate timeout must be greater than zero"
        );
    }
}
