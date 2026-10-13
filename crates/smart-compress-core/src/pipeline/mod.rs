use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, AtomicUsize},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use rayon::ThreadPoolBuilder;
use smart_compress_codecs::{
    Budget, Codec, EncoderBackend, EncoderSettings, MetadataPayload, ToolInventory, WaitEnded,
    acquire_work, cancellation_requested, configure_work_budget,
};

use crate::{
    cache::CandidateCache,
    config::{CliConfig, PrimaryMetric},
    decision::DecisionPolicy,
    gpu::{GpuReport, GpuStatus, detect_gpu_with_adapter},
    memory,
    progress::{NoopProgress, ProgressObserver},
    report::{ImageOutcome, ImageReport, ReportOutput, RunStatus},
    scoring::{
        ActiveComputeBackend, CpuRayonBackend, GpuPreparedOriginal, GpuWgpuBackend, OriginalImage,
    },
};

use self::batch::stable_config_text;

mod batch;
mod evaluate;
mod guards;
mod inputs;
mod output;
mod plan;
mod process;
mod score;
mod search;
mod source;
mod stages;
#[cfg(test)]
mod tests;

const SCORING_VERSION: &str = "score-v10";

/// Folder next to an input that outputs go to when no output folder is given.
const DEFAULT_OUTPUT_DIR: &str = "optimized";

/// Threads the built-in CPU metric runs on, and the units it takes from the
/// work budget while it runs. On a 7.7 MP photo the metric took 89.5, 73.6, 40.8
/// and 26.1 ms a candidate on 1, 2, 4 and 7 threads (each thread worth 61%,
/// 55% and 49% of a thread beyond the first), against 1.3 s of external scoring
/// and 1.1 s of encoding for the same candidate. It is about 7% of a candidate,
/// so it gets two threads: enough to take its latency off the critical path, few
/// enough that it is not where the budget goes.
const DEFAULT_METRIC_THREADS: usize = 2;

/// One unit of the work budget for an in-process step on a single thread.
fn work(units: usize) -> Result<smart_compress_codecs::Permit<'static>> {
    Ok(acquire_work(units, None)?)
}

fn wait_stop_reason(ended: WaitEnded, image_time: Option<std::time::Duration>) -> String {
    match (ended, image_time) {
        (WaitEnded::TimedOut, Some(limit)) => {
            format!("image wall-time budget of {limit:?} reached")
        }
        _ => "cancelled".to_string(),
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

/// The folder a file is in; the working directory for a bare name.
fn folder_of(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn count_failures(reports: &[ImageReport]) -> usize {
    reports
        .iter()
        .filter(|report| report.outcome == ImageOutcome::Failed)
        .count()
}

/// The quality floor on the scale of the metric an image is scored with.
fn quality_floor(config: &CliConfig, perceptual: bool) -> f32 {
    config
        .min_quality
        .unwrap_or_else(|| config.preset.min_quality(perceptual))
}

/// The format a file name asks for, by its extension.
fn codec_for_extension(extension: &str) -> Option<Codec> {
    match extension.to_ascii_lowercase().as_str() {
        "png" => Some(Codec::Png),
        "jpg" | "jpeg" => Some(Codec::Jpeg),
        "webp" => Some(Codec::Webp),
        "avif" => Some(Codec::Avif),
        "jxl" => Some(Codec::Jxl),
        _ => None,
    }
}

fn codec_name(codec: Codec) -> String {
    format!("{codec:?}").to_ascii_lowercase()
}

#[derive(Debug, Clone)]
pub struct BatchSummary {
    pub report: ReportOutput,
}

pub struct Optimizer {
    config: CliConfig,
    backend: Arc<EncoderBackend>,
    gpu: Arc<GpuReport>,
    compute: Arc<ActiveComputeBackend>,
    progress: Arc<dyn ProgressObserver>,
    cache: Option<Arc<CandidateCache>>,
    run_id: String,
    run_started_unix_ms: u64,
    config_fingerprint: String,
    run_temp_usage: Arc<AtomicU64>,
    ambiguous_outputs: Mutex<HashSet<PathBuf>>,
    /// Folders that hold an input, as `fs::canonicalize` names them.
    input_folders: Mutex<HashSet<PathBuf>>,
    use_ssimulacra2: bool,
    /// The floor on the scale of the metric in use.
    min_quality: f32,
    /// The format `--out` names by its extension; only it is tried.
    out_codec: Option<Codec>,
    /// Things about the run as a whole, for the report.
    run_warnings: Mutex<Vec<String>>,
    /// Memory the run plans on using, shared by the three budgets below.
    memory_limit: u64,
    /// Candidates being encoded or scored right now, across all images.
    active_candidates: AtomicUsize,
    /// Images past admission and not yet finished.
    active_files: AtomicUsize,
    /// Runs the candidates of every image and the built-in metrics; file
    /// workers are plain threads that only wait on it.
    pool: OnceLock<rayon::ThreadPool>,
    /// Threads for the built-in CPU metric; a metric holds this many units of
    /// the work budget, so its parallelism is what the budget counted.
    metric_pool: rayon::ThreadPool,
    metric_threads: usize,
    /// Bytes of memory images reserve before they are decoded, and bytes
    /// candidates reserve before they start. Separate budgets, so an image
    /// that holds its own memory never waits on candidates waiting on it.
    file_memory: Budget,
    candidate_memory: Budget,
    /// Bytes the external metric processes of all candidates may hold at once:
    /// about three at a time on 12 MP photos. Taken after the candidate memory
    /// and held only while waiting for a unit of work.
    metric_memory: Budget,
    /// Woken when temporary files are freed or a candidate finishes, for
    /// candidates waiting for room under the soft temporary-disk limit.
    temp_gate: Arc<Budget>,
}

impl Optimizer {
    pub fn new(config: CliConfig) -> Self {
        Self::with_progress(config, Arc::new(NoopProgress))
    }

    pub fn with_progress(config: CliConfig, progress: Arc<dyn ProgressObserver>) -> Self {
        configure_work_budget(config.workers);
        let run_started_unix_ms = unix_time_ms();
        let mut fingerprint_config = config.clone();
        fingerprint_config.report_out = None;
        fingerprint_config.resume_report = None;
        let config_fingerprint = blake3::hash(stable_config_text(&fingerprint_config).as_bytes())
            .to_hex()
            .to_string();
        let run_id = blake3::hash(format!("{run_started_unix_ms}:{config_fingerprint}").as_bytes())
            .to_hex()
            .to_string();
        let tools = ToolInventory::detect();
        let use_ssimulacra2 = config.metrics.ssimulacra2
            || match config.metrics.primary {
                PrimaryMetric::Auto => tools.ssimulacra2.enabled,
                PrimaryMetric::Ssimulacra2 => true,
                PrimaryMetric::Builtin => false,
            };
        let min_quality = quality_floor(&config, use_ssimulacra2);
        let out_codec = config
            .out
            .as_deref()
            .and_then(Path::extension)
            .and_then(|extension| extension.to_str())
            .and_then(codec_for_extension);
        let cache = CandidateCache::from_config(&config.cache, &tools).map(|cache| {
            let _ = cache.prune(config.cache.max_bytes, config.cache.max_age_days);
            Arc::new(cache)
        });
        let (mut gpu, gpu_adapter) = detect_gpu_with_adapter(&config.gpu);
        let use_gpu_metrics = config.gpu.enabled
            && config.gpu.prefer_gpu_metrics
            && matches!(gpu.status, GpuStatus::Enabled);
        let compute = if let Some(adapter) = gpu_adapter.filter(|_| use_gpu_metrics) {
            match GpuWgpuBackend::from_adapter(adapter, config.gpu.max_vram_bytes) {
                Ok(backend) => ActiveComputeBackend::Gpu(Box::new(backend)),
                Err(error) => {
                    gpu.warnings
                        .push(format!("GPU metric backend unavailable: {error:#}"));
                    if config.gpu.required {
                        gpu.status = GpuStatus::RequiredUnavailable;
                    }
                    ActiveComputeBackend::Cpu(CpuRayonBackend)
                }
            }
        } else {
            ActiveComputeBackend::Cpu(CpuRayonBackend)
        };
        let metric_threads = config
            .parallelism
            .metric_threads
            .unwrap_or(DEFAULT_METRIC_THREADS)
            .clamp(1, config.workers.max(1));
        let metric_pool = ThreadPoolBuilder::new()
            .num_threads(metric_threads)
            .build()
            .expect("building metric pool");
        let memory_limit = config
            .max_memory_bytes
            .unwrap_or_else(|| memory::default_memory_limit(memory::physical_memory()));
        let shares = memory::shares(memory_limit);
        Self {
            config,
            backend: Arc::new(EncoderBackend::new(tools)),
            gpu: Arc::new(gpu),
            compute: Arc::new(compute),
            progress,
            cache,
            run_id,
            run_started_unix_ms,
            config_fingerprint,
            run_temp_usage: Arc::new(AtomicU64::new(0)),
            ambiguous_outputs: Mutex::new(HashSet::new()),
            input_folders: Mutex::new(HashSet::new()),
            use_ssimulacra2,
            min_quality,
            out_codec,
            run_warnings: Mutex::new(Vec::new()),
            memory_limit,
            active_candidates: AtomicUsize::new(0),
            active_files: AtomicUsize::new(0),
            pool: OnceLock::new(),
            metric_pool,
            metric_threads,
            file_memory: Budget::new(shares.files),
            candidate_memory: Budget::new(shares.candidates),
            metric_memory: Budget::new(shares.external_metric),
            temp_gate: Arc::new(Budget::new(1)),
        }
    }

    pub fn run(&self) -> Result<BatchSummary> {
        let (inputs, warnings) = self.discover_inputs()?;
        self.check_out_target(inputs.len())?;
        *self.run_warnings.lock().expect("run warning lock poisoned") = warnings;
        *self
            .ambiguous_outputs
            .lock()
            .expect("output name lock poisoned") = self.colliding_inputs(&inputs);
        *self
            .input_folders
            .lock()
            .expect("input folder lock poisoned") = inputs
            .iter()
            .filter_map(|input| fs::canonicalize(folder_of(input)).ok())
            .collect();
        self.progress.batch_started(inputs.len());
        if self.config.report_out.is_some() {
            return self.run_with_checkpoints(inputs);
        }
        self.pool()?;
        let reports = self.process_files(&inputs);
        self.progress.batch_finished();
        let failures = count_failures(&reports);
        let status = if cancellation_requested() {
            RunStatus::Partial
        } else {
            RunStatus::Complete
        };
        let manifest = self.manifest(&reports, inputs.len(), status, None);
        Ok(BatchSummary {
            report: self.report_output(reports, failures, manifest),
        })
    }

    /// The pool candidates run on, built on first use. Twice as many threads as
    /// units of work: a candidate waiting for its units holds a thread and no
    /// CPU, and a few waiting keep the budget full while others start up.
    fn pool(&self) -> Result<&rayon::ThreadPool> {
        if let Some(pool) = self.pool.get() {
            return Ok(pool);
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(self.config.workers.max(1) * 2)
            .build()
            .context("building worker pool")?;
        Ok(self.pool.get_or_init(|| pool))
    }
}

/// A coarse plan split into the candidates to run first and the ones held back.
struct StagedPlan {
    first: Vec<(Codec, EncoderSettings)>,
    held: Vec<(Codec, EncoderSettings)>,
    /// The middle quality of every lossy format in `first`.
    mids: HashMap<Codec, u8>,
}

struct DecodedCandidate {
    image: image::DynamicImage,
    process_ms: Option<u64>,
    io_ms: u64,
}

struct CandidateEvaluationRequest<'a> {
    input: &'a Path,
    codec: Codec,
    settings: EncoderSettings,
    encoder_source: &'a Path,
    original: &'a OriginalImage,
    metadata: &'a MetadataPayload,
    prepared_original: Option<&'a GpuPreparedOriginal>,
    preprocess_ms: Option<u64>,
    gpu_preprocess_fallback_reason: Option<&'a str>,
    has_alpha: bool,
    candidate_dir: &'a Path,
    policy: &'a DecisionPolicy,
    cache_source_hash: Option<&'a str>,
    external_metric_original: Option<&'a Path>,
    candidate_index: usize,
    progress_index: usize,
    progress_total: usize,
    /// Candidates of the image that are evaluated together with this one.
    peers: usize,
    /// When the image runs out of time; ends the wait for candidate memory.
    deadline: Option<Instant>,
    /// Milliseconds the image has spent waiting on memory and temporary room.
    waited: &'a AtomicU64,
}

struct QualityRefinementRequest<'a> {
    input: &'a Path,
    parallelism: usize,
    image_start: Instant,
    waited: &'a AtomicU64,
    encoder_source: &'a Path,
    original: &'a OriginalImage,
    metadata: &'a MetadataPayload,
    prepared_original: Option<&'a GpuPreparedOriginal>,
    preprocess_ms: Option<u64>,
    gpu_preprocess_fallback_reason: Option<&'a str>,
    has_alpha: bool,
    candidate_dir: &'a Path,
    policy: &'a DecisionPolicy,
    cache_source_hash: Option<&'a str>,
    external_metric_original: Option<&'a Path>,
}

struct ScoreDecodedRequest<'a> {
    original: &'a OriginalImage,
    prepared_original: Option<&'a GpuPreparedOriginal>,
    preprocess_ms: Option<u64>,
    gpu_preprocess_fallback_reason: Option<&'a str>,
    decoded: &'a image::DynamicImage,
    has_alpha: bool,
    external_metric_original: Option<&'a Path>,
    candidate_dir: &'a Path,
    candidate_index: usize,
}
