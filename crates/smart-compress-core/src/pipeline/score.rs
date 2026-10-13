use std::{borrow::Cow, collections::BTreeMap, ffi::OsString, fs, path::Path, time::Instant};

use anyhow::{Context, Result, anyhow};
use fast_ssim2::Ssimulacra2Reference;
use image::GenericImageView;
use smart_compress_codecs::{Codec, Permit, run_process, run_process_threads};
use smart_compress_metrics::{QualityScore, alpha_edge_similarity};

use super::{DecodedCandidate, Optimizer, SCORING_VERSION, ScoreDecodedRequest, inprocess, work};
use crate::{
    candidate::{ActualScoreBackend, CandidateReport, GpuVerificationReport},
    candidate_probe::check_candidate,
    image_probe::{decode_image_with_limits, read_pam},
    memory,
    scoring::{CpuRayonBackend, FULL_SIZE_METRIC_PIXELS, OriginalImage},
};

fn optional_delta(left: Option<f32>, right: Option<f32>) -> f32 {
    match (left, right) {
        (Some(left), Some(right)) => (left - right).abs(),
        (None, None) => 0.0,
        _ => f32::INFINITY,
    }
}

impl Optimizer {
    pub(super) fn score_decoded_candidate(
        &self,
        request: ScoreDecodedRequest<'_>,
        candidate: &mut CandidateReport,
    ) -> Result<QualityScore> {
        let ScoreDecodedRequest {
            original,
            prepared_original,
            preprocess_ms,
            gpu_preprocess_fallback_reason,
            decoded,
            has_alpha,
            external_metric_original,
            candidate_dir,
            candidate_index,
        } = request;
        candidate.preprocess_ms = preprocess_ms;
        let score_start = Instant::now();
        let result = (|| -> Result<QualityScore> {
            let mut quality = if let Some(reason) = gpu_preprocess_fallback_reason {
                candidate.score_backend = Some(ActualScoreBackend::CpuFallback {
                    reason: reason.to_string(),
                });
                self.on_metric_pool(self.metric_threads, || {
                    CpuRayonBackend.score_original(original, decoded, has_alpha)
                })?
                .with_context(|| format!("quality metric failed after {reason}"))?
            } else {
                match (prepared_original, self.compute.as_ref().as_gpu()) {
                    (Some(prepared_original), Some(gpu)) => {
                        // The unit is back before any CPU fallback asks for its own.
                        let scored = self.on_metric_pool(1, || {
                            gpu.score_candidate_prepared_with_timings(
                                prepared_original,
                                decoded,
                                has_alpha,
                            )
                        })?;
                        match scored {
                            Ok((score, timings)) => {
                                candidate.score_backend = Some(ActualScoreBackend::Gpu);
                                candidate.gpu_upload_ms = timings.upload_ms;
                                candidate.gpu_transform_ms = timings.transform_ms;
                                candidate.gpu_compute_ms = timings.compute_ms;
                                candidate.gpu_readback_ms = timings.readback_ms;
                                candidate.gpu_timestamp_ns = timings.timestamp_ns;
                                score
                            }
                            Err(error) => {
                                if self.config.gpu.required {
                                    return Err(anyhow!(
                                        "GPU is required but prepared GPU scoring failed: {error:#}"
                                    ));
                                }
                                let reason = format!("prepared GPU scoring failed: {error:#}");
                                candidate
                                    .warnings
                                    .push(format!("CPU scoring fallback: {reason}"));
                                candidate.score_backend = Some(ActualScoreBackend::CpuFallback {
                                    reason: reason.clone(),
                                });
                                self.on_metric_pool(self.metric_threads, || {
                                    CpuRayonBackend.score_original(original, decoded, has_alpha)
                                })?
                                .with_context(|| format!("quality metric failed after {reason}"))?
                            }
                        }
                    }
                    _ => {
                        if let Some(gpu) = self.compute.as_ref().as_gpu() {
                            let outcome = self
                                .on_metric_pool(1, || {
                                    gpu.score_original_with_fallback(original, decoded, has_alpha)
                                })?
                                .context("quality metric failed")?;
                            candidate.gpu_compute_ms = outcome.timings.compute_ms;
                            candidate.gpu_upload_ms = outcome.timings.upload_ms;
                            candidate.gpu_transform_ms = outcome.timings.transform_ms;
                            candidate.gpu_readback_ms = outcome.timings.readback_ms;
                            candidate.gpu_timestamp_ns = outcome.timings.timestamp_ns;
                            if self.config.gpu.required && outcome.fallback_reason.is_some() {
                                return Err(anyhow!(
                                    "GPU is required but scoring fell back to CPU: {}",
                                    outcome
                                        .fallback_reason
                                        .as_deref()
                                        .unwrap_or("unknown reason")
                                ));
                            }
                            candidate.score_backend = Some(match outcome.fallback_reason {
                                Some(reason) => {
                                    candidate
                                        .warnings
                                        .push(format!("CPU scoring fallback: {reason}"));
                                    ActualScoreBackend::CpuFallback { reason }
                                }
                                None => ActualScoreBackend::Gpu,
                            });
                            outcome.quality
                        } else {
                            candidate.score_backend = Some(ActualScoreBackend::Cpu);
                            self.on_metric_pool(self.metric_threads, || {
                                CpuRayonBackend.score_original(original, decoded, has_alpha)
                            })?
                            .context("quality metric failed")?
                        }
                    }
                }
            };
            if has_alpha {
                // Same CPU check for every backend, so it cannot split their verdicts.
                let edge = self.on_metric_pool(self.metric_threads, || {
                    alpha_edge_similarity(original.full_rgba(), &decoded.to_rgba8())
                })?;
                quality.alpha_edge = Some(edge);
                quality.overall = quality.overall.min(edge);
            }
            if self.config.gpu.verify_gpu && self.compute.as_ref().as_gpu().is_some() {
                match self.on_metric_pool(self.metric_threads, || {
                    CpuRayonBackend.score_original(original, decoded, has_alpha)
                })? {
                    Ok(cpu_score) => {
                        const TOLERANCE: f32 = 0.25;
                        let component_deltas = BTreeMap::from([
                            ("rgb".to_string(), (cpu_score.rgb - quality.rgb).abs()),
                            ("edge".to_string(), (cpu_score.edge - quality.edge).abs()),
                            (
                                "alpha".to_string(),
                                optional_delta(cpu_score.alpha, quality.alpha),
                            ),
                            (
                                "composite".to_string(),
                                optional_delta(cpu_score.composite_min, quality.composite_min),
                            ),
                        ]);
                        let max_delta = component_deltas.values().copied().fold(0.0_f32, f32::max);
                        let passed = max_delta <= TOLERANCE;
                        candidate.gpu_verification = Some(GpuVerificationReport {
                            tolerance: TOLERANCE,
                            cpu_overall: cpu_score.overall,
                            gpu_overall: quality.overall,
                            overall_delta: (cpu_score.overall - quality.overall).abs(),
                            component_deltas,
                            passed,
                        });
                        candidate.warnings.push(format!(
                            "verify-gpu passed={passed} max_component_delta={max_delta:.3}"
                        ));
                    }
                    Err(error) => candidate
                        .warnings
                        .push(format!("verify-gpu CPU reference failed: {error}")),
                }
            }
            if self.ssimulacra2_in_process
                && self.scores_ssimulacra2(decoded.width(), decoded.height())
            {
                let reference = original
                    .ssimulacra2_reference()
                    .context("the SSIMULACRA2 reference was not prepared")?;
                let score = self.score_ssimulacra2_in_process(reference, decoded)?;
                candidate.score_backend = Some(ActualScoreBackend::Ssimulacra2Inprocess);
                with_ssimulacra2(&mut quality, score);
            }
            if let Some(original_path) = external_metric_original {
                let external_metric_start = Instant::now();
                // SSIMULACRA2 needs at least an 8x8 image; smaller ones keep
                // the built-in score.
                let run_ssimulacra2 = !self.ssimulacra2_in_process
                    && self.scores_ssimulacra2(decoded.width(), decoded.height());
                // The PNG is deleted once scored, whatever the outcome.
                let candidate_png =
                    candidate_dir.join(format!("metric-candidate-{candidate_index}.png"));
                let _scratch = RemoveOnDrop(&candidate_png);
                if run_ssimulacra2 {
                    write_metric_png(decoded, &candidate_png)?;
                    let score = self.score_ssimulacra2(
                        original_path,
                        &candidate_png,
                        u64::from(decoded.width()) * u64::from(decoded.height()),
                    )?;
                    with_ssimulacra2(&mut quality, score);
                }
                candidate.external_metric_ms =
                    Some(external_metric_start.elapsed().as_millis() as u64);
            }
            Ok(quality)
        })();
        candidate.score_ms = Some(score_start.elapsed().as_millis() as u64);
        if matches!(
            candidate.score_backend,
            Some(ActualScoreBackend::Gpu | ActualScoreBackend::CpuFallback { .. })
        ) {
            candidate.bytes.source_upload_bytes = u64::from(original.width())
                .checked_mul(u64::from(original.height()))
                .and_then(|pixels| pixels.checked_mul(4));
            candidate.bytes.candidate_upload_bytes = u64::from(decoded.width())
                .checked_mul(u64::from(decoded.height()))
                .and_then(|pixels| pixels.checked_mul(4));
        }
        result
    }

    /// Runs a leaf of metric work, which may use Rayon, on the metric threads
    /// while holding `units` of the work budget.
    ///
    /// The leaf is handed to a thread of its own and the caller joins it. A
    /// candidate runs on a worker of the candidate pool, and a Rayon worker that
    /// waits inside `install` or a parallel iterator runs other jobs of its pool
    /// meanwhile: other candidates, which would wait for the units held here by
    /// the very thread they are stacked on.
    pub(super) fn on_metric_pool<R: Send>(
        &self,
        units: usize,
        leaf: impl FnOnce() -> R + Send,
    ) -> Result<R> {
        let _units = work(units)?;
        Ok(std::thread::scope(|scope| {
            scope
                .spawn(|| self.metric_pool.install(leaf))
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        }))
    }

    pub(super) fn scoring_signature(&self) -> String {
        let ssimulacra2 = if self.ssimulacra2_in_process {
            inprocess::SIGNATURE.to_string()
        } else {
            self.use_ssimulacra2.to_string()
        };
        format!(
            "{SCORING_VERSION};backend={};ssimulacra2={};gpu-preprocess={}",
            self.compute.backend_id(),
            ssimulacra2,
            self.config.gpu.prefer_gpu_preprocess
        )
    }

    pub(super) fn decode_candidate(
        &self,
        codec: Codec,
        path: &Path,
        original: &image::DynamicImage,
        has_alpha: bool,
        threads: usize,
    ) -> Result<DecodedCandidate> {
        // The header decides first: a candidate that cannot be the image that
        // was encoded never reaches an external decoder.
        check_candidate(codec, path, original, has_alpha)?;
        match codec {
            Codec::Avif => {
                let decoded = path.with_extension("decoded.png");
                let tool = self
                    .backend
                    .tools()
                    .avifdec
                    .path
                    .as_ref()
                    .ok_or_else(|| anyhow!("avifdec is not available"))?;
                let process = run_process_threads(
                    tool,
                    &[
                        // The PNG is a lossless temporary read straight back;
                        // storing it uncompressed gives the same pixels and
                        // skips the compression and inflation that made up
                        // most of the decode time.
                        OsString::from("--png-compress"),
                        OsString::from("0"),
                        path.as_os_str().to_os_string(),
                        decoded.as_os_str().to_os_string(),
                    ],
                    self.config.candidate_timeout(),
                    1,
                )?;
                let io_start = Instant::now();
                let _unit = work(1)?;
                let image = decode_image_with_limits(&decoded, self.pipeline_decode_limits());
                // Uncompressed, the file is as large as the pixels. Once they are
                // in memory it must not count against temporary storage for the
                // rest of the search.
                let _ = fs::remove_file(&decoded);
                let image = image.context("opening AVIF decode PNG")?;
                Ok(DecodedCandidate {
                    image,
                    process_ms: Some(process.elapsed_ms),
                    io_ms: io_start.elapsed().as_millis() as u64,
                })
            }
            Codec::Jxl => {
                // PAM is uncompressed and carries alpha and 16-bit samples.
                let decoded = path.with_extension("decoded.pam");
                let tool = self
                    .backend
                    .tools()
                    .djxl
                    .path
                    .as_ref()
                    .ok_or_else(|| anyhow!("djxl not found"))?;
                let process = run_process_threads(
                    tool,
                    &[
                        OsString::from(format!("--num_threads={threads}")),
                        path.as_os_str().to_os_string(),
                        decoded.as_os_str().to_os_string(),
                    ],
                    self.config.candidate_timeout(),
                    threads,
                )?;
                let io_start = Instant::now();
                let _unit = work(1)?;
                let image = read_pam(&decoded, self.pipeline_decode_limits())
                    .context("opening JXL decode output")?;
                // The pixels are in memory; do not let the file count
                // against temporary storage for the rest of the search.
                let _ = fs::remove_file(&decoded);
                Ok(DecodedCandidate {
                    image,
                    process_ms: Some(process.elapsed_ms),
                    io_ms: io_start.elapsed().as_millis() as u64,
                })
            }
            _ => {
                let io_start = Instant::now();
                let _unit = work(1)?;
                let image = decode_image_with_limits(path, self.pipeline_decode_limits())
                    .with_context(|| format!("opening candidate {}", path.display()))?;
                Ok(DecodedCandidate {
                    image,
                    process_ms: None,
                    io_ms: io_start.elapsed().as_millis() as u64,
                })
            }
        }
    }

    /// Whether ssimulacra2 scores an image of this size: it needs 8 px a side.
    pub(super) fn scores_ssimulacra2(&self, width: u32, height: u32) -> bool {
        self.use_ssimulacra2 && width >= 8 && height >= 8
    }

    fn score_ssimulacra2(
        &self,
        original_path: &Path,
        candidate_path: &Path,
        pixels: u64,
    ) -> Result<f32> {
        let tool = self
            .backend
            .tools()
            .ssimulacra2
            .path
            .as_ref()
            .ok_or_else(|| anyhow!("ssimulacra2 not found"))?;
        let timeout = self.config.candidate_timeout();
        let _reserved = self.metric_memory.acquire(
            memory::external_metric_bytes(pixels.min(FULL_SIZE_METRIC_PIXELS)),
            Some(Instant::now() + timeout),
        )?;
        run_score_tool(tool, original_path, candidate_path, timeout, "SSIMULACRA2")
    }

    fn score_ssimulacra2_in_process(
        &self,
        reference: &Ssimulacra2Reference,
        decoded: &image::DynamicImage,
    ) -> Result<f32> {
        let _reserved = self.reserve_in_process_metric(decoded)?;
        self.on_metric_pool(self.metric_threads, || inprocess::score(reference, decoded))?
    }

    /// Memory for scoring `image` in this process, from the share the external
    /// metric draws on.
    fn reserve_in_process_metric(&self, image: &image::DynamicImage) -> Result<Permit<'_>> {
        let pixels = u64::from(image.width()) * u64::from(image.height());
        Ok(self.metric_memory.acquire(
            memory::in_process_metric_bytes(pixels.min(FULL_SIZE_METRIC_PIXELS)),
            Some(Instant::now() + self.config.candidate_timeout()),
        )?)
    }

    /// Builds the in-process reference of an image and keeps it with the image.
    pub(super) fn prepare_ssimulacra2_reference(&self, original: &OriginalImage) -> Result<()> {
        let _reserved = self.reserve_in_process_metric(original)?;
        let reference =
            self.on_metric_pool(self.metric_threads, || inprocess::build_reference(original))??;
        original.set_ssimulacra2_reference(reference);
        Ok(())
    }
}

/// Runs an external metric tool on two files and reads the first number it prints.
fn run_score_tool(
    tool: &Path,
    original_path: &Path,
    candidate_path: &Path,
    timeout: std::time::Duration,
    name: &str,
) -> Result<f32> {
    let output = run_process(
        tool,
        &[
            original_path.as_os_str().to_os_string(),
            candidate_path.as_os_str().to_os_string(),
        ],
        timeout,
    )?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split_whitespace()
        .next()
        .ok_or_else(|| anyhow!("{name} produced no score"))?
        .parse::<f32>()
        .with_context(|| format!("parsing {name} score"))
}

/// The perceptual score replaces the PSNR-derived one; the alpha-channel checks
/// still guard transparency.
fn with_ssimulacra2(quality: &mut QualityScore, score: f32) {
    quality.ssimulacra2 = Some(score);
    quality.overall = score
        .min(quality.alpha.unwrap_or(100.0))
        .min(quality.alpha_edge.unwrap_or(100.0));
}

/// Deletes a scratch file when dropped.
pub(super) struct RemoveOnDrop<'a>(pub(super) &'a Path);

impl Drop for RemoveOnDrop<'_> {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.0);
    }
}

pub(super) fn write_metric_png(image: &image::DynamicImage, path: &Path) -> Result<()> {
    let _unit = work(1)?;
    let metric = bounded_metric_image(image);
    metric
        .save_with_format(path, image::ImageFormat::Png)
        .with_context(|| format!("writing metric PNG {}", path.display()))
}

/// Image handed to the external metrics. Scoring a reduced copy of a large
/// image hides compression damage that is plain at full size, so large images
/// are scored on a mosaic of full-resolution samples from a regular grid.
// A fixed 3x3 grid of 1024 px tiles. Sampling by detail, or more tiles, would
// help if large images with small busy regions are scored too kindly.
pub(super) fn bounded_metric_image(image: &image::DynamicImage) -> Cow<'_, image::DynamicImage> {
    const GRID: u32 = 3;
    const TILE: u32 = 1024;
    let (width, height) = image.dimensions();
    if width as u64 * height as u64 <= FULL_SIZE_METRIC_PIXELS {
        return Cow::Borrowed(image);
    }
    let (cell_width, cell_height) = (width / GRID, height / GRID);
    let (tile_width, tile_height) = (TILE.min(cell_width), TILE.min(cell_height));
    let mut mosaic = image::RgbaImage::new(tile_width * GRID, tile_height * GRID);
    for row in 0..GRID {
        for column in 0..GRID {
            // The tile sits in the middle of its grid cell.
            let tile = image
                .crop_imm(
                    cell_width * column + (cell_width - tile_width) / 2,
                    cell_height * row + (cell_height - tile_height) / 2,
                    tile_width,
                    tile_height,
                )
                .to_rgba8();
            image::imageops::replace(
                &mut mosaic,
                &tile,
                i64::from(tile_width * column),
                i64::from(tile_height * row),
            );
        }
    }
    Cow::Owned(if image.color().has_alpha() {
        image::DynamicImage::ImageRgba8(mosaic)
    } else {
        image::DynamicImage::ImageRgb8(image::DynamicImage::ImageRgba8(mosaic).to_rgb8())
    })
}
