use std::{
    borrow::Cow,
    ops::Deref,
    sync::{OnceLock, mpsc},
    time::Instant,
};

use image::{DynamicImage, GenericImageView, RgbaImage, metadata::Orientation};
use smart_compress_metrics::{
    MetricError, QualityScore, alpha_similarity, composite_score, edge_similarity, rgb_similarity,
    score_rgba,
};
use wgpu::util::DeviceExt;

use crate::gpu::{GpuAdapterReport, adapter_report, select_adapters};

pub trait ComputeBackend {
    fn backend_id(&self) -> &'static str;

    fn score_candidate(
        &self,
        original: &DynamicImage,
        candidate: &DynamicImage,
        has_alpha: bool,
    ) -> Result<QualityScore, MetricError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuRayonBackend;

impl ComputeBackend for CpuRayonBackend {
    fn backend_id(&self) -> &'static str {
        "cpu-rayon"
    }

    fn score_candidate(
        &self,
        original: &DynamicImage,
        candidate: &DynamicImage,
        has_alpha: bool,
    ) -> Result<QualityScore, MetricError> {
        let dimensions = original.dimensions();
        score_rgba(
            &metric_level(original, dimensions),
            &metric_level(candidate, dimensions),
            has_alpha,
        )
    }
}

impl CpuRayonBackend {
    /// Same score as `score_candidate`, against an original whose metric
    /// buffer is converted once and shared by every candidate.
    pub fn score_original(
        &self,
        original: &OriginalImage,
        candidate: &DynamicImage,
        has_alpha: bool,
    ) -> Result<QualityScore, MetricError> {
        score_rgba(
            original.metric_rgba(),
            &metric_level(candidate, original.dimensions()),
            has_alpha,
        )
    }
}

/// A decoded original plus its RGBA forms for the built-in metrics. Each is
/// converted on first use and shared by every candidate scored against it,
/// instead of being rebuilt, at full size, for each of them.
pub struct OriginalImage {
    image: DynamicImage,
    full: OnceLock<RgbaImage>,
    metric: OnceLock<RgbaImage>,
}

impl From<DynamicImage> for OriginalImage {
    fn from(image: DynamicImage) -> Self {
        Self {
            image,
            full: OnceLock::new(),
            metric: OnceLock::new(),
        }
    }
}

impl Deref for OriginalImage {
    type Target = DynamicImage;

    fn deref(&self) -> &DynamicImage {
        &self.image
    }
}

impl OriginalImage {
    pub fn full_rgba(&self) -> &RgbaImage {
        self.full.get_or_init(|| self.image.to_rgba8())
    }

    /// The buffer candidates are compared against: the full one, or its
    /// reduction when the image is too large to score at full size.
    pub fn metric_rgba(&self) -> &RgbaImage {
        let dimensions = self.image.dimensions();
        if bounded_metric_dimensions(dimensions.0, dimensions.1) == dimensions {
            self.full_rgba()
        } else {
            self.metric
                .get_or_init(|| metric_level(&self.image, dimensions))
        }
    }
}

pub struct GpuWgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    metric_pipeline: wgpu::ComputePipeline,
    metric_bind_group_layout: wgpu::BindGroupLayout,
    transform_pipeline: wgpu::ComputePipeline,
    transform_bind_group_layout: wgpu::BindGroupLayout,
    max_dispatch_bytes: Option<u64>,
    adapter_report: GpuAdapterReport,
}

pub struct GpuPreparedOriginal {
    oriented_width: u32,
    oriented_height: u32,
    levels: Vec<GpuResidentLevel>,
    overview: RgbaImage,
    preprocess_mode: String,
    /// Largest source band uploaded at once when preprocessing was tiled.
    band_pixels: Option<u64>,
}

pub struct GpuScoreOutcome {
    pub quality: QualityScore,
    pub fallback_reason: Option<String>,
    pub timings: GpuTimingBreakdown,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GpuTimingBreakdown {
    pub upload_ms: Option<u64>,
    pub transform_ms: Option<u64>,
    pub compute_ms: Option<u64>,
    pub readback_ms: Option<u64>,
    /// GPU-side duration of the compute passes from timestamp queries, in
    /// nanoseconds. `None` when the adapter has no timestamp queries.
    pub timestamp_ns: Option<u64>,
}

impl GpuTimingBreakdown {
    fn add(&mut self, other: Self) {
        self.timestamp_ns = add_optional_timing(self.timestamp_ns, other.timestamp_ns);
        self.upload_ms = add_optional_timing(self.upload_ms, other.upload_ms);
        self.transform_ms = add_optional_timing(self.transform_ms, other.transform_ms);
        self.compute_ms = add_optional_timing(self.compute_ms, other.compute_ms);
        self.readback_ms = add_optional_timing(self.readback_ms, other.readback_ms);
    }
}

fn add_optional_timing(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

impl GpuPreparedOriginal {
    pub fn overview(&self) -> &RgbaImage {
        &self.overview
    }

    pub fn preprocess_mode(&self) -> &str {
        &self.preprocess_mode
    }

    pub fn band_pixels(&self) -> Option<u64> {
        self.band_pixels
    }

    fn metric_level(&self) -> &GpuResidentLevel {
        &self.levels[0]
    }
}

struct GpuResidentLevel {
    width: u32,
    height: u32,
    buffer: wgpu::Buffer,
}

/// Start and end timestamps of one compute pass.
struct GpuPassTimer {
    query_set: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl GpuPassTimer {
    fn writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    /// Record after the timed pass, in the same encoder.
    fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolved, 0);
    }
}

const GPU_TIMESTAMP_BYTES: u64 = 16;

/// Setting this variable turns every GPU test skip into a failure.
const REQUIRE_GPU_ENV: &str = "SMART_COMPRESS_REQUIRE_GPU";

/// Test support shared by the unit and integration tests: unwraps the backend
/// a GPU test constructed. Without a GPU the test skips with one line on
/// stderr, unless `SMART_COMPRESS_REQUIRE_GPU` is set, which fails it.
#[doc(hidden)]
pub fn gpu_or_skip(backend: anyhow::Result<GpuWgpuBackend>) -> Option<GpuWgpuBackend> {
    gpu_or_skip_with(backend, std::env::var_os(REQUIRE_GPU_ENV).is_some())
}

fn gpu_or_skip_with(
    backend: anyhow::Result<GpuWgpuBackend>,
    required: bool,
) -> Option<GpuWgpuBackend> {
    let error = match backend {
        // A software adapter is not the GPU these tests are about.
        Ok(gpu) if gpu.adapter_report().device_type != crate::gpu::SOFTWARE_DEVICE_TYPE => {
            return Some(gpu);
        }
        Ok(_) => anyhow::anyhow!("only a software adapter is available"),
        Err(error) => error,
    };
    let test = std::thread::current().name().unwrap_or("test").to_owned();
    assert!(
        !required,
        "{test}: no GPU backend and {REQUIRE_GPU_ENV} is set: {error:#}"
    );
    eprintln!("{test}: skipped, no GPU backend ({error:#}); set {REQUIRE_GPU_ENV} to fail instead");
    None
}

impl GpuWgpuBackend {
    pub fn new() -> anyhow::Result<Self> {
        Self::new_with_max_vram(None)
    }

    pub fn new_with_max_vram(max_vram_bytes: Option<u64>) -> anyhow::Result<Self> {
        Self::new_with_selection(max_vram_bytes, None, None, None)
    }

    pub fn new_with_selection(
        max_vram_bytes: Option<u64>,
        adapter_name: Option<&str>,
        adapter_vendor: Option<u32>,
        adapter_device: Option<u32>,
    ) -> anyhow::Result<Self> {
        let (adapter, _) = select_adapters(adapter_name, adapter_vendor, adapter_device)
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no wgpu adapter available"))?;
        Self::from_adapter(adapter, max_vram_bytes)
    }

    /// Creates the compute device from exactly this adapter.
    pub fn from_adapter(
        adapter: wgpu::Adapter,
        max_vram_bytes: Option<u64>,
    ) -> anyhow::Result<Self> {
        let adapter_report = adapter_report(&adapter);
        let adapter_limits = adapter.limits();
        let max_dispatch_bytes = [
            max_vram_bytes,
            Some(adapter_limits.max_buffer_size),
            Some(adapter_limits.max_storage_buffer_binding_size),
        ]
        .into_iter()
        .flatten()
        .min();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("smart-compress-gpu-metrics"),
                required_features: if adapter_report.timestamp_queries {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
                required_limits: adapter_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))?;

        let metric_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smart-compress-metric-sums"),
            source: wgpu::ShaderSource::Wgsl(METRIC_SUMS_SHADER.into()),
        });
        let metric_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("smart-compress-metric-sums-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("smart-compress-metric-sums-pipeline-layout"),
            bind_group_layouts: &[Some(&metric_bind_group_layout)],
            immediate_size: 0,
        });
        let metric_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("smart-compress-metric-sums-pipeline"),
            layout: Some(&pipeline_layout),
            module: &metric_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let transform_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smart-compress-transform-downsample"),
            source: wgpu::ShaderSource::Wgsl(TRANSFORM_DOWNSAMPLE_SHADER.into()),
        });
        let transform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("smart-compress-transform-downsample-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let transform_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("smart-compress-transform-downsample-pipeline-layout"),
                bind_group_layouts: &[Some(&transform_bind_group_layout)],
                immediate_size: 0,
            });
        let transform_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("smart-compress-transform-downsample-pipeline"),
            layout: Some(&transform_pipeline_layout),
            module: &transform_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            metric_pipeline,
            metric_bind_group_layout,
            transform_pipeline,
            transform_bind_group_layout,
            max_dispatch_bytes,
            adapter_report,
        })
    }

    pub fn adapter_report(&self) -> &GpuAdapterReport {
        &self.adapter_report
    }

    pub fn prepare_original(
        &self,
        original: &DynamicImage,
        orientation: Orientation,
    ) -> anyhow::Result<GpuPreparedOriginal> {
        let source = original.to_rgba8();
        let source_bytes = source.as_raw().len() as u64;
        let source_width = source.width();
        let source_height = source.height();
        let (oriented_width, oriented_height) =
            oriented_dimensions(source_width, source_height, orientation);
        let (metric_width, metric_height) =
            bounded_metric_dimensions(oriented_width, oriented_height);
        let (overview_width, overview_height) =
            bounded_preview_dimensions(oriented_width, oriented_height, 512);
        if let Some(limit) = self.max_dispatch_bytes {
            let metric_bytes = metric_width as u64 * metric_height as u64 * 4;
            let overview_bytes = overview_width as u64 * overview_height as u64 * 4;
            let untiled_peak = source_bytes
                .saturating_add(metric_bytes.max(overview_bytes.saturating_mul(2)))
                .saturating_add(TRANSFORM_PARAMS_BYTES);
            if untiled_peak > limit {
                let tiled = self
                    .transform_tiled(&source, orientation, metric_width, metric_height, limit)
                    .and_then(|(metric, metric_band)| {
                        anyhow::ensure!(
                            overview_bytes.saturating_mul(2) <= limit,
                            "GPU overview readback exceeds limit {limit}"
                        );
                        let (overview, overview_band) = self.transform_tiled(
                            &source,
                            orientation,
                            overview_width,
                            overview_height,
                            limit,
                        )?;
                        let overview =
                            self.readback_rgba(&overview, overview_width, overview_height)?;
                        Ok((metric, overview, metric_band.max(overview_band)))
                    });
                return match tiled {
                    Ok((metric, overview, band_pixels)) => Ok(GpuPreparedOriginal {
                        band_pixels: Some(band_pixels),
                        oriented_width,
                        oriented_height,
                        levels: vec![GpuResidentLevel {
                            width: metric_width,
                            height: metric_height,
                            buffer: metric,
                        }],
                        overview,
                        preprocess_mode: "gpu-transform-tiled".to_string(),
                    }),
                    // The limit cannot hold even one band next to the output
                    // level; bound the image on the CPU instead.
                    Err(_) => self.prepare_original_from_bounded_cpu(
                        original,
                        orientation,
                        (metric_width, metric_height),
                        (overview_width, overview_height),
                        limit,
                    ),
                };
            }
        }

        let source_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("smart-compress-preprocess-source"),
                contents: source.as_raw(),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let (metric_level, _, _) = self.prepare_level(
            &source_buffer,
            source_width,
            source_height,
            orientation,
            metric_width,
            metric_height,
            "metric",
            false,
        )?;
        let (_overview_level, overview, _) = self.prepare_level(
            &source_buffer,
            source_width,
            source_height,
            orientation,
            overview_width,
            overview_height,
            "overview",
            true,
        )?;

        Ok(GpuPreparedOriginal {
            oriented_width,
            oriented_height,
            levels: vec![metric_level],
            overview: overview.expect("overview readback requested"),
            preprocess_mode: "gpu-transform".to_string(),
            band_pixels: None,
        })
    }

    /// Orients and downsamples `source` into a resident buffer, uploading
    /// only the source band each group of output rows reads. The box filter
    /// reads exactly its own footprint, so bands reproduce the untiled result.
    // Each level re-uploads its bands. One pass shared across levels would save
    // that if upload time ever shows up in the preprocess timings.
    fn transform_tiled(
        &self,
        source: &RgbaImage,
        orientation: Orientation,
        output_width: u32,
        output_height: u32,
        limit: u64,
    ) -> anyhow::Result<(wgpu::Buffer, u64)> {
        let (source_width, source_height) = source.dimensions();
        let (oriented_width, oriented_height) =
            oriented_dimensions(source_width, source_height, orientation);
        let pixel_count = output_width as u64 * output_height as u64;
        anyhow::ensure!(pixel_count > 0, "cannot tile an empty GPU level");
        let band_budget = limit.saturating_sub(pixel_count * 4 + TRANSFORM_PARAMS_BYTES);
        let max_band_rows = band_budget / (oriented_width as u64 * 4);
        let footprint_start =
            |row: u32| (row as u64 * oriented_height as u64 / output_height as u64) as u32;
        let footprint_end = |row: u32| {
            ((row as u64 + 1) * oriented_height as u64).div_ceil(output_height as u64) as u32
        };
        let orientation_code = orientation.to_exif() as u32;
        let output_buffer = self.create_transform_output(pixel_count);
        let mut out_row = 0;
        let mut max_band_pixels = 0;
        while out_row < output_height {
            let band_start = footprint_start(out_row);
            let mut rows = 0;
            while out_row + rows < output_height
                && u64::from(
                    footprint_end(out_row + rows).max(footprint_start(out_row + rows) + 1)
                        - band_start,
                ) <= max_band_rows
            {
                rows += 1;
            }
            anyhow::ensure!(
                rows > 0,
                "GPU limit {limit} bytes is too small for one preprocess band"
            );
            let band_end = footprint_end(out_row + rows - 1)
                .max(footprint_start(out_row + rows - 1) + 1)
                .min(oriented_height);
            let band = band_end - band_start;
            // Oriented rows map to source rows for upright orientations and to
            // source columns for transposed ones.
            let (tile_x, tile_y, tile_width, tile_height) = match orientation_code {
                1 | 2 => (0, band_start, source_width, band),
                3 | 4 => (0, source_height - band_end, source_width, band),
                5 | 6 => (band_start, 0, band, source_height),
                _ => (source_width - band_end, 0, band, source_height),
            };
            max_band_pixels = max_band_pixels.max(u64::from(tile_width) * u64::from(tile_height));
            let stride = source_width as usize * 4;
            let raw = source.as_raw();
            let tile: std::borrow::Cow<'_, [u8]> = if tile_width == source_width {
                Cow::Borrowed(
                    &raw[tile_y as usize * stride..(tile_y + tile_height) as usize * stride],
                )
            } else {
                let mut columns =
                    Vec::with_capacity(tile_width as usize * tile_height as usize * 4);
                for row in raw.chunks_exact(stride) {
                    columns.extend_from_slice(
                        &row[tile_x as usize * 4..(tile_x + tile_width) as usize * 4],
                    );
                }
                Cow::Owned(columns)
            };
            let tile_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("smart-compress-preprocess-band"),
                    contents: &tile,
                    usage: wgpu::BufferUsages::STORAGE,
                });
            self.dispatch_transform(
                &tile_buffer,
                &output_buffer,
                &TransformParams {
                    source_width,
                    source_height,
                    output_width,
                    output_height,
                    orientation_code,
                    tile_x,
                    tile_y,
                    tile_width,
                    out_row_offset: out_row,
                    out_row_count: rows,
                },
            );
            // Finish this band before uploading the next so bands never coexist.
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(anyhow::Error::from)?;
            out_row += rows;
        }
        Ok((output_buffer, max_band_pixels))
    }

    fn prepare_original_from_bounded_cpu(
        &self,
        original: &DynamicImage,
        orientation: Orientation,
        metric_dimensions: (u32, u32),
        overview_dimensions: (u32, u32),
        limit: u64,
    ) -> anyhow::Result<GpuPreparedOriginal> {
        let mut oriented = original.clone();
        oriented.apply_orientation(orientation);
        let oriented_rgba = oriented.to_rgba8();
        let metric = box_downsample(&oriented_rgba, metric_dimensions.0, metric_dimensions.1);
        let overview = box_downsample(&oriented_rgba, overview_dimensions.0, overview_dimensions.1);
        let metric_bytes = metric.as_raw().len() as u64;
        anyhow::ensure!(
            metric_bytes <= limit,
            "bounded GPU metric level needs {metric_bytes} bytes, exceeding GPU limit {limit}"
        );
        anyhow::ensure!(
            overview.as_raw().len() as u64 <= limit,
            "bounded GPU overview needs {} bytes, exceeding GPU limit {limit}",
            overview.as_raw().len()
        );
        let metric_buffer = self
            .upload_resident_level(&metric, "smart-compress-tiled-preprocess-metric")?
            .buffer;
        Ok(GpuPreparedOriginal {
            oriented_width: oriented.width(),
            oriented_height: oriented.height(),
            levels: vec![GpuResidentLevel {
                width: metric.width(),
                height: metric.height(),
                buffer: metric_buffer,
            }],
            overview,
            preprocess_mode: "cpu-bounded-upload".to_string(),
            band_pixels: None,
        })
    }

    fn upload_resident_level(
        &self,
        image: &RgbaImage,
        label: &str,
    ) -> anyhow::Result<GpuResidentLevel> {
        anyhow::ensure!(
            !image.as_raw().is_empty(),
            "cannot upload an empty GPU level"
        );
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: image.as_raw(),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Ok(GpuResidentLevel {
            width: image.width(),
            height: image.height(),
            buffer,
        })
    }

    pub fn score_candidate_prepared_with_timings(
        &self,
        original: &GpuPreparedOriginal,
        candidate: &DynamicImage,
        has_alpha: bool,
    ) -> anyhow::Result<(QualityScore, GpuTimingBreakdown)> {
        if (original.oriented_width, original.oriented_height) != candidate.dimensions() {
            return Err(MetricError::DimensionMismatch {
                original: (original.oriented_width, original.oriented_height),
                candidate: candidate.dimensions(),
            }
            .into());
        }
        let candidate_rgba = candidate.to_rgba8();
        let (metric_width, metric_height) =
            bounded_metric_dimensions(candidate_rgba.width(), candidate_rgba.height());
        let transform_start = Instant::now();
        let (candidate_level, transform_timer) = if self
            .max_dispatch_bytes
            .is_some_and(|limit| candidate_rgba.as_raw().len() as u64 > limit)
        {
            let bounded = box_downsample(&candidate_rgba, metric_width, metric_height);
            (
                self.upload_resident_level(&bounded, "smart-compress-tiled-candidate-metric")?,
                None,
            )
        } else {
            let candidate_source =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("smart-compress-candidate-preprocess-source"),
                        contents: candidate_rgba.as_raw(),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
            let (level, _, timer) = self.prepare_level(
                &candidate_source,
                candidate_rgba.width(),
                candidate_rgba.height(),
                Orientation::NoTransforms,
                metric_width,
                metric_height,
                "candidate-metric",
                false,
            )?;
            (level, timer)
        };
        let (gpu_scores, mut timings) =
            self.metric_scores_from_levels(original.metric_level(), &candidate_level)?;
        timings.transform_ms = Some(transform_start.elapsed().as_millis() as u64);
        // The metric readback above already waited for the transform pass.
        if let Some(timer) = &transform_timer {
            timings.timestamp_ns =
                add_optional_timing(timings.timestamp_ns, self.pass_nanos(timer).ok());
        }
        let alpha = has_alpha.then_some(gpu_scores.alpha);
        let composite_min = has_alpha.then_some(gpu_scores.composite_min);
        let overall = [
            Some(gpu_scores.rgb),
            Some(gpu_scores.edge),
            alpha,
            composite_min,
        ]
        .into_iter()
        .flatten()
        .fold(100.0_f32, f32::min);
        Ok((
            QualityScore {
                rgb: gpu_scores.rgb,
                ssimulacra2: None,
                butteraugli: None,
                alpha,
                composite_min,
                alpha_edge: None,
                edge: gpu_scores.edge,
                overall,
            },
            timings,
        ))
    }

    pub fn score_candidate_with_fallback(
        &self,
        original: &DynamicImage,
        candidate: &DynamicImage,
        has_alpha: bool,
    ) -> Result<GpuScoreOutcome, MetricError> {
        check_dimensions(original, candidate)?;
        let dimensions = original.dimensions();
        Ok(self.score_rgba_with_fallback(
            &metric_level(original, dimensions),
            &metric_level(candidate, dimensions),
            has_alpha,
        ))
    }

    /// Same score as `score_candidate_with_fallback`, against an original whose
    /// metric buffer is converted once and shared by every candidate.
    pub fn score_original_with_fallback(
        &self,
        original: &OriginalImage,
        candidate: &DynamicImage,
        has_alpha: bool,
    ) -> Result<GpuScoreOutcome, MetricError> {
        check_dimensions(original, candidate)?;
        Ok(self.score_rgba_with_fallback(
            original.metric_rgba(),
            &metric_level(candidate, original.dimensions()),
            has_alpha,
        ))
    }

    fn score_rgba_with_fallback(
        &self,
        original_rgba: &RgbaImage,
        candidate_rgba: &RgbaImage,
        has_alpha: bool,
    ) -> GpuScoreOutcome {
        let (gpu_scores, fallback_reason, timings) =
            match self.metric_scores(original_rgba, candidate_rgba) {
                Ok((scores, timings)) => (Some(scores), None, timings),
                Err(error) => (
                    None,
                    Some(format!("GPU metric scoring failed: {error:#}")),
                    GpuTimingBreakdown::default(),
                ),
            };

        GpuScoreOutcome {
            quality: quality_from_metric_scores(
                original_rgba,
                candidate_rgba,
                has_alpha,
                gpu_scores.as_ref(),
            ),
            fallback_reason,
            timings,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_level(
        &self,
        source_buffer: &wgpu::Buffer,
        source_width: u32,
        source_height: u32,
        orientation: Orientation,
        output_width: u32,
        output_height: u32,
        label: &str,
        readback: bool,
    ) -> anyhow::Result<(GpuResidentLevel, Option<RgbaImage>, Option<GpuPassTimer>)> {
        if let Some(limit) = self.max_dispatch_bytes {
            let source_bytes = source_width as u64 * source_height as u64 * 4;
            let output_bytes = output_width as u64 * output_height as u64 * 4;
            let peak_bytes = source_bytes
                .saturating_add(output_bytes)
                .saturating_add(if readback { output_bytes } else { 0 })
                .saturating_add(32);
            if peak_bytes > limit {
                anyhow::bail!(
                    "GPU preprocess for {label} needs {peak_bytes} bytes, exceeding limit {limit}"
                );
            }
        }

        let (output_buffer, timer) = self.transform_to_buffer(
            source_buffer,
            source_width,
            source_height,
            orientation,
            output_width,
            output_height,
            label,
        )?;
        let readback_image = if readback {
            Some(self.readback_rgba(&output_buffer, output_width, output_height)?)
        } else {
            None
        };

        Ok((
            GpuResidentLevel {
                width: output_width,
                height: output_height,
                buffer: output_buffer,
            },
            readback_image,
            timer,
        ))
    }

    fn metric_scores_from_levels(
        &self,
        original: &GpuResidentLevel,
        candidate: &GpuResidentLevel,
    ) -> anyhow::Result<(GpuMetricScores, GpuTimingBreakdown)> {
        anyhow::ensure!(
            (original.width, original.height) == (candidate.width, candidate.height),
            "metric level dimensions differ: original=({},{}) candidate=({},{})",
            original.width,
            original.height,
            candidate.width,
            candidate.height
        );

        if let Some(score_rows_per_tile) =
            self.score_rows_per_tile(original.width, original.height)?
        {
            anyhow::bail!(
                "prepared GPU metric buffers require CPU fallback when metric tiling is needed (rows per tile {score_rows_per_tile})"
            );
        }

        let sums = self.metric_sums_from_buffers(
            &original.buffer,
            &candidate.buffer,
            GpuDispatchParams {
                buffer_height: original.height,
                global_y_offset: 0,
                global_height: original.height,
                score_start_local_y: 0,
                score_row_count: original.height,
                width: original.width,
            },
        )?;
        Ok((
            sums.to_scores(original.width, original.height),
            sums.timings,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_to_buffer(
        &self,
        source_buffer: &wgpu::Buffer,
        source_width: u32,
        source_height: u32,
        orientation: Orientation,
        output_width: u32,
        output_height: u32,
        label: &str,
    ) -> anyhow::Result<(wgpu::Buffer, Option<GpuPassTimer>)> {
        let pixel_count = output_width as u64 * output_height as u64;
        if pixel_count == 0 {
            return Ok((
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("smart-compress-empty-transform"),
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                None,
            ));
        }

        let output_buffer = self.create_transform_output(pixel_count);
        let timer = self.dispatch_transform(
            source_buffer,
            &output_buffer,
            &TransformParams {
                source_width,
                source_height,
                output_width,
                output_height,
                orientation_code: orientation.to_exif() as u32,
                tile_x: 0,
                tile_y: 0,
                tile_width: source_width,
                out_row_offset: 0,
                out_row_count: output_height,
            },
        );
        let _ = label;
        Ok((output_buffer, timer))
    }

    fn create_transform_output(&self, pixel_count: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smart-compress-transform-output"),
            size: pixel_count * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn dispatch_transform(
        &self,
        source_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        params: &TransformParams,
    ) -> Option<GpuPassTimer> {
        let timer = self.pass_timer();
        let params_bytes = transform_params_bytes(params);
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("smart-compress-transform-params"),
                contents: &params_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smart-compress-transform-bind-group"),
            layout: &self.transform_bind_group_layout,
            entries: &[
                bind_entry(0, source_buffer),
                bind_entry(1, output_buffer),
                bind_entry(2, &params_buffer),
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("smart-compress-transform-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smart-compress-transform-pass"),
                timestamp_writes: timer.as_ref().map(GpuPassTimer::writes),
            });
            pass.set_pipeline(&self.transform_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups_x = params.output_width.div_ceil(16);
            let groups_y = params.out_row_count.div_ceil(16);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        if let Some(timer) = &timer {
            timer.resolve(&mut encoder);
        }
        self.queue.submit([encoder.finish()]);
        timer
    }

    /// A timer for one compute pass, or `None` when the adapter has no
    /// timestamp queries.
    fn pass_timer(&self) -> Option<GpuPassTimer> {
        self.adapter_report.timestamp_queries.then(|| GpuPassTimer {
            query_set: self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("smart-compress-pass-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolved: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("smart-compress-pass-timestamps-resolved"),
                size: GPU_TIMESTAMP_BYTES,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("smart-compress-pass-timestamps-readback"),
                size: GPU_TIMESTAMP_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        })
    }

    /// Duration of the timed pass in nanoseconds. Its submission must have
    /// completed already.
    fn pass_nanos(&self, timer: &GpuPassTimer) -> anyhow::Result<u64> {
        // Copied in a submission of its own: in the pass's encoder the copy
        // read stale bytes on Vulkan (AMD) for the first passes of a device.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("smart-compress-pass-timestamps-encoder"),
            });
        encoder.copy_buffer_to_buffer(&timer.resolved, 0, &timer.readback, 0, GPU_TIMESTAMP_BYTES);
        self.queue.submit([encoder.finish()]);
        let slice = timer.readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(anyhow::Error::from)?;
        receiver.recv()??;

        let mapped = slice.get_mapped_range();
        let [start, end] = [0, 1].map(|index| u64::from_le_bytes(mapped.as_chunks::<8>().0[index]));
        drop(mapped);
        timer.readback.unmap();
        Ok(
            (end.saturating_sub(start) as f64 * f64::from(self.queue.get_timestamp_period()))
                as u64,
        )
    }

    fn readback_rgba(
        &self,
        buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> anyhow::Result<RgbaImage> {
        let byte_len = width as u64 * height as u64 * 4;
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smart-compress-transform-readback"),
            size: byte_len.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("smart-compress-transform-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &readback_buffer, 0, byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = readback_buffer.slice(..byte_len);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(anyhow::Error::from)?;
        receiver.recv()??;

        let mapped = slice.get_mapped_range();
        let image = RgbaImage::from_raw(width, height, mapped.to_vec())
            .ok_or_else(|| anyhow::anyhow!("GPU readback size did not match {width}x{height}"))?;
        drop(mapped);
        readback_buffer.unmap();
        Ok(image)
    }

    fn metric_scores(
        &self,
        original: &RgbaImage,
        candidate: &RgbaImage,
    ) -> anyhow::Result<(GpuMetricScores, GpuTimingBreakdown)> {
        let pixel_count = original.width() as u64 * original.height() as u64;
        if pixel_count == 0 {
            return Ok((GpuMetricScores::lossless(), GpuTimingBreakdown::default()));
        }

        let sums = self.metric_sums(original, candidate)?;
        Ok((
            sums.to_scores(original.width(), original.height()),
            sums.timings,
        ))
    }

    fn metric_sums(
        &self,
        original: &RgbaImage,
        candidate: &RgbaImage,
    ) -> anyhow::Result<GpuMetricSums> {
        let width = original.width();
        let height = original.height();
        let Some(score_rows_per_tile) = self.score_rows_per_tile(width, height)? else {
            return self.metric_sums_for_rows(
                original.as_raw(),
                candidate.as_raw(),
                GpuDispatchParams {
                    buffer_height: height,
                    global_y_offset: 0,
                    global_height: height,
                    score_start_local_y: 0,
                    score_row_count: height,
                    width,
                },
            );
        };

        let row_bytes = width as usize * 4;
        let mut totals = GpuMetricSums::default();
        let mut score_start = 0_u32;
        while score_start < height {
            let score_rows = score_rows_per_tile.min(height - score_start);
            let halo_start = score_start.saturating_sub(1);
            let score_end = score_start + score_rows;
            let halo_end = score_end.saturating_add(1).min(height);
            let start = halo_start as usize * row_bytes;
            let end = halo_end as usize * row_bytes;
            let sums = self.metric_sums_for_rows(
                &original.as_raw()[start..end],
                &candidate.as_raw()[start..end],
                GpuDispatchParams {
                    buffer_height: halo_end - halo_start,
                    global_y_offset: halo_start,
                    global_height: height,
                    score_start_local_y: score_start - halo_start,
                    score_row_count: score_rows,
                    width,
                },
            )?;
            totals.add(sums);
            score_start = score_end;
        }
        Ok(totals)
    }

    fn metric_sums_for_rows(
        &self,
        original_bytes: &[u8],
        candidate_bytes: &[u8],
        params: GpuDispatchParams,
    ) -> anyhow::Result<GpuMetricSums> {
        let pixel_count = params.width as u64 * params.buffer_height as u64;
        if pixel_count == 0 {
            return Ok(GpuMetricSums::default());
        }

        let upload_start = Instant::now();
        let original_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("smart-compress-original-rgba"),
                contents: original_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let candidate_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("smart-compress-candidate-rgba"),
                contents: candidate_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mut sums =
            self.metric_sums_from_buffers(&original_buffer, &candidate_buffer, params)?;
        sums.timings.upload_ms = Some(upload_start.elapsed().as_millis() as u64);
        Ok(sums)
    }

    fn metric_sums_from_buffers(
        &self,
        original_buffer: &wgpu::Buffer,
        candidate_buffer: &wgpu::Buffer,
        params: GpuDispatchParams,
    ) -> anyhow::Result<GpuMetricSums> {
        let pixel_count = params.width as u64 * params.buffer_height as u64;
        if pixel_count == 0 {
            return Ok(GpuMetricSums::default());
        }

        let workgroup_size = 256_u64;
        let workgroups = pixel_count.div_ceil(workgroup_size);
        let result_values = workgroups * GPU_METRIC_COMPONENTS as u64;
        let result_bytes = result_values * std::mem::size_of::<u32>() as u64;
        let params = params_bytes(&params);

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smart-compress-metric-partials"),
            size: result_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smart-compress-metric-partials-readback"),
            size: result_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("smart-compress-metric-params"),
                contents: &params,
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smart-compress-metric-sums-bind-group"),
            layout: &self.metric_bind_group_layout,
            entries: &[
                bind_entry(0, original_buffer),
                bind_entry(1, candidate_buffer),
                bind_entry(2, &result_buffer),
                bind_entry(3, &params_buffer),
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("smart-compress-metric-sums-encoder"),
            });
        let timer = self.pass_timer();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smart-compress-metric-sums-pass"),
                timestamp_writes: timer.as_ref().map(GpuPassTimer::writes),
            });
            pass.set_pipeline(&self.metric_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        if let Some(timer) = &timer {
            timer.resolve(&mut encoder);
        }
        encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback_buffer, 0, result_bytes);
        let compute_start = Instant::now();
        self.queue.submit([encoder.finish()]);

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(anyhow::Error::from)?;
        let compute_ms = compute_start.elapsed().as_millis() as u64;
        let readback_start = Instant::now();
        receiver.recv()??;

        let mapped = slice.get_mapped_range();
        let mut sums = [0_u64; GPU_METRIC_COMPONENTS];
        for (index, chunk) in mapped.as_chunks::<4>().0.iter().enumerate() {
            sums[index % GPU_METRIC_COMPONENTS] += u32::from_le_bytes(*chunk) as u64;
        }
        drop(mapped);
        readback_buffer.unmap();
        let readback_ms = readback_start.elapsed().as_millis() as u64;
        // A failed timestamp read must not fail the score it measures.
        let timestamp_ns = timer.as_ref().and_then(|timer| self.pass_nanos(timer).ok());

        Ok(GpuMetricSums {
            sums,
            timings: GpuTimingBreakdown {
                upload_ms: None,
                transform_ms: None,
                compute_ms: Some(compute_ms),
                readback_ms: Some(readback_ms),
                timestamp_ns,
            },
        })
    }

    fn score_rows_per_tile(&self, width: u32, height: u32) -> anyhow::Result<Option<u32>> {
        let Some(limit) = self.max_dispatch_bytes else {
            return Ok(None);
        };
        if dispatch_bytes(width, height) <= limit {
            return Ok(None);
        }

        let mut low = 1_u32;
        let mut high = height;
        let mut best = 0_u32;
        while low <= high {
            let mid = low + (high - low) / 2;
            let buffer_rows = mid.saturating_add(2).min(height);
            if dispatch_bytes(width, buffer_rows) <= limit {
                best = mid;
                low = mid.saturating_add(1);
            } else {
                high = mid.saturating_sub(1);
            }
        }
        if best == 0 {
            anyhow::bail!(
                "GPU memory limit {limit} bytes is too small for a single {width}px metric tile"
            );
        }
        Ok(Some(best))
    }
}

impl ComputeBackend for GpuWgpuBackend {
    fn backend_id(&self) -> &'static str {
        "gpu-wgpu"
    }

    fn score_candidate(
        &self,
        original: &DynamicImage,
        candidate: &DynamicImage,
        has_alpha: bool,
    ) -> Result<QualityScore, MetricError> {
        self.score_candidate_with_fallback(original, candidate, has_alpha)
            .map(|outcome| outcome.quality)
    }
}

fn quality_from_metric_scores(
    original_rgba: &RgbaImage,
    candidate_rgba: &RgbaImage,
    has_alpha: bool,
    gpu_scores: Option<&GpuMetricScores>,
) -> QualityScore {
    let rgb = gpu_scores.map_or_else(
        || rgb_similarity(original_rgba, candidate_rgba),
        |scores| scores.rgb,
    );
    let edge = gpu_scores.map_or_else(
        || edge_similarity(original_rgba, candidate_rgba),
        |scores| scores.edge,
    );
    let alpha = has_alpha.then(|| {
        gpu_scores.map_or_else(
            || alpha_similarity(original_rgba, candidate_rgba),
            |scores| scores.alpha,
        )
    });
    let composite_min = has_alpha.then(|| {
        gpu_scores.map_or_else(
            || composite_score(original_rgba, candidate_rgba),
            |scores| scores.composite_min,
        )
    });
    let overall = [Some(rgb), Some(edge), alpha, composite_min]
        .into_iter()
        .flatten()
        .fold(100.0_f32, f32::min);

    QualityScore {
        rgb,
        ssimulacra2: None,
        butteraugli: None,
        alpha,
        composite_min,
        alpha_edge: None,
        edge,
        overall,
    }
}

#[derive(Debug, Clone, Copy)]
struct GpuMetricScores {
    rgb: f32,
    alpha: f32,
    composite_min: f32,
    edge: f32,
}

impl GpuMetricScores {
    fn lossless() -> Self {
        Self {
            rgb: 100.0,
            alpha: 100.0,
            composite_min: 100.0,
            edge: 100.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct GpuMetricSums {
    sums: [u64; GPU_METRIC_COMPONENTS],
    timings: GpuTimingBreakdown,
}

impl GpuMetricSums {
    fn add(&mut self, other: Self) {
        for (left, right) in self.sums.iter_mut().zip(other.sums) {
            *left += right;
        }
        self.timings.add(other.timings);
    }

    fn to_scores(self, width: u32, height: u32) -> GpuMetricScores {
        let pixel_count = width as u64 * height as u64;
        let composite_min = self.sums[2..7]
            .iter()
            .map(|sum| similarity_from_sse(*sum as f64, pixel_count * 3))
            .fold(100.0_f32, f32::min);
        let rgb = similarity_from_sse(self.sums[0] as f64, self.sums[8] * 3);
        let edge = if width < 3 || height < 3 {
            rgb
        } else {
            similarity_from_sse(self.sums[7] as f64, self.sums[9])
        };

        GpuMetricScores {
            rgb,
            alpha: similarity_from_sse(self.sums[1] as f64, pixel_count),
            composite_min,
            edge,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct GpuDispatchParams {
    width: u32,
    buffer_height: u32,
    global_y_offset: u32,
    global_height: u32,
    score_start_local_y: u32,
    score_row_count: u32,
}

#[derive(Debug, Clone, Copy)]
struct TransformParams {
    source_width: u32,
    source_height: u32,
    output_width: u32,
    output_height: u32,
    orientation_code: u32,
    /// Origin and row stride of the uploaded source band, in source pixels.
    tile_x: u32,
    tile_y: u32,
    tile_width: u32,
    /// Output rows this dispatch writes.
    out_row_offset: u32,
    out_row_count: u32,
}

const TRANSFORM_PARAMS_BYTES: u64 = 48;

pub enum ActiveComputeBackend {
    Cpu(CpuRayonBackend),
    Gpu(Box<GpuWgpuBackend>),
}

impl ActiveComputeBackend {
    pub fn as_gpu(&self) -> Option<&GpuWgpuBackend> {
        match self {
            Self::Gpu(gpu) => Some(gpu),
            Self::Cpu(_) => None,
        }
    }
}

impl ComputeBackend for ActiveComputeBackend {
    fn backend_id(&self) -> &'static str {
        match self {
            Self::Cpu(cpu) => cpu.backend_id(),
            Self::Gpu(gpu) => gpu.backend_id(),
        }
    }

    fn score_candidate(
        &self,
        original: &DynamicImage,
        candidate: &DynamicImage,
        has_alpha: bool,
    ) -> Result<QualityScore, MetricError> {
        match self {
            Self::Cpu(cpu) => cpu.score_candidate(original, candidate, has_alpha),
            Self::Gpu(gpu) => gpu.score_candidate(original, candidate, has_alpha),
        }
    }
}

fn check_dimensions(original: &DynamicImage, candidate: &DynamicImage) -> Result<(), MetricError> {
    if original.dimensions() != candidate.dimensions() {
        return Err(MetricError::DimensionMismatch {
            original: original.dimensions(),
            candidate: candidate.dimensions(),
        });
    }
    Ok(())
}

/// The RGBA buffer the built-in metrics compare for `image`: reduced to the
/// metric size of an original with `original_dimensions`, if that is smaller.
fn metric_level(image: &DynamicImage, original_dimensions: (u32, u32)) -> RgbaImage {
    // Decoded channel values are taken as sRGB-relative ones, as they are;
    // `report::METRIC_COLOR_SPACE` names this policy.
    let (width, height) = original_dimensions;
    let (metric_width, metric_height) = bounded_metric_dimensions(width, height);
    let rgba = image.to_rgba8();
    if (metric_width, metric_height) != (width, height) {
        box_downsample(&rgba, metric_width, metric_height)
    } else {
        rgba
    }
}

/// The canonical metric downsample, shared by every backend: each output
/// pixel is the rounded mean of the source pixels in its footprint
/// `[floor(i * source / output), ceil((i + 1) * source / output))`, which is
/// never empty and never reaches past the image edge. The GPU transform
/// shader computes exactly this, so reduced levels match bit for bit.
pub(crate) fn box_downsample(source: &RgbaImage, width: u32, height: u32) -> RgbaImage {
    use rayon::prelude::*;

    let (source_width, source_height) = source.dimensions();
    if (source_width, source_height) == (width, height) || width == 0 || height == 0 {
        return source.clone();
    }
    let span = |index: u32, source: u32, output: u32| {
        let start = (u64::from(index) * u64::from(source) / u64::from(output)) as usize;
        let end = ((u64::from(index) + 1) * u64::from(source)).div_ceil(u64::from(output)) as usize;
        (start, end.max(start + 1))
    };
    let columns = (0..width)
        .map(|x| span(x, source_width, width))
        .collect::<Vec<_>>();
    let raw = source.as_raw();
    let stride = source_width as usize * 4;
    let mut reduced = vec![0_u8; width as usize * height as usize * 4];
    reduced
        .par_chunks_exact_mut(width as usize * 4)
        .enumerate()
        .for_each(|(y, row)| {
            let (top, bottom) = span(y as u32, source_height, height);
            for (pixel, (left, right)) in row.as_chunks_mut::<4>().0.iter_mut().zip(&columns) {
                let mut sums = [0_u64; 4];
                for line in raw[top * stride..bottom * stride].chunks_exact(stride) {
                    for source_pixel in line[left * 4..right * 4].as_chunks::<4>().0 {
                        for (sum, channel) in sums.iter_mut().zip(source_pixel) {
                            *sum += u64::from(*channel);
                        }
                    }
                }
                let count = ((bottom - top) * (right - left)) as u64;
                for (channel, sum) in pixel.iter_mut().zip(sums) {
                    *channel = ((sum + count / 2) / count) as u8;
                }
            }
        });
    RgbaImage::from_raw(width, height, reduced).expect("reduced buffer matches its dimensions")
}

/// Largest image, in pixels, the metrics read as it is. The built-in metrics
/// reduce a larger one to at most 2048 px on a side; the external ones read a
/// mosaic of samples from it.
pub const FULL_SIZE_METRIC_PIXELS: u64 = 16_000_000;

pub(crate) fn bounded_metric_dimensions(width: u32, height: u32) -> (u32, u32) {
    let pixels = width as u64 * height as u64;
    if pixels <= FULL_SIZE_METRIC_PIXELS || width == 0 || height == 0 {
        return (width, height);
    }
    let longest = width.max(height) as f64;
    let scale = 2048.0 / longest;
    (
        ((width as f64 * scale).round() as u32).max(1),
        ((height as f64 * scale).round() as u32).max(1),
    )
}

fn bounded_preview_dimensions(width: u32, height: u32, max_side: u32) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (width, height);
    }
    let longest = width.max(height);
    if longest <= max_side {
        return (width, height);
    }
    let scale = max_side as f64 / longest as f64;
    (
        ((width as f64 * scale).round() as u32).max(1),
        ((height as f64 * scale).round() as u32).max(1),
    )
}

fn oriented_dimensions(width: u32, height: u32, orientation: Orientation) -> (u32, u32) {
    match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Rotate90FlipH
        | Orientation::Rotate270FlipH => (height, width),
        _ => (width, height),
    }
}

fn similarity_from_sse(sum: f64, count: u64) -> f32 {
    if count == 0 {
        return 100.0;
    }
    let mse = sum / count as f64;
    if mse <= f64::EPSILON {
        return 100.0;
    }
    let psnr = 10.0 * ((255.0_f64 * 255.0_f64) / mse).log10();
    (psnr * 2.0).clamp(0.0, 100.0) as f32
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn params_bytes(params: &GpuDispatchParams) -> [u8; 32] {
    let mut bytes = [0_u8; 32];
    let pixel_count = params.width * params.buffer_height;
    bytes[0..4].copy_from_slice(&pixel_count.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.width.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.buffer_height.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.global_y_offset.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.global_height.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.score_start_local_y.to_le_bytes());
    bytes[24..28].copy_from_slice(&params.score_row_count.to_le_bytes());
    bytes
}

fn transform_params_bytes(params: &TransformParams) -> [u8; TRANSFORM_PARAMS_BYTES as usize] {
    let mut bytes = [0_u8; TRANSFORM_PARAMS_BYTES as usize];
    for (index, value) in [
        params.source_width,
        params.source_height,
        params.output_width,
        params.output_height,
        params.orientation_code,
        params.tile_x,
        params.tile_y,
        params.tile_width,
        params.out_row_offset,
        params.out_row_count,
    ]
    .into_iter()
    .enumerate()
    {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Sums a dispatch returns: raw RGB, alpha, five composites, edge, then the
/// pixels and the edge samples that are visible in at least one image.
const GPU_METRIC_COMPONENTS: usize = 10;
const GPU_WORKGROUP_SIZE: u64 = 256;

fn dispatch_bytes(width: u32, rows: u32) -> u64 {
    let pixels = width as u64 * rows as u64;
    let rgba_bytes = pixels.saturating_mul(4);
    let partial_bytes = pixels
        .div_ceil(GPU_WORKGROUP_SIZE)
        .saturating_mul(GPU_METRIC_COMPONENTS as u64)
        .saturating_mul(std::mem::size_of::<u32>() as u64);
    rgba_bytes
        .saturating_mul(2)
        .saturating_add(partial_bytes.saturating_mul(2))
        .saturating_add(32)
}

const METRIC_SUMS_SHADER: &str = r#"
struct Params {
    pixel_count: u32,
    width: u32,
    buffer_height: u32,
    global_y_offset: u32,
    global_height: u32,
    score_start_local_y: u32,
    score_row_count: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> original: array<u32>;
@group(0) @binding(1) var<storage, read> candidate: array<u32>;
@group(0) @binding(2) var<storage, read_write> partials: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> scratch: array<u32, 256>;
var<workgroup> scratch_alpha: array<u32, 256>;
var<workgroup> scratch_comp0: array<u32, 256>;
var<workgroup> scratch_comp1: array<u32, 256>;
var<workgroup> scratch_comp2: array<u32, 256>;
var<workgroup> scratch_comp3: array<u32, 256>;
var<workgroup> scratch_comp4: array<u32, 256>;
var<workgroup> scratch_edge: array<u32, 256>;
var<workgroup> scratch_visible: array<u32, 256>;
var<workgroup> scratch_visible_edge: array<u32, 256>;

fn channel(value: u32, shift: u32) -> i32 {
    return i32((value >> shift) & 0xffu);
}

fn sqdiff(a: i32, b: i32) -> u32 {
    let diff = a - b;
    return u32(diff * diff);
}

fn blend_channel(foreground: i32, alpha: i32, background: i32) -> i32 {
    return (foreground * alpha + background * (255i - alpha) + 127i) / 255i;
}

fn luma(value: u32) -> i32 {
    let r = f32(channel(value, 0u));
    let g = f32(channel(value, 8u));
    let b = f32(channel(value, 16u));
    return i32(round(0.2126 * r + 0.7152 * g + 0.0722 * b));
}

// Fully transparent in both images: its colour is invisible, so it is no sample
// of the raw RGB and edge scores and has luma 0 in both.
fn hidden(image_index: u32) -> bool {
    return channel(original[image_index], 24u) == 0 && channel(candidate[image_index], 24u) == 0;
}

fn luma_original(image_index: u32) -> i32 {
    if (hidden(image_index)) {
        return 0;
    }
    return luma(original[image_index]);
}

fn luma_candidate(image_index: u32) -> i32 {
    if (hidden(image_index)) {
        return 0;
    }
    return luma(candidate[image_index]);
}

fn edge_original(image_index: u32) -> i32 {
    let left = luma_original(image_index - 1u);
    let right = luma_original(image_index + 1u);
    let top = luma_original(image_index - params.width);
    let bottom = luma_original(image_index + params.width);
    return abs(left - right) + abs(top - bottom);
}

fn edge_candidate(image_index: u32) -> i32 {
    let left = luma_candidate(image_index - 1u);
    let right = luma_candidate(image_index + 1u);
    let top = luma_candidate(image_index - params.width);
    let bottom = luma_candidate(image_index + params.width);
    return abs(left - right) + abs(top - bottom);
}

fn composite_sse(a: u32, b: u32, background: vec3<i32>) -> u32 {
    let aa = channel(a, 24u);
    let ba = channel(b, 24u);
    let ar = blend_channel(channel(a, 0u), aa, background.x);
    let ag = blend_channel(channel(a, 8u), aa, background.y);
    let ab = blend_channel(channel(a, 16u), aa, background.z);
    let br = blend_channel(channel(b, 0u), ba, background.x);
    let bg = blend_channel(channel(b, 8u), ba, background.y);
    let bb = blend_channel(channel(b, 16u), ba, background.z);
    return sqdiff(ar, br) + sqdiff(ag, bg) + sqdiff(ab, bb);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let index = global_id.x;
    var sum_rgb: u32 = 0u;
    var sum_alpha: u32 = 0u;
    var sum_comp0: u32 = 0u;
    var sum_comp1: u32 = 0u;
    var sum_comp2: u32 = 0u;
    var sum_comp3: u32 = 0u;
    var sum_comp4: u32 = 0u;
    var sum_edge: u32 = 0u;
    var sum_visible: u32 = 0u;
    var sum_visible_edge: u32 = 0u;
    if (index < params.pixel_count) {
        let x = index % params.width;
        let y = index / params.width;
        let global_y = params.global_y_offset + y;
        let scored = y >= params.score_start_local_y
            && y < params.score_start_local_y + params.score_row_count;
        if (scored) {
            let a = original[index];
            let b = candidate[index];
            let visible = !hidden(index);
            if (visible) {
                sum_rgb = sqdiff(channel(a, 0u), channel(b, 0u))
                    + sqdiff(channel(a, 8u), channel(b, 8u))
                    + sqdiff(channel(a, 16u), channel(b, 16u));
                sum_visible = 1u;
            }
            sum_alpha = sqdiff(channel(a, 24u), channel(b, 24u));
            sum_comp0 = composite_sse(a, b, vec3<i32>(255i, 255i, 255i));
            sum_comp1 = composite_sse(a, b, vec3<i32>(0i, 0i, 0i));
            sum_comp2 = composite_sse(a, b, vec3<i32>(128i, 128i, 128i));
            sum_comp3 = composite_sse(a, b, vec3<i32>(238i, 238i, 238i));
            sum_comp4 = composite_sse(a, b, vec3<i32>(64i, 64i, 64i));
            if (visible
                && x > 0u
                && y > 0u
                && global_y > 0u
                && x + 1u < params.width
                && y + 1u < params.buffer_height
                && global_y + 1u < params.global_height) {
                sum_edge = sqdiff(edge_original(index), edge_candidate(index));
                sum_visible_edge = 1u;
            }
        }
    }
    scratch[local_id.x] = sum_rgb;
    scratch_alpha[local_id.x] = sum_alpha;
    scratch_comp0[local_id.x] = sum_comp0;
    scratch_comp1[local_id.x] = sum_comp1;
    scratch_comp2[local_id.x] = sum_comp2;
    scratch_comp3[local_id.x] = sum_comp3;
    scratch_comp4[local_id.x] = sum_comp4;
    scratch_edge[local_id.x] = sum_edge;
    scratch_visible[local_id.x] = sum_visible;
    scratch_visible_edge[local_id.x] = sum_visible_edge;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (local_id.x < stride) {
            scratch[local_id.x] = scratch[local_id.x] + scratch[local_id.x + stride];
            scratch_alpha[local_id.x] = scratch_alpha[local_id.x] + scratch_alpha[local_id.x + stride];
            scratch_comp0[local_id.x] = scratch_comp0[local_id.x] + scratch_comp0[local_id.x + stride];
            scratch_comp1[local_id.x] = scratch_comp1[local_id.x] + scratch_comp1[local_id.x + stride];
            scratch_comp2[local_id.x] = scratch_comp2[local_id.x] + scratch_comp2[local_id.x + stride];
            scratch_comp3[local_id.x] = scratch_comp3[local_id.x] + scratch_comp3[local_id.x + stride];
            scratch_comp4[local_id.x] = scratch_comp4[local_id.x] + scratch_comp4[local_id.x + stride];
            scratch_edge[local_id.x] = scratch_edge[local_id.x] + scratch_edge[local_id.x + stride];
            scratch_visible[local_id.x] = scratch_visible[local_id.x] + scratch_visible[local_id.x + stride];
            scratch_visible_edge[local_id.x] = scratch_visible_edge[local_id.x] + scratch_visible_edge[local_id.x + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (local_id.x == 0u) {
        let base = workgroup_id.x * 10u;
        partials[base] = scratch[0];
        partials[base + 1u] = scratch_alpha[0];
        partials[base + 2u] = scratch_comp0[0];
        partials[base + 3u] = scratch_comp1[0];
        partials[base + 4u] = scratch_comp2[0];
        partials[base + 5u] = scratch_comp3[0];
        partials[base + 6u] = scratch_comp4[0];
        partials[base + 7u] = scratch_edge[0];
        partials[base + 8u] = scratch_visible[0];
        partials[base + 9u] = scratch_visible_edge[0];
    }
}
"#;

const TRANSFORM_DOWNSAMPLE_SHADER: &str = r#"
struct Params {
    source_width: u32,
    source_height: u32,
    output_width: u32,
    output_height: u32,
    orientation_code: u32,
    tile_x: u32,
    tile_y: u32,
    tile_width: u32,
    out_row_offset: u32,
    out_row_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn channel(value: u32, shift: u32) -> u32 {
    return (value >> shift) & 0xffu;
}

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

fn oriented_width() -> u32 {
    if (params.orientation_code >= 5u) {
        return params.source_height;
    }
    return params.source_width;
}

fn oriented_height() -> u32 {
    if (params.orientation_code >= 5u) {
        return params.source_width;
    }
    return params.source_height;
}

fn source_index(oriented_x: u32, oriented_y: u32) -> u32 {
    var source_x = oriented_x;
    var source_y = oriented_y;
    switch params.orientation_code {
        case 1u: {
        }
        case 2u: {
            source_x = params.source_width - 1u - oriented_x;
        }
        case 3u: {
            source_x = params.source_width - 1u - oriented_x;
            source_y = params.source_height - 1u - oriented_y;
        }
        case 4u: {
            source_y = params.source_height - 1u - oriented_y;
        }
        case 5u: {
            source_x = oriented_y;
            source_y = oriented_x;
        }
        case 6u: {
            source_x = oriented_y;
            source_y = params.source_height - 1u - oriented_x;
        }
        case 7u: {
            source_x = params.source_width - 1u - oriented_y;
            source_y = params.source_height - 1u - oriented_x;
        }
        default: {
            source_x = params.source_width - 1u - oriented_y;
            source_y = oriented_x;
        }
    }
    return (source_y - params.tile_y) * params.tile_width + (source_x - params.tile_x);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y + params.out_row_offset;
    if (out_x >= params.output_width || global_id.y >= params.out_row_count) {
        return;
    }

    let src_oriented_width = oriented_width();
    let src_oriented_height = oriented_height();
    let start_x = (out_x * src_oriented_width) / params.output_width;
    let start_y = (out_y * src_oriented_height) / params.output_height;
    let end_x = max(
        start_x + 1u,
        ((out_x + 1u) * src_oriented_width + params.output_width - 1u) / params.output_width,
    );
    let end_y = max(
        start_y + 1u,
        ((out_y + 1u) * src_oriented_height + params.output_height - 1u) / params.output_height,
    );

    var sum_r = 0u;
    var sum_g = 0u;
    var sum_b = 0u;
    var sum_a = 0u;
    var count = 0u;
    for (var source_y = start_y; source_y < end_y; source_y = source_y + 1u) {
        for (var source_x = start_x; source_x < end_x; source_x = source_x + 1u) {
            let pixel = source[source_index(source_x, source_y)];
            sum_r = sum_r + channel(pixel, 0u);
            sum_g = sum_g + channel(pixel, 8u);
            sum_b = sum_b + channel(pixel, 16u);
            sum_a = sum_a + channel(pixel, 24u);
            count = count + 1u;
        }
    }

    let half = count / 2u;
    let average_r = (sum_r + half) / count;
    let average_g = (sum_g + half) / count;
    let average_b = (sum_b + half) / count;
    let average_a = (sum_a + half) / count;
    let output_index = out_y * params.output_width + out_x;
    output[output_index] = pack_rgba(average_r, average_g, average_b, average_a);
}
"#;

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgba, RgbaImage, metadata::Orientation};

    use super::*;

    #[test]
    fn missing_gpu_skips_unless_required() {
        assert!(gpu_or_skip_with(Err(anyhow::anyhow!("no adapter")), false).is_none());
    }

    #[test]
    #[should_panic(expected = "SMART_COMPRESS_REQUIRE_GPU is set")]
    fn missing_gpu_fails_when_required() {
        let _ = gpu_or_skip_with(Err(anyhow::anyhow!("no adapter")), true);
    }

    #[test]
    fn device_is_created_from_the_adapter_named_in_the_report() {
        let (report, adapter) =
            crate::gpu::detect_gpu_with_adapter(&crate::config::GpuConfig::default());
        let Some(adapter) = adapter else {
            let _ = gpu_or_skip(Err(anyhow::anyhow!("{}", report.warnings.join("; "))));
            return;
        };
        let info = adapter.get_info();
        let selected = report.selected_adapter.expect("report names the adapter");
        assert_eq!(
            (&selected.name, selected.vendor, selected.device),
            (&info.name, info.vendor, info.device)
        );
        assert_eq!(selected.backend, format!("{:?}", info.backend));

        let Some(gpu) = gpu_or_skip(GpuWgpuBackend::from_adapter(adapter, None)) else {
            return;
        };
        let used = gpu.adapter_report();
        assert_eq!(
            (&used.name, used.vendor, used.device, &used.backend),
            (
                &selected.name,
                selected.vendor,
                selected.device,
                &selected.backend
            )
        );
    }

    #[test]
    fn shared_original_scores_exactly_like_a_fresh_conversion() {
        let original = DynamicImage::ImageRgba8(RgbaImage::from_fn(67, 41, |x, y| {
            Rgba([(x * 7) as u8, (y * 5) as u8, 90, ((x + y * 3) % 256) as u8])
        }));
        let candidate = DynamicImage::ImageRgba8(RgbaImage::from_fn(67, 41, |x, y| {
            Rgba([
                (x * 7 + 3) as u8,
                (y * 5) as u8,
                91,
                ((x + y * 3) % 251) as u8,
            ])
        }));
        let shared = OriginalImage::from(original.clone());

        assert_eq!(
            CpuRayonBackend
                .score_original(&shared, &candidate, true)
                .unwrap(),
            CpuRayonBackend
                .score_candidate(&original, &candidate, true)
                .unwrap()
        );
        if let Some(gpu) = gpu_or_skip(GpuWgpuBackend::new()) {
            assert_eq!(
                gpu.score_original_with_fallback(&shared, &candidate, true)
                    .unwrap()
                    .quality,
                gpu.score_candidate_with_fallback(&original, &candidate, true)
                    .unwrap()
                    .quality
            );
        }
    }

    #[test]
    fn gpu_scores_match_cpu_for_transparent_fixture_when_available() {
        let Some(gpu) = gpu_or_skip(GpuWgpuBackend::new()) else {
            return;
        };

        let mut original = RgbaImage::new(32, 32);
        let mut candidate = RgbaImage::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                let alpha = ((x * 7 + y * 3) % 256) as u8;
                original.put_pixel(x, y, Rgba([x as u8 * 7, y as u8 * 5, 120, alpha]));
                candidate.put_pixel(
                    x,
                    y,
                    Rgba([
                        x as u8 * 7,
                        y as u8 * 5,
                        120_u8.saturating_add((x % 3) as u8),
                        alpha.saturating_sub((y % 5) as u8),
                    ]),
                );
            }
        }

        let original = DynamicImage::ImageRgba8(original);
        let candidate = DynamicImage::ImageRgba8(candidate);
        let cpu_score = CpuRayonBackend
            .score_candidate(&original, &candidate, true)
            .unwrap();
        let gpu_score = gpu.score_candidate(&original, &candidate, true).unwrap();

        assert_close(cpu_score.rgb, gpu_score.rgb);
        assert_close(cpu_score.edge, gpu_score.edge);
        assert_close(cpu_score.alpha.unwrap(), gpu_score.alpha.unwrap());
        assert_close(
            cpu_score.composite_min.unwrap(),
            gpu_score.composite_min.unwrap(),
        );
        assert_close(cpu_score.overall, gpu_score.overall);
    }

    #[test]
    fn forced_tiled_gpu_scores_match_untiled_gpu_when_available() {
        let Some(untiled_gpu) = gpu_or_skip(GpuWgpuBackend::new()) else {
            return;
        };
        let Some(tiled_gpu) = gpu_or_skip(GpuWgpuBackend::new_with_max_vram(Some(4 * 1024))) else {
            return;
        };

        let mut original = RgbaImage::new(64, 65);
        let mut candidate = RgbaImage::new(64, 65);
        for y in 0..65 {
            for x in 0..64 {
                let alpha = ((x * 5 + y * 9) % 256) as u8;
                original.put_pixel(
                    x,
                    y,
                    Rgba([
                        ((x * 11 + y * 3) % 256) as u8,
                        ((x * 7 + y * 13) % 256) as u8,
                        ((x * 17 + y * 2) % 256) as u8,
                        alpha,
                    ]),
                );
                candidate.put_pixel(
                    x,
                    y,
                    Rgba([
                        ((x * 11 + y * 3 + 1) % 256) as u8,
                        ((x * 7 + y * 13) % 256) as u8,
                        ((x * 17 + y * 2 + 2) % 256) as u8,
                        alpha.saturating_sub((x % 4) as u8),
                    ]),
                );
            }
        }

        let original = DynamicImage::ImageRgba8(original);
        let candidate = DynamicImage::ImageRgba8(candidate);
        let untiled = untiled_gpu
            .score_candidate(&original, &candidate, true)
            .unwrap();
        let tiled = tiled_gpu
            .score_candidate(&original, &candidate, true)
            .unwrap();

        assert_close(untiled.rgb, tiled.rgb);
        assert_close(untiled.edge, tiled.edge);
        assert_close(untiled.alpha.unwrap(), tiled.alpha.unwrap());
        assert_close(untiled.composite_min.unwrap(), tiled.composite_min.unwrap());
        assert_close(untiled.overall, tiled.overall);
    }

    #[test]
    fn gpu_metric_failure_reports_cpu_fallback_when_available() {
        let Some(gpu) = gpu_or_skip(GpuWgpuBackend::new_with_max_vram(Some(1))) else {
            return;
        };
        let original =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, Rgba([10, 20, 30, 255])));
        let candidate =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, Rgba([11, 20, 30, 255])));

        let outcome = gpu
            .score_candidate_with_fallback(&original, &candidate, false)
            .unwrap();
        let cpu = CpuRayonBackend
            .score_candidate(&original, &candidate, false)
            .unwrap();

        assert!(outcome.fallback_reason.is_some());
        assert_close(cpu.overall, outcome.quality.overall);
    }

    #[test]
    fn gpu_timestamp_time_is_reported_only_with_timestamp_queries_when_available() {
        let Some(gpu) = gpu_or_skip(GpuWgpuBackend::new()) else {
            return;
        };
        let pixel = |x: u32, y: u32, shift: u32| {
            Rgba([
                ((x * 13 + y * 3 + shift) % 256) as u8,
                ((x * 5 + y * 11) % 256) as u8,
                ((x * 17 + y * 7) % 256) as u8,
                255,
            ])
        };
        let original =
            DynamicImage::ImageRgba8(RgbaImage::from_fn(1024, 1024, |x, y| pixel(x, y, 0)));
        let candidate =
            DynamicImage::ImageRgba8(RgbaImage::from_fn(1024, 1024, |x, y| pixel(x, y, 2)));
        // Metric pass alone first, then transform plus metric.
        let start = Instant::now();
        let metric_only = gpu
            .score_candidate_with_fallback(&original, &candidate, false)
            .unwrap()
            .timings;
        let metric_wall_ns = start.elapsed().as_nanos() as u64;
        let prepared = gpu
            .prepare_original(&original, Orientation::NoTransforms)
            .unwrap();
        let start = Instant::now();
        let (_, prepared_timings) = gpu
            .score_candidate_prepared_with_timings(&prepared, &candidate, false)
            .unwrap();
        let prepared_wall_ns = start.elapsed().as_nanos() as u64;

        for (timings, wall_ns) in [
            (metric_only, metric_wall_ns),
            (prepared_timings, prepared_wall_ns),
        ] {
            if gpu.adapter_report().timestamp_queries {
                let gpu_ns = timings
                    .timestamp_ns
                    .expect("timestamp queries are supported");
                assert!(gpu_ns > 0);
                assert!(
                    gpu_ns <= wall_ns,
                    "GPU {gpu_ns} ns above host wall {wall_ns} ns"
                );
            } else {
                assert_eq!(timings.timestamp_ns, None);
            }
        }
    }

    #[test]
    fn prepared_gpu_original_scores_match_cpu_for_large_oriented_image_when_available() {
        let Some(gpu) = gpu_or_skip(GpuWgpuBackend::new()) else {
            return;
        };

        let width = 3000;
        let height = 6000;
        let mut original = RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                original.put_pixel(
                    x,
                    y,
                    Rgba([
                        ((x * 13 + y * 3) % 256) as u8,
                        ((x * 5 + y * 11) % 256) as u8,
                        ((x * 17 + y * 7) % 256) as u8,
                        255,
                    ]),
                );
            }
        }

        let original_raw = DynamicImage::ImageRgba8(original);
        let prepared = gpu
            .prepare_original(&original_raw, Orientation::Rotate90)
            .unwrap();

        let mut oriented = original_raw.clone();
        oriented.apply_orientation(Orientation::Rotate90);
        let mut candidate_image = oriented.to_rgba8();
        for (x, y) in [(0, 0), (width - 1, 10), (height / 2, width / 2)] {
            let pixel = candidate_image.get_pixel_mut(x, y);
            pixel[0] = pixel[0].saturating_add(4);
            pixel[1] = pixel[1].saturating_sub(3);
            pixel[2] = pixel[2].saturating_add(2);
        }
        let candidate = DynamicImage::ImageRgba8(candidate_image);
        let cpu_score = CpuRayonBackend
            .score_candidate(&oriented, &candidate, false)
            .unwrap();
        let gpu_score = gpu
            .score_candidate_prepared_with_timings(&prepared, &candidate, false)
            .unwrap()
            .0;

        assert_close(cpu_score.rgb, gpu_score.rgb);
        assert_close(cpu_score.edge, gpu_score.edge);
        assert_close(cpu_score.overall, gpu_score.overall);
    }

    #[test]
    fn tiled_preprocess_matches_untiled_for_every_orientation_when_available() {
        let Some(gpu) = gpu_or_skip(GpuWgpuBackend::new()) else {
            return;
        };
        // Odd, narrow and wide sources; downsampled and same-size outputs.
        for ((width, height), scale) in [
            ((301_u32, 157_u32), 8_u32),
            ((301, 157), 1),
            ((5, 400), 3),
            ((400, 5), 3),
        ] {
            let source = RgbaImage::from_fn(width, height, |x, y| {
                Rgba([
                    ((x * 13 + y * 3) % 256) as u8,
                    ((x * 5 + y * 11) % 256) as u8,
                    ((x * 17 + y * 7) % 256) as u8,
                    ((x + y * 2) % 256) as u8,
                ])
            });
            let source_buffer = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: source.as_raw(),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            for code in 1..=8_u8 {
                let orientation = Orientation::from_exif(code).unwrap();
                let (oriented_width, oriented_height) =
                    oriented_dimensions(width, height, orientation);
                let output_width = oriented_width.div_ceil(scale);
                let output_height = oriented_height.div_ceil(scale);
                let untiled = gpu
                    .transform_to_buffer(
                        &source_buffer,
                        width,
                        height,
                        orientation,
                        output_width,
                        output_height,
                        "untiled",
                    )
                    .unwrap()
                    .0;
                // Room for the output level plus a few source rows per band.
                let limit = output_width as u64 * output_height as u64 * 4
                    + TRANSFORM_PARAMS_BYTES
                    + oriented_width as u64 * 4 * (scale as u64 + 2);
                let (tiled, band_pixels) = gpu
                    .transform_tiled(&source, orientation, output_width, output_height, limit)
                    .unwrap();
                assert!(band_pixels <= u64::from(width) * u64::from(height));
                // The CPU reduction is the same filter as the shader.
                let mut oriented = DynamicImage::ImageRgba8(source.clone());
                oriented.apply_orientation(orientation);
                assert_eq!(
                    box_downsample(&oriented.to_rgba8(), output_width, output_height),
                    gpu.readback_rgba(&untiled, output_width, output_height)
                        .unwrap(),
                    "CPU reduction: {width}x{height} scale {scale} orientation {code}"
                );
                assert_eq!(
                    gpu.readback_rgba(&untiled, output_width, output_height)
                        .unwrap(),
                    gpu.readback_rgba(&tiled, output_width, output_height)
                        .unwrap(),
                    "{width}x{height} scale {scale} orientation {code}"
                );
            }
        }
    }

    /// Content that punishes any filter or orientation disagreement: 1 px
    /// checkerboard, glyph-like strokes, a smooth gradient and a soft alpha edge.
    fn parity_content(kind: &str, width: u32, height: u32) -> RgbaImage {
        RgbaImage::from_fn(width, height, |x, y| match kind {
            "checkerboard" => {
                let value = if (x + y) % 2 == 0 { 255 } else { 0 };
                Rgba([value, value, value, 255])
            }
            "text" => {
                let stroke = (x % 7 < 2 && y % 11 < 8) || (y % 11 == 3 && x % 7 < 5);
                let value = if stroke { 20 } else { 240 };
                Rgba([value, value, value, 255])
            }
            "gradient" => Rgba([
                (x * 255 / width.max(2).saturating_sub(1)) as u8,
                (y * 255 / height.max(2).saturating_sub(1)) as u8,
                ((x + y) * 255 / (width + height)) as u8,
                255,
            ]),
            _ => {
                let distance = x.abs_diff(width / 2) + y.abs_diff(height / 2);
                let alpha = 255_u32.saturating_sub(distance * 8).min(255) as u8;
                Rgba([200, 60, (x * 3 % 256) as u8, alpha])
            }
        })
    }

    fn perturb(image: &mut RgbaImage) {
        let (width, height) = image.dimensions();
        for (index, pixel) in image.pixels_mut().enumerate() {
            if index % 5 == 0 {
                pixel[0] = pixel[0].saturating_add(6);
                pixel[1] = pixel[1].saturating_sub(4);
                pixel[3] = pixel[3].saturating_sub(pixel[3].min(3));
            }
        }
        let corner = image.get_pixel_mut(width - 1, height - 1);
        corner[2] = corner[2].wrapping_add(40);
    }

    #[test]
    fn prepared_gpu_scores_match_cpu_for_adversarial_content_in_every_orientation() {
        let Some(gpu) = gpu_or_skip(GpuWgpuBackend::new()) else {
            return;
        };
        for kind in ["checkerboard", "text", "gradient", "alpha-edge"] {
            let has_alpha = kind == "alpha-edge";
            let raw = DynamicImage::ImageRgba8(parity_content(kind, 97, 61));
            for code in 1..=8_u8 {
                let orientation = Orientation::from_exif(code).unwrap();
                let prepared = gpu.prepare_original(&raw, orientation).unwrap();
                let mut oriented = raw.clone();
                oriented.apply_orientation(orientation);
                let mut candidate = oriented.to_rgba8();
                perturb(&mut candidate);
                let candidate = DynamicImage::ImageRgba8(candidate);
                let cpu = CpuRayonBackend
                    .score_candidate(&oriented, &candidate, has_alpha)
                    .unwrap();
                let prepared_score = gpu
                    .score_candidate_prepared_with_timings(&prepared, &candidate, has_alpha)
                    .unwrap()
                    .0;
                let direct_score = gpu
                    .score_candidate(&oriented, &candidate, has_alpha)
                    .unwrap();
                for (label, score) in [("prepared", prepared_score), ("direct", direct_score)] {
                    for (component, expected, actual) in [
                        ("rgb", Some(cpu.rgb), Some(score.rgb)),
                        ("edge", Some(cpu.edge), Some(score.edge)),
                        ("alpha", cpu.alpha, score.alpha),
                        ("composite", cpu.composite_min, score.composite_min),
                        ("overall", Some(cpu.overall), Some(score.overall)),
                    ] {
                        assert_eq!(
                            expected.is_some(),
                            actual.is_some(),
                            "{kind} orientation {code} {label} {component}"
                        );
                        if let (Some(expected), Some(actual)) = (expected, actual) {
                            assert!(
                                (expected - actual).abs() <= 0.01,
                                "{kind} orientation {code} {label} {component}: cpu={expected} gpu={actual}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn colour_under_fully_transparent_pixels_is_free_on_every_backend() {
        let raw = DynamicImage::ImageRgba8(parity_content("alpha-edge", 97, 61));
        let mut hidden_colour = raw.to_rgba8();
        let mut changed = 0;
        for pixel in hidden_colour.pixels_mut().filter(|pixel| pixel[3] == 0) {
            pixel.0 = [255 - pixel[0], 17, 99, 0];
            changed += 1;
        }
        assert!(changed > 100, "the fixture needs hidden pixels");
        let candidate = DynamicImage::ImageRgba8(hidden_colour);

        let mut scores = vec![
            CpuRayonBackend
                .score_candidate(&raw, &candidate, true)
                .unwrap(),
        ];
        if let Some(gpu) = gpu_or_skip(GpuWgpuBackend::new()) {
            scores.push(gpu.score_candidate(&raw, &candidate, true).unwrap());
            let prepared = gpu
                .prepare_original(&raw, Orientation::NoTransforms)
                .unwrap();
            scores.push(
                gpu.score_candidate_prepared_with_timings(&prepared, &candidate, true)
                    .unwrap()
                    .0,
            );
        }
        for score in scores {
            assert_eq!(
                (score.rgb, score.edge, score.overall),
                (100.0, 100.0, 100.0),
                "{score:?}"
            );
        }
    }

    #[test]
    fn reduced_gpu_scores_match_cpu_for_a_large_checkerboard() {
        let Some(gpu) = gpu_or_skip(GpuWgpuBackend::new()) else {
            return;
        };
        // Just over 16 MP, so every backend scores a reduced level; a 1 px
        // checkerboard exposes any difference between their filters.
        let (width, height) = (4104, 4099);
        let raw = DynamicImage::ImageRgba8(parity_content("checkerboard", width, height));
        let prepared = gpu.prepare_original(&raw, Orientation::Rotate270).unwrap();
        let mut oriented = raw.clone();
        oriented.apply_orientation(Orientation::Rotate270);
        let mut candidate = oriented.to_rgba8();
        perturb(&mut candidate);
        let candidate = DynamicImage::ImageRgba8(candidate);
        let cpu = CpuRayonBackend
            .score_candidate(&oriented, &candidate, false)
            .unwrap();
        for score in [
            gpu.score_candidate_prepared_with_timings(&prepared, &candidate, false)
                .unwrap()
                .0,
            gpu.score_candidate(&oriented, &candidate, false).unwrap(),
        ] {
            assert!((cpu.rgb - score.rgb).abs() <= 0.01, "{cpu:?} {score:?}");
            assert!((cpu.edge - score.edge).abs() <= 0.01, "{cpu:?} {score:?}");
            assert!(
                (cpu.overall - score.overall).abs() <= 0.01,
                "{cpu:?} {score:?}"
            );
        }
    }

    fn assert_close(cpu: f32, gpu: f32) {
        let delta = (cpu - gpu).abs();
        assert!(
            delta <= 0.01,
            "CPU score {cpu:.6} and GPU score {gpu:.6} differ by {delta:.6}"
        );
    }
}
