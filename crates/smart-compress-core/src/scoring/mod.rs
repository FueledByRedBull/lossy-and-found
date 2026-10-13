use std::sync::mpsc;

use image::RgbaImage;
use smart_compress_metrics::QualityScore;

use crate::gpu::GpuAdapterReport;

mod cpu;
mod device;
mod dimensions;
mod prepare;
mod score;
mod shaders;
#[cfg(test)]
mod tests;

pub(crate) use cpu::box_downsample;
pub use cpu::{CpuRayonBackend, OriginalImage};
pub use dimensions::FULL_SIZE_METRIC_PIXELS;
pub(crate) use dimensions::bounded_metric_dimensions;

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
}

#[derive(Debug, Clone, Copy)]
struct GpuMetricScores {
    rgb: f32,
    alpha: f32,
    composite_min: f32,
    edge: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct GpuMetricSums {
    sums: [u64; GPU_METRIC_COMPONENTS],
    timings: GpuTimingBreakdown,
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

    pub fn backend_id(&self) -> &'static str {
        match self {
            Self::Cpu(cpu) => cpu.backend_id(),
            Self::Gpu(gpu) => gpu.backend_id(),
        }
    }
}

/// Sums a dispatch returns: raw RGB, alpha, five composites, edge, then the
/// pixels and the edge samples that are visible in at least one image.
const GPU_METRIC_COMPONENTS: usize = 10;
const GPU_WORKGROUP_SIZE: u64 = 256;
