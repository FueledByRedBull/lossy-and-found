use std::{sync::mpsc, time::Instant};

use image::{DynamicImage, GenericImageView, RgbaImage, metadata::Orientation};
use smart_compress_metrics::{
    MetricError, QualityScore, alpha_similarity, composite_score, edge_similarity, rgb_similarity,
};
use wgpu::util::DeviceExt;

use super::{
    ComputeBackend, GPU_METRIC_COMPONENTS, GPU_WORKGROUP_SIZE, GpuDispatchParams, GpuMetricScores,
    GpuMetricSums, GpuPassTimer, GpuPreparedOriginal, GpuResidentLevel, GpuScoreOutcome,
    GpuTimingBreakdown, GpuWgpuBackend, add_optional_timing, box_downsample,
    cpu::{OriginalImage, check_dimensions, metric_level},
    device::bind_entry,
    dimensions::bounded_metric_dimensions,
};

impl GpuWgpuBackend {
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
