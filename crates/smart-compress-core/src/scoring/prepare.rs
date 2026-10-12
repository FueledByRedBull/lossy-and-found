use std::{borrow::Cow, sync::mpsc};

use image::{DynamicImage, RgbaImage, metadata::Orientation};
use wgpu::util::DeviceExt;

use super::{
    GpuPassTimer, GpuPreparedOriginal, GpuResidentLevel, GpuWgpuBackend, TRANSFORM_PARAMS_BYTES,
    TransformParams, box_downsample,
    device::bind_entry,
    dimensions::{bounded_metric_dimensions, bounded_preview_dimensions, oriented_dimensions},
};

impl GpuWgpuBackend {
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
    pub(super) fn transform_tiled(
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

    pub(super) fn upload_resident_level(
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_level(
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn transform_to_buffer(
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

    pub(super) fn readback_rgba(
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
