use std::time::Instant;

use image::{DynamicImage, Rgba, RgbaImage, metadata::Orientation};
use wgpu::util::DeviceExt;

use super::dimensions::oriented_dimensions;
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
    let original = DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, Rgba([10, 20, 30, 255])));
    let candidate = DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, Rgba([11, 20, 30, 255])));

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
    let original = DynamicImage::ImageRgba8(RgbaImage::from_fn(1024, 1024, |x, y| pixel(x, y, 0)));
    let candidate = DynamicImage::ImageRgba8(RgbaImage::from_fn(1024, 1024, |x, y| pixel(x, y, 2)));
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
            let (oriented_width, oriented_height) = oriented_dimensions(width, height, orientation);
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
