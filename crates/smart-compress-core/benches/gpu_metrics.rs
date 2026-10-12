use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use image::{DynamicImage, Rgba, RgbaImage};
use smart_compress_core::scoring::{ComputeBackend, CpuRayonBackend, GpuWgpuBackend};

fn bench_cpu_and_gpu_scoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("score_candidate");
    let gpu = GpuWgpuBackend::new().ok();

    for side in [512_u32, 2048] {
        let original = fixture_image(side);
        let candidate = perturbed_image(&original);

        group.bench_with_input(BenchmarkId::new("cpu-rayon", side), &side, |b, _| {
            b.iter(|| {
                CpuRayonBackend
                    .score_candidate(black_box(&original), black_box(&candidate), true)
                    .unwrap()
            })
        });

        if let Some(gpu) = &gpu {
            group.bench_with_input(BenchmarkId::new("gpu-wgpu", side), &side, |b, _| {
                b.iter(|| {
                    gpu.score_candidate(black_box(&original), black_box(&candidate), true)
                        .unwrap()
                })
            });
        }
    }

    group.finish();
}

fn fixture_image(side: u32) -> DynamicImage {
    let mut image = RgbaImage::new(side, side);
    for y in 0..side {
        for x in 0..side {
            image.put_pixel(
                x,
                y,
                Rgba([
                    ((x * 13 + y * 3) % 256) as u8,
                    ((x * 5 + y * 11) % 256) as u8,
                    ((x * 7 + y * 17) % 256) as u8,
                    ((x + y) % 256) as u8,
                ]),
            );
        }
    }
    DynamicImage::ImageRgba8(image)
}

fn perturbed_image(original: &DynamicImage) -> DynamicImage {
    let mut image = original.to_rgba8();
    for y in (0..image.height()).step_by(17) {
        for x in (0..image.width()).step_by(19) {
            let pixel = image.get_pixel_mut(x, y);
            pixel[0] = pixel[0].saturating_add(2);
            pixel[1] = pixel[1].saturating_sub(1);
            pixel[2] = pixel[2].saturating_add(3);
            pixel[3] = pixel[3].saturating_sub(1);
        }
    }
    DynamicImage::ImageRgba8(image)
}

criterion_group!(benches, bench_cpu_and_gpu_scoring);
criterion_main!(benches);
