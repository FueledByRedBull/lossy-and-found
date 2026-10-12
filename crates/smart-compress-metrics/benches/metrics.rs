use criterion::{Criterion, criterion_group, criterion_main};
use image::{DynamicImage, Rgba, RgbaImage};
use smart_compress_metrics::score_images;

fn bench_score_identical(c: &mut Criterion) {
    let image =
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(512, 512, Rgba([64, 128, 192, 255])));
    c.bench_function("score_identical_512", |b| {
        b.iter(|| score_images(&image, &image, false).unwrap())
    });
}

criterion_group!(benches, bench_score_identical);
criterion_main!(benches);
