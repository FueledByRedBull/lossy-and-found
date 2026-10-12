use image::{DynamicImage, GenericImageView, RgbaImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetricError {
    #[error("image dimensions differ: original={original:?}, candidate={candidate:?}")]
    DimensionMismatch {
        original: (u32, u32),
        candidate: (u32, u32),
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct QualityScore {
    /// Raw RGB agreement: twice the PSNR over the colour channels, capped at
    /// 100. Older reports and cache entries call it `ssim`.
    #[serde(alias = "ssim")]
    pub rgb: f32,
    #[serde(default)]
    pub ssimulacra2: Option<f32>,
    #[serde(default)]
    pub butteraugli: Option<f32>,
    #[serde(default)]
    pub alpha: Option<f32>,
    #[serde(default)]
    pub composite_min: Option<f32>,
    /// Worst local agreement around transparency edges; see
    /// `alpha_edge_similarity`.
    #[serde(default)]
    pub alpha_edge: Option<f32>,
    pub edge: f32,
    pub overall: f32,
}

impl QualityScore {
    pub fn lossless() -> Self {
        Self {
            rgb: 100.0,
            ssimulacra2: Some(100.0),
            butteraugli: Some(0.0),
            alpha: Some(100.0),
            composite_min: Some(100.0),
            alpha_edge: Some(100.0),
            edge: 100.0,
            overall: 100.0,
        }
    }
}

pub fn score_images(
    original: &DynamicImage,
    candidate: &DynamicImage,
    has_alpha: bool,
) -> Result<QualityScore, MetricError> {
    if original.dimensions() != candidate.dimensions() {
        return Err(MetricError::DimensionMismatch {
            original: original.dimensions(),
            candidate: candidate.dimensions(),
        });
    }

    score_rgba(&original.to_rgba8(), &candidate.to_rgba8(), has_alpha)
}

/// [`score_images`] for callers that already hold the RGBA buffers, such as
/// one converted original scored against many candidates.
pub fn score_rgba(
    original_rgba: &RgbaImage,
    candidate_rgba: &RgbaImage,
    has_alpha: bool,
) -> Result<QualityScore, MetricError> {
    if original_rgba.dimensions() != candidate_rgba.dimensions() {
        return Err(MetricError::DimensionMismatch {
            original: original_rgba.dimensions(),
            candidate: candidate_rgba.dimensions(),
        });
    }

    let rgb = rgb_similarity(original_rgba, candidate_rgba);
    let edge = edge_similarity(original_rgba, candidate_rgba);
    let alpha = has_alpha.then(|| alpha_similarity(original_rgba, candidate_rgba));
    let composite_min = has_alpha.then(|| composite_score(original_rgba, candidate_rgba));
    let overall = [Some(rgb), Some(edge), alpha, composite_min]
        .into_iter()
        .flatten()
        .fold(100.0_f32, f32::min);

    Ok(QualityScore {
        rgb,
        ssimulacra2: None,
        butteraugli: None,
        alpha,
        composite_min,
        alpha_edge: None,
        edge,
        overall,
    })
}

/// Worst local agreement around transparency edges, on the same scale as the
/// other built-in scores (twice the PSNR, capped at 100).
///
/// A fringe or halo touches few pixels, so whole-image scores barely move.
/// This looks only at the tiles where the original's alpha varies and reports
/// the worst one. Each tile compares the original colour drawn with the
/// original alpha against the same colour drawn with the candidate's alpha,
/// over white, black and both phases of a checkerboard, so it measures what
/// the alpha change does on screen and nothing else: colour fidelity is the
/// main metric's job, and pixels that are fully transparent in both images
/// cost nothing whatever colour they hold.
pub fn alpha_edge_similarity(a: &RgbaImage, b: &RgbaImage) -> f32 {
    const TILE: usize = 16;
    const CHECK: usize = 8;
    let (width, height) = (a.width() as usize, a.height() as usize);
    if (a.width(), a.height()) != (b.width(), b.height()) || width == 0 || height == 0 {
        return 0.0;
    }
    let (raw_a, raw_b) = (a.as_raw(), b.as_raw());
    let worst_sse_per_sample = (0..height.div_ceil(TILE))
        .into_par_iter()
        .map(|tile_row| {
            let mut worst = 0.0_f64;
            for tile_column in 0..width.div_ceil(TILE) {
                let (x0, y0) = (tile_column * TILE, tile_row * TILE);
                let (x1, y1) = ((x0 + TILE).min(width), (y0 + TILE).min(height));
                let (mut low, mut high) = (u8::MAX, u8::MIN);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let alpha = raw_a[(y * width + x) * 4 + 3];
                        low = low.min(alpha);
                        high = high.max(alpha);
                    }
                }
                if low == high {
                    continue;
                }
                // White, black, checkerboard, inverse checkerboard.
                let mut sse = [0.0_f64; 4];
                for y in y0..y1 {
                    for x in x0..x1 {
                        let index = (y * width + x) * 4;
                        let delta = f64::from(raw_a[index + 3]) - f64::from(raw_b[index + 3]);
                        if delta == 0.0 {
                            continue;
                        }
                        let light = (x / CHECK + y / CHECK).is_multiple_of(2);
                        let backgrounds = [
                            255.0,
                            0.0,
                            if light { 255.0 } else { 0.0 },
                            if light { 0.0 } else { 255.0 },
                        ];
                        for channel in 0..3 {
                            let colour = f64::from(raw_a[index + channel]);
                            for (sum, background) in sse.iter_mut().zip(backgrounds) {
                                let difference = (colour - background) * delta / 255.0;
                                *sum += difference * difference;
                            }
                        }
                    }
                }
                let samples = ((x1 - x0) * (y1 - y0) * 3) as f64;
                worst = worst.max(sse.into_iter().fold(0.0, f64::max) / samples);
            }
            worst
        })
        .reduce(|| 0.0, f64::max);
    if worst_sse_per_sample <= f64::EPSILON {
        return 100.0;
    }
    let psnr = 10.0 * ((255.0_f64 * 255.0_f64) / worst_sse_per_sample).log10();
    (psnr * 2.0).clamp(0.0, 100.0) as f32
}

/// Whether a pixel is fully transparent in both images. Its colour is
/// invisible, so the raw RGB and edge scores neither charge for it nor count it.
fn hidden(pa: &[u8], pb: &[u8]) -> bool {
    pa[3] == 0 && pb[3] == 0
}

pub fn rgb_similarity(a: &RgbaImage, b: &RgbaImage) -> f32 {
    scaled_similarity(a, b, Channels::Rgb)
}

pub fn alpha_similarity(a: &RgbaImage, b: &RgbaImage) -> f32 {
    scaled_similarity(a, b, Channels::Alpha)
}

pub fn edge_similarity(a: &RgbaImage, b: &RgbaImage) -> f32 {
    let (width, height) = a.dimensions();
    if width < 3 || height < 3 {
        return scaled_similarity(a, b, Channels::Rgb);
    }

    // One luma plane per image, so each pixel is converted once instead of
    // once per neighbour. Squared differences are whole numbers, so the
    // integer sum equals the exact value the floating-point one held. A pixel
    // that is fully transparent in both images has luma 0 in both and is not
    // a sample.
    let (luma_a, luma_b) = (luma_plane(a, b), luma_plane(b, a));
    let row_width = width as usize;
    let (raw_a, raw_b) = (a.as_raw(), b.as_raw());
    let (sum, count) = (1..height as usize - 1)
        .into_par_iter()
        .map(|y| {
            let mut sum = 0_u64;
            let mut count = 0_u64;
            let edges = edge_row(&luma_a, row_width, y).zip(edge_row(&luma_b, row_width, y));
            for (x, (ea, eb)) in (1..row_width - 1).zip(edges) {
                let at = (y * row_width + x) * 4;
                if !hidden(&raw_a[at..at + 4], &raw_b[at..at + 4]) {
                    sum += u64::from(ea.abs_diff(eb)).pow(2);
                    count += 1;
                }
            }
            (sum, count)
        })
        .reduce(
            || (0, 0),
            |left, right| (left.0 + right.0, left.1 + right.1),
        );
    let (sum, count) = (sum as f64, count);
    if count == 0 {
        return 100.0;
    }

    let mse = sum / count as f64;
    if mse <= f64::EPSILON {
        return 100.0;
    }
    let psnr = 10.0 * ((255.0 * 255.0) / mse).log10();
    (psnr * 2.0).clamp(0.0, 100.0) as f32
}

pub fn composite_score(a: &RgbaImage, b: &RgbaImage) -> f32 {
    let backgrounds = [
        [255_u8, 255, 255],
        [0, 0, 0],
        [128, 128, 128],
        [238, 238, 238],
        [64, 64, 64],
    ];

    backgrounds
        .iter()
        .map(|background| composite_similarity(a, b, *background))
        .fold(100.0_f32, f32::min)
}

#[derive(Clone, Copy)]
enum Channels {
    Rgb,
    Alpha,
}

fn scaled_similarity(a: &RgbaImage, b: &RgbaImage, channels: Channels) -> f32 {
    let (sum, count) = a
        .as_raw()
        .par_chunks_exact(4)
        .zip(b.as_raw().par_chunks_exact(4))
        .map(|(pa, pb)| match channels {
            Channels::Rgb if hidden(pa, pb) => (0.0, 0_u64),
            Channels::Rgb => {
                let dr = pa[0] as f64 - pb[0] as f64;
                let dg = pa[1] as f64 - pb[1] as f64;
                let db = pa[2] as f64 - pb[2] as f64;
                (dr * dr + dg * dg + db * db, 3_u64)
            }
            Channels::Alpha => {
                let da = pa[3] as f64 - pb[3] as f64;
                (da * da, 1_u64)
            }
        })
        .reduce(
            || (0.0, 0),
            |left, right| (left.0 + right.0, left.1 + right.1),
        );

    if count == 0 {
        return 100.0;
    }

    let mse = sum / count as f64;
    if mse <= f64::EPSILON {
        return 100.0;
    }
    let psnr = 10.0 * ((255.0 * 255.0) / mse).log10();
    (psnr * 2.0).clamp(0.0, 100.0) as f32
}

fn composite_similarity(a: &RgbaImage, b: &RgbaImage, background: [u8; 3]) -> f32 {
    let (sum, count) = a
        .as_raw()
        .par_chunks_exact(4)
        .zip(b.as_raw().par_chunks_exact(4))
        .map(|(pa, pb)| {
            let ca = composite_rgba(pa, background);
            let cb = composite_rgba(pb, background);
            let dr = ca[0] as f64 - cb[0] as f64;
            let dg = ca[1] as f64 - cb[1] as f64;
            let db = ca[2] as f64 - cb[2] as f64;
            (dr * dr + dg * dg + db * db, 3_u64)
        })
        .reduce(
            || (0.0, 0),
            |left, right| (left.0 + right.0, left.1 + right.1),
        );

    if count == 0 {
        return 100.0;
    }

    let mse = sum / count as f64;
    if mse <= f64::EPSILON {
        return 100.0;
    }
    let psnr: f64 = 10.0 * ((255.0_f64 * 255.0_f64) / mse).log10();
    (psnr * 2.0).clamp(0.0, 100.0) as f32
}

fn composite_rgba(rgba: &[u8], background: [u8; 3]) -> [u8; 3] {
    let alpha = rgba[3] as f32 / 255.0;
    [
        blend(rgba[0], background[0], alpha),
        blend(rgba[1], background[1], alpha),
        blend(rgba[2], background[2], alpha),
    ]
}

fn blend(foreground: u8, background: u8, alpha: f32) -> u8 {
    ((foreground as f32 * alpha) + (background as f32 * (1.0 - alpha))).round() as u8
}

/// Luma of every pixel of `image`, or 0 where it is fully transparent in
/// `other` as well.
fn luma_plane(image: &RgbaImage, other: &RgbaImage) -> Vec<u8> {
    image
        .as_raw()
        .par_chunks_exact(4)
        .zip(other.as_raw().par_chunks_exact(4))
        .map(|(pixel, other)| {
            if hidden(pixel, other) {
                return 0;
            }
            (0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32).round()
                as u8
        })
        .collect()
}

/// Gradient magnitude of the inner pixels of row `y` of a luma plane.
fn edge_row(luma: &[u8], width: usize, y: usize) -> impl Iterator<Item = u16> + '_ {
    let above = &luma[(y - 1) * width..y * width];
    let row = &luma[y * width..(y + 1) * width];
    let below = &luma[(y + 1) * width..(y + 2) * width];
    (1..width - 1)
        .map(move |x| row[x - 1].abs_diff(row[x + 1]) as u16 + above[x].abs_diff(below[x]) as u16)
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgba, RgbaImage};

    use super::*;

    #[test]
    fn identical_images_score_100() {
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, Rgba([10, 20, 30, 255])));
        let score = score_images(&image, &image, false).unwrap();
        assert_eq!(score.overall, 100.0);
    }

    /// Opaque disc on a transparent field with a soft four-pixel edge.
    fn disc(size: u32, radius: f32) -> RgbaImage {
        RgbaImage::from_fn(size, size, |x, y| {
            let centre = size as f32 / 2.0;
            let distance = ((x as f32 - centre).powi(2) + (y as f32 - centre).powi(2)).sqrt();
            let alpha = ((radius - distance) / 4.0).clamp(0.0, 1.0) * 255.0;
            Rgba([30, 60, 200, alpha as u8])
        })
    }

    #[test]
    fn alpha_edge_gate_catches_a_fringe_the_global_score_hides() {
        let original = disc(1024, 24.0);
        // The same disc one pixel larger: a thin ring of changed alpha.
        let fringed = disc(1024, 25.0);
        let global = alpha_similarity(&original, &fringed);
        let local = alpha_edge_similarity(&original, &fringed);
        assert!(global > 80.0, "global alpha score {global}, local {local}");
        assert!(local < 70.0, "local edge score {local}");
        assert_eq!(alpha_edge_similarity(&original, &original), 100.0);
    }

    #[test]
    fn alpha_edge_gate_ignores_colour_and_hidden_pixels() {
        let original = disc(128, 30.0);
        // Straight-alpha encoders may store anything under fully transparent
        // pixels, and lossy colour is judged by the main metric.
        let mut candidate = original.clone();
        for pixel in candidate.pixels_mut() {
            if pixel[3] == 0 {
                pixel.0 = [255, 0, 255, 0];
            } else {
                pixel[0] = pixel[0].saturating_add(9);
            }
        }
        assert_eq!(alpha_edge_similarity(&original, &candidate), 100.0);

        // An image with no transparency edge has nothing to check.
        let opaque = RgbaImage::from_pixel(64, 64, Rgba([1, 2, 3, 255]));
        let mut noisy = opaque.clone();
        noisy.get_pixel_mut(5, 5)[3] = 200;
        assert_eq!(alpha_edge_similarity(&opaque, &noisy), 100.0);
        assert!(alpha_edge_similarity(&noisy, &opaque) < 100.0);
    }

    /// The per-pixel formulation the luma-plane version replaced.
    fn edge_similarity_per_pixel(a: &RgbaImage, b: &RgbaImage) -> f32 {
        let hidden = |x: u32, y: u32| a.get_pixel(x, y)[3] == 0 && b.get_pixel(x, y)[3] == 0;
        let luma = |image: &RgbaImage, x: u32, y: u32| {
            if hidden(x, y) {
                return 0;
            }
            let p = image.get_pixel(x, y);
            (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32).round() as u8
        };
        let edge = |image: &RgbaImage, x: u32, y: u32| {
            luma(image, x - 1, y).abs_diff(luma(image, x + 1, y)) as u16
                + luma(image, x, y - 1).abs_diff(luma(image, x, y + 1)) as u16
        };
        let (width, height) = a.dimensions();
        let mut sum = 0.0_f64;
        let mut count = 0_u64;
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                if hidden(x, y) {
                    continue;
                }
                let diff = edge(a, x, y) as f64 - edge(b, x, y) as f64;
                sum += diff * diff;
                count += 1;
            }
        }
        if count == 0 {
            return 100.0;
        }
        let mse = sum / count as f64;
        if mse <= f64::EPSILON {
            return 100.0;
        }
        ((10.0 * ((255.0 * 255.0) / mse).log10()) * 2.0).clamp(0.0, 100.0) as f32
    }

    #[test]
    fn edge_similarity_matches_the_per_pixel_formulation() {
        let mut state = 12345_u32;
        let mut noise = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        for (width, height) in [(3, 3), (7, 5), (64, 33), (301, 157)] {
            let a = RgbaImage::from_fn(width, height, |_, _| {
                Rgba([noise(), noise(), noise(), noise()])
            });
            let b = RgbaImage::from_fn(width, height, |x, y| {
                let p = a.get_pixel(x, y);
                Rgba([
                    p[0] ^ (noise() & 7),
                    p[1],
                    p[2].wrapping_add(noise() & 3),
                    p[3],
                ])
            });
            assert_eq!(
                edge_similarity(&a, &b),
                edge_similarity_per_pixel(&a, &b),
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn colour_under_fully_transparent_pixels_costs_nothing() {
        let original = disc(96, 30.0);
        let mut noisy = original.clone();
        let mut state = 7_u32;
        for pixel in noisy.pixels_mut().filter(|pixel| pixel[3] == 0) {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            pixel.0 = [
                (state >> 24) as u8,
                (state >> 16) as u8,
                (state >> 8) as u8,
                0,
            ];
        }
        assert_ne!(original, noisy);
        assert_eq!(rgb_similarity(&original, &noisy), 100.0);
        assert_eq!(edge_similarity(&original, &noisy), 100.0);
        let score = score_rgba(&original, &noisy, true).unwrap();
        assert_eq!(score.overall, 100.0);

        // Visible colour still counts, and so does colour that becomes visible.
        let mut visible = original.clone();
        visible
            .pixels_mut()
            .filter(|pixel| pixel[3] != 0)
            .for_each(|pixel| pixel[0] ^= 0x40);
        assert!(rgb_similarity(&original, &visible) < 100.0);
        let mut revealed = noisy.clone();
        revealed
            .pixels_mut()
            .filter(|pixel| pixel[3] == 0)
            .for_each(|pixel| pixel[3] = 1);
        assert!(rgb_similarity(&original, &revealed) < 100.0);
        // Nothing visible in either image: nothing to disagree about.
        let blank = RgbaImage::from_pixel(8, 8, Rgba([1, 2, 3, 0]));
        let other = RgbaImage::from_pixel(8, 8, Rgba([200, 100, 50, 0]));
        assert_eq!(rgb_similarity(&blank, &other), 100.0);
        assert_eq!(edge_similarity(&blank, &other), 100.0);
    }

    #[test]
    fn different_images_score_lower() {
        let a = DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, Rgba([10, 20, 30, 255])));
        let b = DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, Rgba([200, 20, 30, 255])));
        let score = score_images(&a, &b, false).unwrap();
        assert!(score.overall < 100.0);
    }
}
