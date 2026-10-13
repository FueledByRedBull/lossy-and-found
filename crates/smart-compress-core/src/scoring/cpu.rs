use std::{ops::Deref, sync::OnceLock};

use fast_ssim2::Ssimulacra2Reference;
use image::{DynamicImage, GenericImageView, RgbaImage};
use smart_compress_metrics::{MetricError, QualityScore, score_rgba};

use super::{ComputeBackend, dimensions::bounded_metric_dimensions};

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
/// instead of being rebuilt, at full size, for each of them. The in-process
/// SSIMULACRA2 reference is built once, before the candidates, and kept here.
pub struct OriginalImage {
    image: DynamicImage,
    full: OnceLock<RgbaImage>,
    metric: OnceLock<RgbaImage>,
    ssimulacra2: OnceLock<Ssimulacra2Reference>,
}

impl From<DynamicImage> for OriginalImage {
    fn from(image: DynamicImage) -> Self {
        Self {
            image,
            full: OnceLock::new(),
            metric: OnceLock::new(),
            ssimulacra2: OnceLock::new(),
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

    pub fn ssimulacra2_reference(&self) -> Option<&Ssimulacra2Reference> {
        self.ssimulacra2.get()
    }

    pub fn set_ssimulacra2_reference(&self, reference: Ssimulacra2Reference) {
        let _ = self.ssimulacra2.set(reference);
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

pub(super) fn check_dimensions(
    original: &DynamicImage,
    candidate: &DynamicImage,
) -> Result<(), MetricError> {
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
pub(super) fn metric_level(image: &DynamicImage, original_dimensions: (u32, u32)) -> RgbaImage {
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
