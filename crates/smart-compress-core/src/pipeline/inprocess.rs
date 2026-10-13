//! SSIMULACRA2 computed in this process, against a reference kept for the life
//! of an image, on the pixels the external tool reads from the metric PNG.

use anyhow::{Context, Result, anyhow};
use fast_ssim2::{PixelDescriptor, PixelSlice, Ssimulacra2Reference};
use image::{DynamicImage, GenericImageView};

use super::score::bounded_metric_image;

/// Names the scorer in the quality-cache signature. The version is the one
/// pinned in the workspace manifest; change them together.
pub(super) const SIGNATURE: &str = "inprocess-fast-ssim2-0.9.0";

/// The samples of an image as the metric PNG would hold them: 8 or 16 bits a
/// channel, with an alpha channel when the image has one.
struct MetricPixels {
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
    descriptor: PixelDescriptor,
}

impl MetricPixels {
    fn new(image: &DynamicImage) -> Self {
        let image = bounded_metric_image(image);
        let (width, height) = image.dimensions();
        let color = image.color();
        let sixteen = color.bytes_per_pixel() > color.channel_count();
        let wide = |samples: Vec<u16>| samples.into_iter().flat_map(u16::to_ne_bytes).collect();
        let (bytes, bytes_per_pixel, descriptor) = match (color.has_alpha(), sixteen) {
            (false, false) => (image.to_rgb8().into_raw(), 3, PixelDescriptor::RGB8_SRGB),
            (true, false) => (image.to_rgba8().into_raw(), 4, PixelDescriptor::RGBA8_SRGB),
            (false, true) => (
                wide(image.to_rgb16().into_raw()),
                6,
                PixelDescriptor::RGB16_SRGB,
            ),
            (true, true) => (
                wide(image.to_rgba16().into_raw()),
                8,
                PixelDescriptor::RGBA16_SRGB,
            ),
        };
        Self {
            bytes,
            width,
            height,
            bytes_per_pixel,
            descriptor,
        }
    }

    fn slice(&self) -> Result<PixelSlice<'_>> {
        PixelSlice::new(
            &self.bytes,
            self.width,
            self.height,
            self.width as usize * self.bytes_per_pixel,
            self.descriptor,
        )
        .map_err(|error| anyhow!("SSIMULACRA2 input: {error}"))
    }
}

pub(super) fn build_reference(original: &DynamicImage) -> Result<Ssimulacra2Reference> {
    Ssimulacra2Reference::new(&MetricPixels::new(original).slice()?)
        .context("building the SSIMULACRA2 reference")
}

pub(super) fn score(reference: &Ssimulacra2Reference, candidate: &DynamicImage) -> Result<f32> {
    reference
        .compare(&MetricPixels::new(candidate).slice()?)
        .map(|score| score as f32)
        .context("scoring with SSIMULACRA2")
}

#[cfg(test)]
mod tests {
    use image::{Luma, Rgb, Rgba};

    use super::*;

    #[test]
    fn the_metric_pixels_keep_the_layout_the_metric_png_would_have() {
        let layout = |image: DynamicImage| {
            let pixels = MetricPixels::new(&image);
            (pixels.bytes_per_pixel, pixels.bytes.len())
        };
        let grey = DynamicImage::ImageLuma8(image::GrayImage::from_pixel(8, 8, Luma([9])));
        let rgb16 = DynamicImage::ImageRgb16(image::ImageBuffer::from_pixel(8, 8, Rgb([9; 3])));
        let rgba = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(8, 8, Rgba([9; 4])));
        assert_eq!(layout(grey), (3, 8 * 8 * 3));
        assert_eq!(layout(rgb16), (6, 8 * 8 * 6));
        assert_eq!(layout(rgba), (4, 8 * 8 * 4));
    }

    #[test]
    fn a_large_image_is_scored_on_the_mosaic_the_external_tool_gets() {
        let large = DynamicImage::ImageRgb8(image::RgbImage::new(5000, 4000));
        let pixels = MetricPixels::new(&large);
        assert_eq!((pixels.width, pixels.height), (3072, 3072));
        assert_eq!(pixels.bytes_per_pixel, 3);
    }
}
