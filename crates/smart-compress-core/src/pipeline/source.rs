use std::{ffi::OsString, fs, path::Path};

use anyhow::{Context, Result, anyhow};
use smart_compress_codecs::run_process;

use super::{Optimizer, work};
use crate::{
    candidate_probe::{probe_avif, read_prefix},
    config::AlphaPolicy,
    image_probe::{
        DecodedSource, ImageDecodeLimits, decode_source_with_limits, header_dimensions,
        validate_decoder_dimensions,
    },
    scoring::FULL_SIZE_METRIC_PIXELS,
};

impl Optimizer {
    pub(super) fn should_use_gpu_preprocess(&self, image: &image::DynamicImage) -> bool {
        self.config.gpu.enabled
            && self.config.gpu.prefer_gpu_preprocess
            && self.compute.as_ref().as_gpu().is_some()
            && image.width() as u64 * image.height() as u64 > FULL_SIZE_METRIC_PIXELS
    }

    /// Pixels in the source from its header alone, so memory can be reserved
    /// before the image is decoded. An unreadable header falls back to a guess
    /// from the file size; the decode will report what is wrong with the file.
    pub(super) fn source_pixels(&self, input: &Path, file_bytes: u64) -> u64 {
        let from_header = if input
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("avif"))
        {
            read_prefix(input)
                .and_then(|header| probe_avif(&header))
                .and_then(|header| header.dimensions)
        } else {
            header_dimensions(input, self.image_decode_limits())
        };
        from_header
            .map_or(file_bytes.saturating_mul(4), |(width, height)| {
                u64::from(width) * u64::from(height)
            })
            // Larger images are refused when they are decoded.
            .min(self.config.max_pixels)
    }

    pub(super) fn image_decode_limits(&self) -> ImageDecodeLimits {
        ImageDecodeLimits {
            max_input_bytes: self.config.max_input_bytes,
            max_pixels: self.config.max_pixels,
            max_width: self.config.max_width,
            max_height: self.config.max_height,
            max_decoded_bytes: self.config.max_decoded_bytes,
            max_metadata_bytes: self.config.max_metadata_bytes,
            max_bits_per_pixel: self.config.max_bits_per_pixel,
        }
    }

    /// Limits for files the pipeline wrote itself, such as a decoded candidate.
    /// The limit on input bytes is about the user's files, not these.
    pub(super) fn pipeline_decode_limits(&self) -> ImageDecodeLimits {
        ImageDecodeLimits {
            max_input_bytes: u64::MAX,
            ..self.image_decode_limits()
        }
    }

    pub(super) fn decode_input(&self, input: &Path) -> Result<DecodedSource> {
        if input
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("avif"))
        {
            let source_size = fs::metadata(input)
                .with_context(|| format!("reading AVIF source metadata {}", input.display()))?
                .len();
            anyhow::ensure!(
                source_size <= self.config.max_input_bytes,
                "input is {source_size} bytes, exceeding the configured {} byte limit",
                self.config.max_input_bytes
            );
            let header = read_prefix(input)
                .with_context(|| format!("reading AVIF header {}", input.display()))?;
            let probe = probe_avif(&header);
            anyhow::ensure!(
                !probe.as_ref().is_some_and(|probe| probe.animated),
                "animated AVIF is not supported; only still images are optimized"
            );
            // The header's size is checked before the decoder tool allocates it.
            let limits = self.image_decode_limits();
            if let Some((width, height)) = probe.as_ref().and_then(|probe| probe.dimensions) {
                validate_decoder_dimensions(width, height, limits)?;
                // At most 8 bits a sample unless the header says more, four samples a pixel.
                let wide = probe
                    .and_then(|probe| probe.bit_depth)
                    .is_some_and(|bits| bits > 8);
                let decoded_bytes = u64::from(width) * u64::from(height) * if wide { 8 } else { 4 };
                anyhow::ensure!(
                    decoded_bytes <= limits.max_decoded_bytes,
                    "decoded image requires {decoded_bytes} bytes, exceeding the configured {} byte limit",
                    limits.max_decoded_bytes
                );
            }
            let tool = self
                .backend
                .tools()
                .avifdec
                .path
                .as_ref()
                .ok_or_else(|| anyhow!("avifdec is not available"))?;
            let temp = tempfile::tempdir().context("creating AVIF decode directory")?;
            let decoded_path = temp.path().join("source.png");
            run_process(
                tool,
                &[
                    OsString::from("--png-compress"),
                    OsString::from("1"),
                    // The tool refuses what the configured limits refuse, not only its own.
                    OsString::from("--size-limit"),
                    OsString::from(self.config.max_pixels.min(u64::from(u32::MAX)).to_string()),
                    OsString::from("--dimension-limit"),
                    OsString::from(
                        self.config
                            .max_width
                            .max(self.config.max_height)
                            .to_string(),
                    ),
                    std::path::absolute(input)?.into_os_string(),
                    decoded_path.as_os_str().to_os_string(),
                ],
                self.config.candidate_timeout(),
            )?;
            let _unit = work(1)?;
            let mut decoded =
                decode_source_with_limits(&decoded_path, self.pipeline_decode_limits())
                    .context("decoding AVIF source through avifdec")?;
            decoded.metadata.format = Some("Avif".to_string());
            Ok(decoded)
        } else {
            let _unit = work(1)?;
            decode_source_with_limits(input, self.image_decode_limits())
        }
    }

    /// Returns the replacement image when the alpha policy changes the pixels.
    pub(super) fn apply_alpha_policy(
        &self,
        image: &image::DynamicImage,
    ) -> Result<Option<image::DynamicImage>> {
        if !image.color().has_alpha() {
            return Ok(None);
        }
        let deep = matches!(
            image.color(),
            image::ColorType::Rgba16 | image::ColorType::La16
        );
        match &self.config.alpha {
            AlphaPolicy::Preserve => Ok(None),
            AlphaPolicy::Auto => {
                let opaque = image.to_rgba16().pixels().all(|pixel| pixel[3] == u16::MAX);
                Ok(opaque.then(|| {
                    if deep {
                        image::DynamicImage::ImageRgb16(image.to_rgb16())
                    } else {
                        image::DynamicImage::ImageRgb8(image.to_rgb8())
                    }
                }))
            }
            AlphaPolicy::Flatten { background } => {
                let background = parse_hex_color(background)?;
                let blend = |channel: u64, background: u64, alpha: u64, max: u64| {
                    (channel * alpha + background * (max - alpha) + max / 2) / max
                };
                Ok(Some(if deep {
                    let mut flat = image.to_rgb16();
                    for (out, source) in flat.pixels_mut().zip(image.to_rgba16().pixels()) {
                        for channel in 0..3 {
                            out[channel] = blend(
                                u64::from(source[channel]),
                                u64::from(background[channel]) * 257,
                                u64::from(source[3]),
                                65_535,
                            ) as u16;
                        }
                    }
                    image::DynamicImage::ImageRgb16(flat)
                } else {
                    let mut flat = image.to_rgb8();
                    for (out, source) in flat.pixels_mut().zip(image.to_rgba8().pixels()) {
                        for channel in 0..3 {
                            out[channel] = blend(
                                u64::from(source[channel]),
                                u64::from(background[channel]),
                                u64::from(source[3]),
                                255,
                            ) as u8;
                        }
                    }
                    image::DynamicImage::ImageRgb8(flat)
                }))
            }
        }
    }
}

pub(super) fn external_tools_accept_input(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png"
            )
        })
        .unwrap_or(false)
}

fn parse_hex_color(value: &str) -> Result<[u8; 3]> {
    let hex = value.trim().trim_start_matches('#');
    anyhow::ensure!(
        hex.len() == 6 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "background colour '{value}' is not in #rrggbb form"
    );
    let channel = |index: usize| u8::from_str_radix(&hex[index..index + 2], 16);
    Ok([channel(0)?, channel(2)?, channel(4)?])
}
