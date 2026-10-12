use std::{
    collections::HashSet,
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use anyhow::{Context, Result};
use image::{
    DynamicImage, GenericImageView, ImageDecoder, Limits, Pixel, RgbaImage, metadata::Orientation,
};
use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_IMAGE_WIDTH: u32 = 100_000;
pub const DEFAULT_MAX_IMAGE_HEIGHT: u32 = 100_000;
pub const DEFAULT_MAX_DECODED_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const DEFAULT_MAX_METADATA_BYTES: u64 = 16 * 1024 * 1024;
pub const DEFAULT_MAX_INPUT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct ImageDecodeLimits {
    pub max_input_bytes: u64,
    pub max_pixels: u64,
    pub max_width: u32,
    pub max_height: u32,
    pub max_decoded_bytes: u64,
    pub max_metadata_bytes: u64,
    pub max_bits_per_pixel: u16,
}

impl ImageDecodeLimits {
    pub fn with_max_pixels(max_pixels: u64) -> Self {
        Self {
            max_pixels,
            ..Self::default()
        }
    }
}

impl Default for ImageDecodeLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            max_pixels: 100_000_000,
            max_width: DEFAULT_MAX_IMAGE_WIDTH,
            max_height: DEFAULT_MAX_IMAGE_HEIGHT,
            max_decoded_bytes: DEFAULT_MAX_DECODED_BYTES,
            max_metadata_bytes: DEFAULT_MAX_METADATA_BYTES,
            max_bits_per_pixel: 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ContentClass {
    Photo,
    Screenshot,
    TransparentAsset,
    SimpleGraphic,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProbe {
    pub path: String,
    pub format: Option<String>,
    pub width: u32,
    pub height: u32,
    pub pixels: u64,
    pub color_type: String,
    pub original_color_type: String,
    pub has_alpha: bool,
    /// Bytes of ICC and EXIF data found in the source, not the file size.
    pub metadata_size: u64,
    pub icc_profile_size: Option<u64>,
    pub unique_color_ratio: f32,
    pub edge_density: f32,
    pub flat_region_ratio: f32,
    pub alpha_coverage: f32,
    pub orientation_applied: bool,
    pub content_class: ContentClass,
}

#[derive(Debug, Clone)]
pub struct SourceMetadata {
    pub format: Option<String>,
    pub original_color_type: String,
    pub icc_profile: Option<Vec<u8>>,
    pub exif_metadata: Option<Vec<u8>>,
    pub orientation: Orientation,
    /// Things about the source worth telling the user that do not stop it
    /// being processed.
    pub warnings: Vec<String>,
}

impl SourceMetadata {
    pub fn orientation_applied(&self) -> bool {
        self.orientation != Orientation::NoTransforms
    }

    /// ICC plus EXIF bytes as read from the file, which is what
    /// `max_metadata_bytes` limits.
    pub fn metadata_size(&self) -> u64 {
        let len = |data: &Option<Vec<u8>>| data.as_ref().map_or(0, Vec::len) as u64;
        len(&self.icc_profile) + len(&self.exif_metadata)
    }

    pub fn icc_profile_size(&self) -> Option<u64> {
        self.icc_profile
            .as_ref()
            .map(|profile| profile.len() as u64)
    }

    pub fn sanitized_exif_metadata(&self) -> Option<Vec<u8>> {
        self.exif_metadata.as_ref().map(|exif| {
            let mut sanitized = exif.clone();
            if self.orientation_applied() {
                let _ = Orientation::remove_from_exif_chunk(&mut sanitized);
            }
            sanitized
        })
    }
}

pub struct DecodedSource {
    pub image: DynamicImage,
    pub metadata: SourceMetadata,
}

pub fn decode_source(path: &Path, max_pixels: u64) -> Result<DecodedSource> {
    decode_source_with_limits(path, ImageDecodeLimits::with_max_pixels(max_pixels))
}

pub fn decode_source_with_limits(path: &Path, limits: ImageDecodeLimits) -> Result<DecodedSource> {
    let input_bytes = std::fs::metadata(path)
        .with_context(|| format!("reading image metadata {}", path.display()))?
        .len();
    anyhow::ensure!(
        input_bytes <= limits.max_input_bytes,
        "input is {input_bytes} bytes, exceeding the configured {} byte limit",
        limits.max_input_bytes
    );
    let mut reader = image::ImageReader::open(path)
        .with_context(|| format!("opening image {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("detecting image format for {}", path.display()))?;
    let detected_format = reader.format();
    validate_detected_format(path, detected_format)?;
    validate_animation(path, detected_format)?;
    if detected_format == Some(image::ImageFormat::WebP) {
        check_webp_metadata_chunks(path, limits.max_metadata_bytes)?;
    }
    let mut warnings = Vec::new();
    if detected_format == Some(image::ImageFormat::Png)
        && let Some(tag) = png_uncarried_colour_tag(path)?
    {
        match tag {
            PngColourTag::TreatedAsSrgb(tag) => {
                warnings.push(format!(
                    "the image declares its colour with {tag}, which is treated as sRGB"
                ));
            }
            PngColourTag::Refused(tag) => anyhow::bail!(
                "the image declares its colour with {tag}, which is neither carried to the output nor converted, so it is skipped rather than risk shifting its colours"
            ),
        }
    }
    let mut jpeg_components = None;
    if detected_format == Some(image::ImageFormat::Jpeg) {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        anyhow::ensure!(
            jpeg_is_complete(&bytes),
            "JPEG data is truncated: no end-of-image marker"
        );
        jpeg_components = jpeg_component_count(&bytes);
    }
    let format = detected_format.map(|format| format!("{format:?}"));
    // The PNG decoder inflates the embedded profile while it is created, and
    // its allocation budget cannot be changed afterwards. It is therefore made
    // with one that fits the metadata, and given the pixel limit once it exists.
    reader.limits(metadata_limits(limits));
    let mut decoder = reader.into_decoder().map_err(|error| match error {
        image::ImageError::Limits(_) => anyhow::anyhow!(
            "image metadata exceeds the configured {} byte limit",
            limits.max_metadata_bytes
        ),
        error => anyhow::Error::new(error)
            .context(format!("creating image decoder for {}", path.display())),
    })?;
    decoder
        .set_limits(decode_limits(limits))
        .with_context(|| format!("limiting the decoder for {}", path.display()))?;
    let (width, height) = decoder.dimensions();
    validate_decoder_dimensions(width, height, limits)?;
    let decoded_bytes = decoder.total_bytes();
    anyhow::ensure!(
        decoded_bytes <= limits.max_decoded_bytes,
        "decoded image requires {decoded_bytes} bytes, exceeding the configured {} byte limit",
        limits.max_decoded_bytes
    );
    // The decoder converts four-component JPEGs to RGB and reports them as
    // such; the frame header is the only place the source model survives.
    let original_color_type = if jpeg_components == Some(4) {
        "Cmyk8".to_string()
    } else {
        format!("{:?}", decoder.original_color_type())
    };
    let bits_per_pixel = decoder.original_color_type().bits_per_pixel();
    anyhow::ensure!(
        bits_per_pixel <= limits.max_bits_per_pixel,
        "image uses {bits_per_pixel} bits per pixel, exceeding the configured {} bit limit",
        limits.max_bits_per_pixel
    );
    let icc_profile = decoder
        .icc_profile()
        .with_context(|| format!("reading ICC profile from {}", path.display()))?;
    // The PNG decoder drops a profile it cannot read, and one that inflates
    // past its allowance, without saying so.
    anyhow::ensure!(
        !(detected_format == Some(image::ImageFormat::Png)
            && icc_profile.is_none()
            && png_has_profile_chunk(path)?),
        "the embedded ICC profile is unreadable or larger than the configured {} byte metadata limit allows, so the image is skipped rather than risk shifting its colours",
        limits.max_metadata_bytes
    );
    let exif_metadata = decoder
        .exif_metadata()
        .with_context(|| format!("reading EXIF metadata from {}", path.display()))?;
    anyhow::ensure!(
        !(jpeg_components == Some(4)
            && icc_profile
                .as_deref()
                .and_then(smart_compress_codecs::icc_color_space)
                .as_ref()
                == Some(b"CMYK")),
        "CMYK JPEG with an embedded CMYK ICC profile: converting it to RGB without colour management would change its colours, so it is skipped"
    );
    let metadata_bytes = icc_profile
        .as_ref()
        .map_or(0, Vec::len)
        .checked_add(exif_metadata.as_ref().map_or(0, Vec::len))
        .context("image metadata size overflow")? as u64;
    anyhow::ensure!(
        metadata_bytes <= limits.max_metadata_bytes,
        "image metadata is {metadata_bytes} bytes, exceeding the configured {} byte limit",
        limits.max_metadata_bytes
    );
    let orientation = decoder
        .orientation()
        .with_context(|| format!("reading orientation from {}", path.display()))?;
    let image = DynamicImage::from_decoder(decoder)
        .with_context(|| format!("decoding image {}", path.display()))?;
    Ok(DecodedSource {
        image,
        metadata: SourceMetadata {
            format,
            original_color_type,
            icc_profile,
            exif_metadata,
            orientation,
            warnings,
        },
    })
}

pub fn decode_and_probe(path: &Path, max_pixels: u64) -> Result<(DynamicImage, ImageProbe)> {
    let mut decoded = decode_source(path, max_pixels)?;
    apply_orientation(&mut decoded.image, decoded.metadata.orientation);
    let probe = probe_image(path, &decoded.image, &decoded.metadata, None);
    Ok((decoded.image, probe))
}

pub fn apply_orientation(image: &mut DynamicImage, orientation: Orientation) {
    if orientation != Orientation::NoTransforms {
        image.apply_orientation(orientation);
    }
}

pub fn probe_image(
    path: &Path,
    image: &DynamicImage,
    metadata: &SourceMetadata,
    overview: Option<&RgbaImage>,
) -> ImageProbe {
    let color_type = format!("{:?}", image.color());
    let has_alpha = image.color().has_alpha();
    let (width, height) = image.dimensions();
    let analysis = overview.map_or_else(
        || analyze_image(image, has_alpha),
        |overview| analyze_overview(overview, has_alpha),
    );
    let content_class = classify(&analysis, has_alpha, width as u64 * height as u64);

    ImageProbe {
        path: path.display().to_string(),
        format: metadata.format.clone(),
        width,
        height,
        pixels: width as u64 * height as u64,
        color_type,
        original_color_type: metadata.original_color_type.clone(),
        has_alpha,
        metadata_size: metadata.metadata_size(),
        icc_profile_size: metadata.icc_profile_size(),
        unique_color_ratio: analysis.unique_color_ratio,
        edge_density: analysis.edge_density,
        flat_region_ratio: analysis.flat_region_ratio,
        alpha_coverage: analysis.alpha_coverage,
        orientation_applied: metadata.orientation_applied(),
        content_class,
    }
}

pub fn decode_image_with_limits(path: &Path, limits: ImageDecodeLimits) -> Result<DynamicImage> {
    let mut decoded = decode_source_with_limits(path, limits)?;
    apply_orientation(&mut decoded.image, decoded.metadata.orientation);
    Ok(decoded.image)
}

/// Why an embedded ICC profile cannot describe the decoded pixels, if it
/// cannot. Such a profile is worse than none: viewers would apply the wrong
/// transform, and CMYK sources are already converted to RGB by the decoder.
pub fn icc_profile_mismatch(profile: &[u8], color: image::ColorType) -> Option<String> {
    let Some(space) = smart_compress_codecs::icc_color_space(profile) else {
        return Some("the embedded data is not an ICC profile".to_string());
    };
    let expected: &[u8; 4] = if color.has_color() { b"RGB " } else { b"GRAY" };
    (&space != expected).then(|| {
        format!(
            "it describes {} data but the image decodes to {}",
            String::from_utf8_lossy(&space).trim(),
            String::from_utf8_lossy(expected).trim()
        )
    })
}

enum PngColourTag {
    /// Declared colour that viewers show the same as sRGB.
    TreatedAsSrgb(String),
    Refused(String),
}

/// Colour description in a PNG that smart-compress neither carries (it writes
/// only an ICC profile) nor converts, and that a viewer would apply, if the
/// file has one. An embedded profile is carried, so it settles the question;
/// otherwise `cICP` outranks `sRGB`, `gAMA` and `cHRM`, and each must say sRGB.
/// `avifdec` writes `cICP` for AVIF sources whose colour is not an ICC profile.
fn png_uncarried_colour_tag(path: &Path) -> Result<Option<PngColourTag>> {
    let mut file =
        BufReader::new(File::open(path).with_context(|| format!("opening {}", path.display()))?);
    file.seek_relative(8)?;
    let (mut icc, mut srgb) = (false, false);
    let (mut cicp, mut gamma, mut chromaticities) = (None, None, None);
    let mut head = [0_u8; 8];
    while file.read_exact(&mut head).is_ok() {
        let length = i64::from(u32::from_be_bytes([head[0], head[1], head[2], head[3]]));
        match &head[4..] {
            b"IDAT" | b"IEND" => break,
            b"iCCP" => icc = true,
            b"sRGB" => srgb = true,
            kind @ (b"cICP" | b"gAMA" | b"cHRM") if length <= 32 => {
                let mut data = vec![0_u8; length as usize];
                file.read_exact(&mut data)?;
                match kind {
                    b"cICP" => cicp = Some(data),
                    b"gAMA" => gamma = Some(data),
                    _ => chromaticities = Some(data),
                }
                file.seek_relative(4)?;
                continue;
            }
            _ => {}
        }
        file.seek_relative(length + 4)?;
    }
    if icc {
        return Ok(None);
    }
    // 1 is BT.709 and sRGB primaries, 13 the sRGB curve; 2 is "unspecified".
    // The BT.709 curve (1) is close enough to it that viewers show the two alike.
    if let Some(&[primaries, transfer, ..]) = cicp.as_deref() {
        let tag = format!("cICP (primaries {primaries}, transfer {transfer})");
        return Ok(match (primaries, transfer) {
            (1 | 2, 2 | 13) => None,
            (1, 1) => Some(PngColourTag::TreatedAsSrgb(tag)),
            _ => Some(PngColourTag::Refused(tag)),
        });
    }
    if srgb {
        return Ok(None);
    }
    let value = |data: &[u8], index: usize| {
        data.get(4 * index..4 * index + 4).map_or(f64::NAN, |word| {
            f64::from(u32::from_be_bytes([word[0], word[1], word[2], word[3]])) / 100_000.0
        })
    };
    if let Some(data) = &gamma
        && !(0.44..=0.47).contains(&value(data, 0))
    {
        return Ok(Some(PngColourTag::Refused(format!(
            "gAMA {}",
            value(data, 0)
        ))));
    }
    // White point, then red, green and blue as x, y pairs.
    const SRGB: [f64; 8] = [0.3127, 0.329, 0.64, 0.33, 0.3, 0.6, 0.15, 0.06];
    if let Some(data) = &chromaticities
        && SRGB
            .iter()
            .enumerate()
            .any(|(index, expected)| (value(data, index) - expected).abs() > 0.02)
    {
        return Ok(Some(PngColourTag::Refused(
            "cHRM chromaticities that are not sRGB".to_string(),
        )));
    }
    Ok(None)
}

/// Reads the uncompressed PAM (`P7`) file a decoder tool wrote. It replaces a
/// PNG round trip, which spent most of the candidate decode time compressing
/// and decompressing a temporary file.
pub fn read_pam(path: &Path, limits: ImageDecodeLimits) -> Result<DynamicImage> {
    let mut bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let end_marker = b"ENDHDR\n";
    let header_end = bytes
        .windows(end_marker.len())
        .take(4096)
        .position(|window| window == end_marker)
        .context("PAM header is not terminated")?
        + end_marker.len();
    let header = std::str::from_utf8(&bytes[..header_end]).context("PAM header is not text")?;
    anyhow::ensure!(header.starts_with("P7\n"), "not a PAM file");
    let field = |name: &str| -> Result<u32> {
        header
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|value| value.trim().parse().ok())
            .with_context(|| format!("PAM header has no valid {}", name.trim()))
    };
    let (width, height) = (field("WIDTH ")?, field("HEIGHT ")?);
    let (depth, max_value) = (field("DEPTH ")?, field("MAXVAL ")?);
    anyhow::ensure!((1..=4).contains(&depth), "unsupported PAM depth {depth}");
    validate_decoder_dimensions(width, height, limits)?;
    let wide = match max_value {
        255 => false,
        65_535 => true,
        other => anyhow::bail!("unsupported PAM sample range {other}"),
    };
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|samples| samples.checked_mul(u64::from(depth) * if wide { 2 } else { 1 }))
        .context("PAM dimensions overflow")?;
    anyhow::ensure!(
        expected <= limits.max_decoded_bytes,
        "decoded image requires {expected} bytes, exceeding the configured {} byte limit",
        limits.max_decoded_bytes
    );
    bytes.drain(..header_end);
    anyhow::ensure!(
        bytes.len() as u64 == expected,
        "PAM data is {} bytes, expected {expected}",
        bytes.len()
    );
    let mismatch = || anyhow::anyhow!("PAM buffer does not match {width}x{height}");
    if wide {
        // Samples are big-endian.
        let words = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_be_bytes(*pair))
            .collect::<Vec<_>>();
        Ok(match depth {
            1 => DynamicImage::ImageLuma16(
                image::ImageBuffer::from_raw(width, height, words).ok_or_else(mismatch)?,
            ),
            2 => DynamicImage::ImageLumaA16(
                image::ImageBuffer::from_raw(width, height, words).ok_or_else(mismatch)?,
            ),
            3 => DynamicImage::ImageRgb16(
                image::ImageBuffer::from_raw(width, height, words).ok_or_else(mismatch)?,
            ),
            4 => DynamicImage::ImageRgba16(
                image::ImageBuffer::from_raw(width, height, words).ok_or_else(mismatch)?,
            ),
            other => anyhow::bail!("unsupported PAM depth {other}"),
        })
    } else {
        Ok(match depth {
            1 => DynamicImage::ImageLuma8(
                image::ImageBuffer::from_raw(width, height, bytes).ok_or_else(mismatch)?,
            ),
            2 => DynamicImage::ImageLumaA8(
                image::ImageBuffer::from_raw(width, height, bytes).ok_or_else(mismatch)?,
            ),
            3 => DynamicImage::ImageRgb8(
                image::ImageBuffer::from_raw(width, height, bytes).ok_or_else(mismatch)?,
            ),
            4 => DynamicImage::ImageRgba8(
                image::ImageBuffer::from_raw(width, height, bytes).ok_or_else(mismatch)?,
            ),
            other => anyhow::bail!("unsupported PAM depth {other}"),
        })
    }
}

/// Whether a PNG has an `iCCP` chunk before its image data.
fn png_has_profile_chunk(path: &Path) -> Result<bool> {
    let mut file =
        BufReader::new(File::open(path).with_context(|| format!("opening {}", path.display()))?);
    file.seek_relative(8)?;
    let mut head = [0_u8; 8];
    while file.read_exact(&mut head).is_ok() {
        match &head[4..] {
            b"iCCP" => return Ok(true),
            b"IDAT" | b"IEND" => break,
            _ => {}
        }
        let length = i64::from(u32::from_be_bytes([head[0], head[1], head[2], head[3]]));
        file.seek_relative(length + 4)?;
    }
    Ok(false)
}

/// Dimensions from the header alone, read with the same allowance as the
/// decode so that the profile is not inflated beyond it here either.
pub(crate) fn header_dimensions(path: &Path, limits: ImageDecodeLimits) -> Option<(u32, u32)> {
    let mut reader = image::ImageReader::open(path)
        .and_then(|reader| reader.with_guessed_format())
        .ok()?;
    // A format that is refused later is not parsed here either.
    if !reader.format().is_some_and(is_supported_format) {
        return None;
    }
    reader.limits(metadata_limits(limits));
    reader.into_dimensions().ok()
}

/// Room for the metadata the PNG decoder keeps (a chunk is buffered while it
/// is read and its inflated profile is kept) and a few rows of pixels.
fn metadata_limits(config: ImageDecodeLimits) -> Limits {
    let mut limits = Limits::default();
    limits.max_alloc = Some(
        config
            .max_metadata_bytes
            .saturating_mul(4)
            .saturating_add(u64::from(config.max_width) * 32),
    );
    limits
}

/// A WebP chunk is read into memory at the size its header gives, which the
/// decoder does not compare with the file. The metadata chunks are checked
/// against the metadata limit before it reads them.
fn check_webp_metadata_chunks(path: &Path, max_bytes: u64) -> Result<()> {
    let mut file =
        BufReader::new(File::open(path).with_context(|| format!("opening {}", path.display()))?);
    file.seek_relative(12)?;
    let mut head = [0_u8; 8];
    while file.read_exact(&mut head).is_ok() {
        let length = u64::from(u32::from_le_bytes([head[4], head[5], head[6], head[7]]));
        anyhow::ensure!(
            !matches!(&head[..4], b"ICCP" | b"EXIF" | b"XMP ") || length <= max_bytes,
            "image metadata is {length} bytes, exceeding the configured {max_bytes} byte limit"
        );
        // Chunks are padded to an even size.
        file.seek_relative((length + (length & 1)) as i64)?;
    }
    Ok(())
}

fn decode_limits(config: ImageDecodeLimits) -> Limits {
    // Width, height and pixel limits are checked from the decoder header by
    // `validate_decoder_dimensions`, which names the limit that was hit.
    let mut limits = Limits::default();
    limits.max_alloc = Some(config.max_decoded_bytes);
    limits
}

/// Component count from the first frame header (SOF) of a JPEG.
fn jpeg_component_count(bytes: &[u8]) -> Option<u8> {
    let mut index = 2;
    while index + 3 < bytes.len() {
        if bytes[index] != 0xFF {
            return None;
        }
        let marker = bytes[index + 1];
        let length = usize::from(u16::from_be_bytes([bytes[index + 2], bytes[index + 3]]));
        // SOF0..SOF15, except DHT (C4), JPG (C8) and DAC (CC).
        if (0xC0..=0xCF).contains(&marker) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
            return bytes.get(index + 9).copied();
        }
        index += 2 + length;
    }
    None
}

/// Walks the JPEG markers and reports whether the image ends with an
/// end-of-image marker. The decoder accepts cut-off files and fills in the
/// missing rows, which would be optimized as if they were the picture.
fn jpeg_is_complete(bytes: &[u8]) -> bool {
    let mut index = 2;
    while index + 1 < bytes.len() {
        if bytes[index] != 0xFF {
            index += 1;
            continue;
        }
        match bytes[index + 1] {
            0xD9 => return true,
            // Fill byte before a marker.
            0xFF => index += 1,
            // Stuffed data byte, restart markers and TEM carry no length.
            0x00 | 0x01 | 0xD0..=0xD7 => index += 2,
            // Every other marker is a segment; skipping by length also steps
            // over thumbnails embedded in APP segments.
            _ => {
                let Some(length) = bytes.get(index + 2..index + 4) else {
                    return false;
                };
                index += 2 + usize::from(u16::from_be_bytes([length[0], length[1]]));
            }
        }
    }
    false
}

pub(crate) fn validate_decoder_dimensions(
    width: u32,
    height: u32,
    limits: ImageDecodeLimits,
) -> Result<()> {
    anyhow::ensure!(
        width <= limits.max_width,
        "image width {width} exceeds the configured {} pixel width limit",
        limits.max_width
    );
    anyhow::ensure!(
        height <= limits.max_height,
        "image height {height} exceeds the configured {} pixel height limit",
        limits.max_height
    );
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .context("image dimensions overflow the pixel count")?;
    anyhow::ensure!(
        pixels <= limits.max_pixels,
        "image has {pixels} pixels, exceeding the configured {} pixel limit",
        limits.max_pixels
    );
    Ok(())
}

/// The formats the decoders here read; AVIF is probed separately.
fn is_supported_format(format: image::ImageFormat) -> bool {
    matches!(
        format,
        image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP
    )
}

fn validate_detected_format(path: &Path, format: Option<image::ImageFormat>) -> Result<()> {
    let format = format.context("image format could not be detected from file content")?;
    anyhow::ensure!(
        is_supported_format(format),
        "unsupported input format {format:?}; supported inputs are PNG, JPEG, WebP, and AVIF"
    );
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .context("input has no recognizable file extension")?;
    let extension_format = match extension.as_str() {
        "png" => image::ImageFormat::Png,
        "jpg" | "jpeg" => image::ImageFormat::Jpeg,
        "webp" => image::ImageFormat::WebP,
        "bmp" => image::ImageFormat::Bmp,
        "gif" => image::ImageFormat::Gif,
        "tif" | "tiff" => image::ImageFormat::Tiff,
        "svg" => anyhow::bail!("SVG input is not supported"),
        _ => anyhow::bail!("unsupported input extension '.{extension}'"),
    };
    anyhow::ensure!(
        extension_format == format,
        "file extension indicates {extension_format:?}, but content is {format:?}"
    );
    Ok(())
}

/// Animated WebP and PNG are refused: the optimizer works on one frame, and
/// would drop the rest.
fn validate_animation(path: &Path, format: Option<image::ImageFormat>) -> Result<()> {
    match format {
        Some(image::ImageFormat::WebP) => {
            let decoder = image::codecs::webp::WebPDecoder::new(BufReader::new(
                File::open(path).with_context(|| format!("opening WebP {path:?}"))?,
            ))
            .with_context(|| format!("reading WebP animation metadata {}", path.display()))?;
            anyhow::ensure!(
                !decoder.has_animation(),
                "animated WebP input is not supported"
            );
        }
        Some(image::ImageFormat::Png) if contains_marker(path, b"acTL")? => {
            anyhow::bail!("animated PNG input is not supported");
        }
        _ => {}
    }
    Ok(())
}

fn contains_marker(path: &Path, marker: &[u8]) -> Result<bool> {
    const MAX_SCAN_BYTES: usize = 1024 * 1024;
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut bytes = Vec::new();
    file.by_ref()
        .take(MAX_SCAN_BYTES as u64)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading animation metadata {}", path.display()))?;
    Ok(bytes.windows(marker.len()).any(|window| window == marker))
}

struct Analysis {
    unique_color_ratio: f32,
    edge_density: f32,
    flat_region_ratio: f32,
    alpha_coverage: f32,
}

fn analyze_image(image: &DynamicImage, has_alpha: bool) -> Analysis {
    // Never upscale: interpolated pixels are smooth and inflate the flat share.
    let overview = if image.width().max(image.height()) <= 512 {
        image.to_rgba8()
    } else {
        image.thumbnail(512, 512).to_rgba8()
    };
    analyze_overview(&overview, has_alpha)
}

fn analyze_overview(overview: &RgbaImage, has_alpha: bool) -> Analysis {
    let total = overview.width() as usize * overview.height() as usize;
    if total == 0 {
        return Analysis {
            unique_color_ratio: 0.0,
            edge_density: 0.0,
            flat_region_ratio: 0.0,
            alpha_coverage: 0.0,
        };
    }

    let mut unique = HashSet::with_capacity(total.min(4096));
    let mut alpha_pixels = 0_usize;
    for pixel in overview.pixels().step_by(4) {
        unique.insert([pixel[0] / 8, pixel[1] / 8, pixel[2] / 8, pixel[3] / 8]);
        if pixel[3] < 250 {
            alpha_pixels += 1;
        }
    }

    let mut edge_count = 0_usize;
    let mut flat_count = 0_usize;
    if overview.width() > 2 && overview.height() > 2 {
        for y in 1..overview.height() - 1 {
            for x in 1..overview.width() - 1 {
                let center = luma(overview.get_pixel(x, y));
                let right = luma(overview.get_pixel(x + 1, y));
                let bottom = luma(overview.get_pixel(x, y + 1));
                let gradient = center.abs_diff(right) as u16 + center.abs_diff(bottom) as u16;
                if gradient > 48 {
                    edge_count += 1;
                }
                if gradient < 6 {
                    flat_count += 1;
                }
            }
        }
    }

    Analysis {
        unique_color_ratio: unique.len() as f32 / (total / 4).max(1) as f32,
        edge_density: edge_count as f32 / total as f32,
        flat_region_ratio: flat_count as f32 / total as f32,
        alpha_coverage: if has_alpha {
            alpha_pixels as f32 / (total / 4).max(1) as f32
        } else {
            0.0
        },
    }
}

/// Below this the overview has few colour samples, and `unique_color_ratio`
/// (unique colours per sample) reads near 1 for any smooth gradient, so the
/// colour statistics cannot tell a photo from a graphic. 128 x 128 px.
const MIN_CLASSIFIED_PIXELS: u64 = 128 * 128;

/// Thresholds come from the 512 px overview features of the labelled set
/// (generated corpus, tests/fixtures, ten samples/photos photographs):
/// - `unique_color_ratio` of palette graphics (flat icons, screenshots, text,
///   lines) is at most 0.005 at 512 px and above; the colour photographs span
///   0.0099 to 0.067, and 0.007 sits between them. Greyscale and foggy
///   photographs fall below it too (0.0007 to 0.0018; a grey image has at most
///   32 quantised levels), so below 0.007 the next two features decide.
/// - `edge_density` of text, lines and screenshots is at least 0.2 and of the
///   sharpest photograph 0.145; 0.175 splits them. A greyscale photograph
///   sharper than that would be called a screenshot.
/// - `flat_region_ratio` of the photographs is at most 0.946, flat graphics are
///   at least 0.969, smooth gradients and ramps are 0.99. At 0.95 and above the
///   content is too smooth to call a photo.
fn classify(analysis: &Analysis, has_alpha: bool, pixels: u64) -> ContentClass {
    if has_alpha {
        return ContentClass::TransparentAsset;
    }
    if pixels < MIN_CLASSIFIED_PIXELS {
        return ContentClass::Unknown;
    }
    let smooth = analysis.flat_region_ratio >= 0.95;
    if analysis.unique_color_ratio < 0.007 {
        if analysis.edge_density > 0.175 {
            return ContentClass::Screenshot;
        }
        if smooth {
            return ContentClass::SimpleGraphic;
        }
    }
    if smooth {
        ContentClass::Unknown
    } else {
        ContentClass::Photo
    }
}

fn luma<P: Pixel<Subpixel = u8>>(pixel: &P) -> u8 {
    let rgba = pixel.to_rgba();
    (0.2126 * rgba[0] as f32 + 0.7152 * rgba[1] as f32 + 0.0722 * rgba[2] as f32).round() as u8
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_MAX_IMAGE_HEIGHT, DEFAULT_MAX_IMAGE_WIDTH, ImageDecodeLimits,
        validate_decoder_dimensions,
    };
    use image::{ImageFormat, Rgb, RgbImage};

    #[test]
    fn pam_depth_is_bounded_before_sizes_are_computed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.pam");
        let header = |depth: u32, width: u32| {
            format!(
                "P7\nWIDTH {width}\nHEIGHT 4\nDEPTH {depth}\nMAXVAL 65535\nTUPLTYPE RGB\nENDHDR\n"
            )
        };
        let limits = ImageDecodeLimits {
            max_width: u32::MAX,
            max_pixels: u64::MAX,
            max_decoded_bytes: u64::MAX,
            ..ImageDecodeLimits::default()
        };
        std::fs::write(&path, header(u32::MAX, 4)).unwrap();
        let error = super::read_pam(&path, limits).unwrap_err();
        assert!(error.to_string().contains("unsupported PAM depth"));

        // Sizes that overflow are an error, not a wrapped (small) number.
        std::fs::write(&path, header(4, u32::MAX)).unwrap();
        let error = super::read_pam(&path, limits).unwrap_err();
        assert!(!error.to_string().contains("expected 0"), "{error:#}");
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = !0_u32;
        for &byte in data {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    /// A zlib stream of zeros: one literal, then `matches` copies of 258 bytes
    /// (a fixed-Huffman block), so a few KB inflate to about 258 times that.
    fn zlib_zeros(matches: usize) -> (Vec<u8>, usize) {
        let mut bits = vec![1, 1, 0]; // final block, type 01 (low bit first)
        let mut code = |value: u32, length: u32| {
            bits.extend((0..length).rev().map(|bit| ((value >> bit) & 1) as u8));
        };
        code(0x30, 8); // literal 0
        for _ in 0..matches {
            code(0xC5, 8); // length 258
            code(0, 5); // distance 1
        }
        code(0, 7); // end of block
        let mut zlib = vec![0x78, 0x01];
        zlib.extend(bits.chunks(8).map(|byte| {
            byte.iter()
                .enumerate()
                .fold(0_u8, |packed, (index, bit)| packed | (bit << index))
        }));
        let inflated = 1 + 258 * matches;
        // Adler-32 of zeros: 1 in the low half, the length (mod 65521) in the high.
        zlib.extend_from_slice(&((((inflated % 65_521) as u32) << 16) | 1).to_be_bytes());
        (zlib, inflated)
    }

    /// A 2x2 PNG with an iCCP chunk whose profile is `matches` * 258 zero bytes.
    fn png_with_profile(matches: usize) -> (Vec<u8>, usize) {
        let mut png = Vec::new();
        RgbImage::from_pixel(2, 2, Rgb([1, 2, 3]))
            .write_to(&mut std::io::Cursor::new(&mut png), ImageFormat::Png)
            .unwrap();
        let (zlib, inflated) = zlib_zeros(matches);
        let mut data = b"p\0\0".to_vec();
        data.extend_from_slice(&zlib);
        let mut chunk = (data.len() as u32).to_be_bytes().to_vec();
        chunk.extend_from_slice(b"iCCP");
        chunk.extend_from_slice(&data);
        chunk.extend_from_slice(&crc32(&chunk[4..]).to_be_bytes());
        // After the signature (8) and IHDR (25 bytes).
        png.splice(33..33, chunk);
        (png, inflated)
    }

    #[test]
    fn a_png_profile_is_bounded_before_it_is_inflated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.png");
        let limits = ImageDecodeLimits {
            max_metadata_bytes: 8_000,
            max_width: 10,
            ..ImageDecodeLimits::default()
        };
        // About 6 KB of file for a megabyte of profile.
        let (png, _) = png_with_profile(4_000);
        std::fs::write(&path, png).unwrap();
        let error = super::decode_source_with_limits(&path, limits)
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("ICC profile is unreadable or larger"),
            "{error:#}"
        );
        assert!(super::header_dimensions(&path, limits).is_some());

        // The same limits let a small profile and the pixels through.
        let (png, inflated) = png_with_profile(1);
        std::fs::write(&path, png).unwrap();
        let decoded = super::decode_source_with_limits(&path, limits).unwrap();
        assert_eq!(
            decoded.metadata.icc_profile.map(|profile| profile.len()),
            Some(inflated)
        );
    }

    #[test]
    fn webp_metadata_chunks_are_checked_against_the_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.webp");
        // RIFF header, then an ICCP chunk that claims 3 GiB in a 30 byte file.
        let mut bytes = b"RIFF\x16\x00\x00\x00WEBPICCP".to_vec();
        bytes.extend_from_slice(&0xC000_0000_u32.to_le_bytes());
        bytes.extend_from_slice(&[0; 10]);
        std::fs::write(&path, bytes).unwrap();

        let error = super::check_webp_metadata_chunks(&path, 16 * 1024 * 1024).unwrap_err();

        assert!(error.to_string().contains("3221225472 bytes"), "{error}");
    }

    #[test]
    fn decoder_dimensions_accept_limits_exactly() {
        let limits = ImageDecodeLimits {
            max_pixels: u64::MAX,
            ..ImageDecodeLimits::default()
        };
        validate_decoder_dimensions(DEFAULT_MAX_IMAGE_WIDTH, 1, limits).unwrap();
        validate_decoder_dimensions(1, DEFAULT_MAX_IMAGE_HEIGHT, limits).unwrap();
    }

    #[test]
    fn decoder_dimensions_reject_oversized_width() {
        let error = validate_decoder_dimensions(
            DEFAULT_MAX_IMAGE_WIDTH + 1,
            1,
            ImageDecodeLimits::default(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("image width 100001"));
    }

    #[test]
    fn decoder_dimensions_reject_oversized_height() {
        let error = validate_decoder_dimensions(
            1,
            DEFAULT_MAX_IMAGE_HEIGHT + 1,
            ImageDecodeLimits::default(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("image height 100001"));
    }

    #[test]
    fn decoder_dimensions_reject_oversized_pixel_count() {
        let limits = ImageDecodeLimits {
            max_pixels: 109,
            ..ImageDecodeLimits::default()
        };
        let error = validate_decoder_dimensions(10, 11, limits).unwrap_err();

        assert_eq!(
            error.to_string(),
            "image has 110 pixels, exceeding the configured 109 pixel limit"
        );
    }

    #[test]
    fn spoofed_extension_is_rejected_from_content_probe() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.jpg");
        RgbImage::from_pixel(2, 2, Rgb([1, 2, 3]))
            .save_with_format(&path, ImageFormat::Png)
            .unwrap();

        let error = match super::decode_source_with_limits(&path, ImageDecodeLimits::default()) {
            Ok(_) => panic!("spoofed extension was accepted"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("file extension indicates Jpeg"));
    }
}
