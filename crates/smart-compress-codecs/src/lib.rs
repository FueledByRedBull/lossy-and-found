use std::{
    collections::VecDeque,
    ffi::{OsStr, OsString},
    fs,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Condvar, Mutex, MutexGuard, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use image::{
    DynamicImage, ImageEncoder,
    codecs::{jpeg::JpegEncoder, png::PngEncoder},
};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Codec {
    Png,
    Jpeg,
    Webp,
    Avif,
    Jxl,
}

impl Codec {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Webp => "webp",
            Self::Avif => "avif",
            Self::Jxl => "jxl",
        }
    }

    pub fn supports_alpha(self) -> bool {
        !matches!(self, Self::Jpeg)
    }

    pub fn compatibility_rank(self) -> u8 {
        match self {
            Self::Jpeg => 5,
            Self::Png => 5,
            Self::Webp => 4,
            Self::Avif => 3,
            Self::Jxl => 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EncoderSettings {
    Lossless,
    Quality { quality: u8, effort: u8 },
    Palette { colors: u16 },
}

impl EncoderSettings {
    pub fn label(&self) -> String {
        match self {
            Self::Lossless => "lossless".to_string(),
            Self::Quality { quality, effort } => format!("q{quality}/e{effort}"),
            Self::Palette { colors } => format!("palette-{colors}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    /// Where the tool runs from; reports and the control panel do not carry it.
    #[serde(skip_serializing)]
    pub path: Option<PathBuf>,
    pub version: Option<String>,
    pub enabled: bool,
    pub install_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInventory {
    pub cwebp: ToolInfo,
    pub webpmux: ToolInfo,
    pub avifenc: ToolInfo,
    pub avifdec: ToolInfo,
    pub cjxl: ToolInfo,
    pub djxl: ToolInfo,
    pub oxipng: ToolInfo,
    pub pngquant: ToolInfo,
    pub cjpegli: ToolInfo,
    pub cjpeg: ToolInfo,
    pub ssimulacra2: ToolInfo,
    pub butteraugli: ToolInfo,
}

impl ToolInventory {
    /// Detection spawns every tool once; the result is reused for the process lifetime.
    pub fn detect() -> Self {
        static INVENTORY: OnceLock<ToolInventory> = OnceLock::new();
        INVENTORY.get_or_init(Self::detect_uncached).clone()
    }

    fn detect_uncached() -> Self {
        Self {
            cwebp: detect_tool(
                "cwebp",
                &["-version"],
                "Install libwebp tools or place cwebp.exe under tools/libwebp/.../bin.",
            ),
            webpmux: detect_tool(
                "webpmux",
                &["-version"],
                "Install libwebp tools or place webpmux.exe next to cwebp.exe; WebP metadata needs it.",
            ),
            avifenc: detect_tool(
                "avifenc",
                &["--version"],
                "Install libavif tools or place avifenc.exe and avifdec.exe under tools/libavif.",
            ),
            avifdec: detect_tool(
                "avifdec",
                &["--version"],
                "Install libavif tools or place avifenc.exe and avifdec.exe under tools/libavif.",
            ),
            cjxl: detect_tool(
                "cjxl",
                &["--version"],
                "Install JPEG XL tools or place cjxl.exe and djxl.exe under tools/libjxl/bin.",
            ),
            djxl: detect_tool(
                "djxl",
                &["--version"],
                "Install JPEG XL tools or place cjxl.exe and djxl.exe under tools/libjxl/bin.",
            ),
            oxipng: detect_tool(
                "oxipng",
                &["--version"],
                "Install oxipng, for example with winget, cargo install, or a project-local binary.",
            ),
            pngquant: detect_tool(
                "pngquant",
                &["--version"],
                "Install pngquant or place pngquant.exe under tools/pngquant/pngquant.",
            ),
            cjpegli: detect_tool(
                "cjpegli",
                &["--version"],
                "Install JPEG XL/jpegli tools or place cjpegli.exe under tools/libjxl/bin.",
            ),
            cjpeg: detect_tool(
                "cjpeg",
                &["-version"],
                "Install mozjpeg/libjpeg tools if you want cjpeg as an emergency JPEG fallback.",
            ),
            ssimulacra2: detect_tool(
                "ssimulacra2",
                &["--help"],
                "Install JPEG XL tools or place ssimulacra2.exe under tools/libjxl/bin.",
            ),
            butteraugli: detect_tool(
                "butteraugli_main",
                &["--help"],
                "Install JPEG XL tools or place butteraugli_main.exe under tools/libjxl/bin.",
            ),
        }
    }

    /// Tools a format needs that were not found: its encoder, and the decoder
    /// that reads a candidate back to score it.
    pub fn missing_for(&self, codec: Codec) -> Vec<&str> {
        let needed: &[&ToolInfo] = match codec {
            Codec::Png | Codec::Jpeg => &[],
            Codec::Webp => &[&self.cwebp],
            Codec::Avif => &[&self.avifenc, &self.avifdec],
            Codec::Jxl => &[&self.cjxl, &self.djxl],
        };
        needed
            .iter()
            .filter(|tool| !tool.enabled)
            .map(|tool| tool.name.as_str())
            .collect()
    }

    pub fn available_for(&self, codec: Codec) -> bool {
        self.missing_for(codec).is_empty()
    }

    fn tool_named(&self, name: &str) -> Option<&ToolInfo> {
        [
            &self.cwebp,
            &self.webpmux,
            &self.avifenc,
            &self.avifdec,
            &self.cjxl,
            &self.djxl,
            &self.oxipng,
            &self.pngquant,
            &self.cjpegli,
            &self.cjpeg,
            &self.ssimulacra2,
            &self.butteraugli,
        ]
        .into_iter()
        .find(|tool| tool.name == name)
    }
}

/// Setting this variable turns every skipped test that needs an external tool
/// into a failure.
const REQUIRE_TOOLS_ENV: &str = "SMART_COMPRESS_REQUIRE_TOOLS";

/// Test support shared by the unit and integration tests: the tool inventory,
/// if every tool named in `needed` (by the name the inventory gives it, such
/// as `cwebp` or `butteraugli_main`) is installed. Otherwise the test skips
/// with one line on stderr naming what is missing, unless
/// `SMART_COMPRESS_REQUIRE_TOOLS` is set, which fails it.
#[doc(hidden)]
pub fn tools_or_skip(needed: &[&str]) -> Option<ToolInventory> {
    tools_or_skip_with(
        ToolInventory::detect(),
        needed,
        std::env::var_os(REQUIRE_TOOLS_ENV).is_some(),
    )
}

fn tools_or_skip_with(
    tools: ToolInventory,
    needed: &[&str],
    required: bool,
) -> Option<ToolInventory> {
    let missing = needed
        .iter()
        .filter(|name| {
            !tools
                .tool_named(name)
                .unwrap_or_else(|| panic!("no tool is named {name}"))
                .enabled
        })
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    if missing.is_empty() {
        return Some(tools);
    }
    let test = std::thread::current().name().unwrap_or("test").to_owned();
    assert!(
        !required,
        "{test}: missing {missing} and {REQUIRE_TOOLS_ENV} is set"
    );
    eprintln!("{test}: skipped, missing {missing}; set {REQUIRE_TOOLS_ENV} to fail instead");
    None
}

/// The tools `tools_or_skip` should be asked for to run `codecs`: their
/// encoders, and the decoders that read a candidate back.
#[doc(hidden)]
pub fn tool_names_for(codecs: &[Codec]) -> Vec<&'static str> {
    codecs
        .iter()
        .flat_map(|codec| match codec {
            Codec::Png | Codec::Jpeg => &[][..],
            Codec::Webp => &["cwebp"][..],
            Codec::Avif => &["avifenc", "avifdec"][..],
            Codec::Jxl => &["cjxl", "djxl"][..],
        })
        .copied()
        .collect()
}

#[derive(Debug)]
pub struct EncodeJob<'a> {
    pub codec: Codec,
    pub settings: EncoderSettings,
    pub normalized_png: &'a Path,
    pub original: &'a DynamicImage,
    pub output_path: &'a Path,
    pub timeout: Duration,
    pub has_alpha: bool,
    pub metadata: &'a MetadataPayload,
    /// Worker threads an external encoder may use for this job.
    pub threads: usize,
}

#[derive(Debug, Clone, Default)]
pub struct MetadataPayload {
    pub icc_profile: Option<Vec<u8>>,
    pub exif_metadata: Option<Vec<u8>>,
}

/// Colour space an ICC profile describes (`RGB `, `GRAY`, `CMYK`, ...), or
/// `None` when the bytes are not an ICC profile at all.
pub fn icc_color_space(profile: &[u8]) -> Option<[u8; 4]> {
    (profile.len() >= 128 && &profile[36..40] == b"acsp")
        .then(|| [profile[16], profile[17], profile[18], profile[19]])
}

/// Whether a profile provably describes plain sRGB, so that a viewer which
/// finds no profile shows the same colours. Only matrix/TRC (RGB) and
/// single-curve (grey) profiles can be shown to be sRGB; anything else,
/// including every profile with lookup tables or that cannot be parsed, is
/// reported as not sRGB, which keeps it. The primaries must match sRGB to
/// 0.4% of XYZ and each curve the sRGB curve to 0.01 of linear light, so
/// gamma 2.2 stand-ins count as sRGB and gamma 1.8, 2.4, Display P3 and
/// Adobe RGB do not.
pub fn icc_is_plain_srgb(profile: &[u8]) -> bool {
    const SRGB_D50: [[f64; 3]; 3] = [
        [0.4361, 0.2225, 0.0139],
        [0.3851, 0.7169, 0.0971],
        [0.1431, 0.0606, 0.7141],
    ];
    let srgb_curve = |x: f64| {
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    };
    let curve_matches = |signature: &[u8; 4]| {
        icc_tag(profile, signature).is_some_and(|curve| {
            [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
                .iter()
                .all(|&x| icc_curve_at(curve, x).is_some_and(|y| (y - srgb_curve(x)).abs() <= 0.01))
        })
    };
    if icc_tag(profile, b"A2B0").is_some() {
        return false;
    }
    match icc_color_space(profile).as_ref() {
        Some(b"GRAY") => curve_matches(b"kTRC"),
        Some(b"RGB ") => {
            let colorant = |signature: &[u8; 4]| {
                let data = icc_tag(profile, signature)?;
                let value = |at: usize| {
                    Some(
                        f64::from(i32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?))
                            / 65536.0,
                    )
                };
                (data.starts_with(b"XYZ ")).then_some([value(8)?, value(12)?, value(16)?])
            };
            [b"rXYZ", b"gXYZ", b"bXYZ"]
                .iter()
                .zip(SRGB_D50)
                .all(|(signature, expected)| {
                    colorant(signature).is_some_and(|actual| {
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(a, e)| (a - e).abs() <= 0.004)
                    })
                })
                && [b"rTRC", b"gTRC", b"bTRC"].iter().all(|s| curve_matches(s))
        }
        _ => false,
    }
}

/// The data of one tag of an ICC profile.
fn icc_tag<'a>(profile: &'a [u8], signature: &[u8; 4]) -> Option<&'a [u8]> {
    let word = |at: usize| {
        Some(u32::from_be_bytes(
            profile.get(at..at + 4)?.try_into().ok()?,
        ))
    };
    (0..word(128)?.min(256) as usize).find_map(|index| {
        let entry = 132 + 12 * index;
        if profile.get(entry..entry + 4)? != signature {
            return None;
        }
        let offset = word(entry + 4)? as usize;
        profile.get(offset..offset.checked_add(word(entry + 8)? as usize)?)
    })
}

/// Value of a `curv` or `para` tag at `x` in 0..=1.
fn icc_curve_at(data: &[u8], x: f64) -> Option<f64> {
    let half = |at: usize| {
        Some(f64::from(u16::from_be_bytes(
            data.get(at..at + 2)?.try_into().ok()?,
        )))
    };
    let fixed = |at: usize| {
        Some(f64::from(i32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?)) / 65536.0)
    };
    match data.get(..4)? {
        b"curv" => {
            let count = u32::from_be_bytes(data.get(8..12)?.try_into().ok()?) as usize;
            match count {
                0 => Some(x),
                1 => Some(x.powf(half(12)? / 256.0)),
                _ => {
                    let position = x * (count - 1) as f64;
                    let low = position.floor() as usize;
                    let at = |index: usize| Some(half(12 + 2 * index)? / 65535.0);
                    let (a, b) = (at(low)?, at((low + 1).min(count - 1))?);
                    Some(a + (b - a) * (position - low as f64))
                }
            }
        }
        b"para" => {
            let function = u16::from_be_bytes(data.get(8..10)?.try_into().ok()?);
            let [g, a, b, c, d, e, f] =
                [0, 1, 2, 3, 4, 5, 6].map(|i| fixed(12 + 4 * i).unwrap_or(0.0));
            Some(match function {
                0 => x.powf(g),
                1 if x >= -b / a => (a * x + b).powf(g),
                1 => 0.0,
                2 if x >= -b / a => (a * x + b).powf(g) + c,
                2 => c,
                3 if x >= d => (a * x + b).powf(g),
                3 => c * x,
                4 if x >= d => (a * x + b).powf(g) + e,
                4 => c * x + f,
                _ => return None,
            })
        }
        _ => None,
    }
}

impl MetadataPayload {
    pub fn is_empty(&self) -> bool {
        self.icc_profile.is_none() && self.exif_metadata.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedArtifact {
    pub codec: Codec,
    pub settings: EncoderSettings,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub encode_ms: u64,
    pub backend: String,
    pub warnings: Vec<String>,
}

#[derive(Debug, Error)]
pub enum EncodeError {
    #[error("encoder unavailable for {0:?}")]
    Unavailable(Codec),
    #[error("encoder failed for {codec:?}: {message}")]
    Failed { codec: Codec, message: String },
    /// The encoder made no file because it had nothing worth making, which is
    /// no failure: the candidate simply does not exist.
    #[error("{0}")]
    Declined(String),
}

pub struct EncoderBackend {
    tools: ToolInventory,
}

impl EncoderBackend {
    pub fn new(tools: ToolInventory) -> Self {
        Self { tools }
    }

    pub fn tools(&self) -> &ToolInventory {
        &self.tools
    }

    pub fn encode(&self, job: EncodeJob<'_>) -> Result<EncodedArtifact> {
        if job.has_alpha && !job.codec.supports_alpha() {
            return Err(anyhow!(EncodeError::Failed {
                codec: job.codec,
                message: "codec cannot preserve alpha".to_string(),
            }));
        }

        match job.codec {
            Codec::Png => self.encode_png(job),
            Codec::Jpeg => self.encode_jpeg(job),
            Codec::Webp => self.encode_webp(job),
            Codec::Avif => self.encode_avif(job),
            Codec::Jxl => self.encode_jxl(job),
        }
    }

    fn encode_png(&self, job: EncodeJob<'_>) -> Result<EncodedArtifact> {
        let start = Instant::now();
        if let EncoderSettings::Palette { colors } = job.settings {
            if !job.metadata.is_empty() {
                return Err(anyhow!(EncodeError::Declined(
                    "pngquant palette mode is skipped when metadata must be preserved".to_string()
                )));
            }
            let Some(path) = &self.tools.pngquant.path else {
                return Err(anyhow!(EncodeError::Unavailable(Codec::Png)));
            };
            let run = run_command(
                path,
                &[
                    OsString::from("--force"),
                    OsString::from("--skip-if-larger"),
                    OsString::from("--output"),
                    job.output_path.as_os_str().to_os_string(),
                    OsString::from(colors.to_string()),
                    job.normalized_png.as_os_str().to_os_string(),
                ],
                job.timeout,
                1,
            );
            // pngquant exits with 98 when the palette file would be larger than
            // the input (`--skip-if-larger`) and with 99 when it cannot reach
            // the quality asked for: it made no file, and that is its answer.
            if let Err(error) = run {
                return Err(match error.downcast_ref::<ProcessFailure>() {
                    Some(failure)
                        if failure.kind == ProcessFailureKind::Exit
                            && matches!(failure.exit_code, Some(98 | 99)) =>
                    {
                        anyhow!(EncodeError::Declined(format!(
                            "pngquant made no {colors}-colour file: {failure}"
                        )))
                    }
                    _ => error,
                });
            }
            return artifact(job, "pngquant", start.elapsed(), Vec::new());
        }

        {
            let _unit = work_unit(job.timeout)?;
            write_png(job.output_path, job.original, job.metadata)?;
        }

        let mut warnings = Vec::new();
        if !job.metadata.is_empty() {
            warnings.push("oxipng skipped to preserve embedded metadata".to_string());
        } else if self.tools.oxipng.enabled
            && let Some(path) = &self.tools.oxipng.path
        {
            let result = run_command(
                path,
                &[
                    OsString::from("-o"),
                    OsString::from("2"),
                    OsString::from("--strip"),
                    OsString::from("safe"),
                    OsString::from("--threads"),
                    OsString::from(job.threads.max(1).to_string()),
                    job.output_path.as_os_str().to_os_string(),
                ],
                job.timeout,
                job.threads,
            );
            if let Err(error) = result {
                warnings.push(format!("oxipng skipped: {error}"));
            }
        }

        artifact(job, "native-png+optional-oxipng", start.elapsed(), warnings)
    }

    fn encode_jpeg(&self, job: EncodeJob<'_>) -> Result<EncodedArtifact> {
        let EncoderSettings::Quality { quality, .. } = job.settings else {
            return Err(anyhow!("JPEG requires quality settings"));
        };

        let start = Instant::now();
        // cjpegli carries the ICC profile over from the normalized PNG but
        // cannot attach EXIF, so retained EXIF needs the native encoder.
        let mut warnings = Vec::new();
        if job.metadata.exif_metadata.is_none()
            && let Some(path) = &self.tools.cjpegli.path
        {
            match run_command(
                path,
                &[
                    job.normalized_png.as_os_str().to_os_string(),
                    job.output_path.as_os_str().to_os_string(),
                    OsString::from(format!("--quality={quality}")),
                    OsString::from("--progressive_level=2"),
                ],
                job.timeout,
                1,
            ) {
                Ok(()) => return artifact(job, "cjpegli", start.elapsed(), Vec::new()),
                Err(error) if cancellation_requested() => return Err(error),
                // cjpegli refuses inputs it cannot interpret, such as a
                // damaged colour profile; the built-in encoder still can.
                Err(error) => warnings.push(format!(
                    "cjpegli failed, used the built-in JPEG encoder: {error}"
                )),
            }
        }

        // The JPEG encoder emits tiny writes; unbuffered they dominate encode time.
        let unit = work_unit(job.timeout)?;
        let mut file = BufWriter::new(
            fs::File::create(job.output_path)
                .with_context(|| format!("creating {}", job.output_path.display()))?,
        );
        let mut encoder = JpegEncoder::new_with_quality(&mut file, quality);
        apply_metadata(&mut encoder, job.metadata)?;
        // A grey source stays single-channel so a grey ICC profile still fits
        // it. The buffers are passed concretely: through `DynamicImage` the
        // encoder always sees RGBA and writes three components.
        if job.original.color().has_color() {
            encoder.encode_image(&job.original.to_rgb8())
        } else {
            encoder.encode_image(&job.original.to_luma8())
        }
        .with_context(|| format!("encoding JPEG {}", job.output_path.display()))?;
        file.flush()
            .with_context(|| format!("flushing JPEG {}", job.output_path.display()))?;
        drop(file);
        drop(unit);
        let backend = if job.metadata.is_empty() {
            "native-jpeg"
        } else {
            "native-jpeg+metadata"
        };
        artifact(job, backend, start.elapsed(), warnings)
    }

    fn encode_webp(&self, job: EncodeJob<'_>) -> Result<EncodedArtifact> {
        let Some(path) = &self.tools.cwebp.path else {
            return Err(anyhow!(EncodeError::Unavailable(Codec::Webp)));
        };

        let mut args = Vec::new();
        match job.settings {
            EncoderSettings::Lossless => {
                args.push(OsString::from("-lossless"));
                args.push(OsString::from("-m"));
                args.push(OsString::from("4"));
            }
            EncoderSettings::Quality { quality, effort } => {
                args.push(OsString::from("-q"));
                args.push(OsString::from(quality.to_string()));
                args.push(OsString::from("-m"));
                args.push(OsString::from(effort.to_string()));
            }
            EncoderSettings::Palette { .. } => {
                return Err(anyhow!("WebP does not use palette settings"));
            }
        }
        if job.threads >= 2 {
            // Multithreading does not change the bitstream.
            args.push(OsString::from("-mt"));
        }
        args.push(job.normalized_png.as_os_str().to_os_string());
        args.push(OsString::from("-o"));
        args.push(job.output_path.as_os_str().to_os_string());

        let mut metadata = job.metadata.clone();
        let mut warnings = Vec::new();
        // WebP always stores colour, so a grey profile would no longer match.
        // Without it a viewer assumes sRGB, which is only the same picture when
        // the profile is plain sRGB.
        if let Some(profile) = metadata
            .icc_profile
            .as_deref()
            .filter(|profile| icc_color_space(profile).as_ref() == Some(b"GRAY"))
        {
            anyhow::ensure!(
                icc_is_plain_srgb(profile),
                "WebP stores colour only and cannot carry this grey ICC profile, which is not plain sRGB"
            );
            metadata.icc_profile = None;
            warnings.push("grey ICC profile not attached: WebP stores colour only".to_string());
        }
        let start = Instant::now();
        run_command(path, &args, job.timeout, job.threads.clamp(1, 2))?;
        if !metadata.is_empty() {
            self.attach_webp_metadata(job.output_path, &metadata, job.timeout)?;
        }
        artifact(job, "cwebp", start.elapsed(), warnings)
    }

    fn encode_avif(&self, job: EncodeJob<'_>) -> Result<EncodedArtifact> {
        let Some(path) = &self.tools.avifenc.path else {
            return Err(anyhow!(EncodeError::Unavailable(Codec::Avif)));
        };

        let mut args = Vec::new();
        match job.settings {
            EncoderSettings::Lossless => args.push(OsString::from("--lossless")),
            EncoderSettings::Quality { quality, effort } => {
                args.push(OsString::from("-q"));
                args.push(OsString::from(quality.to_string()));
                args.push(OsString::from("--speed"));
                args.push(OsString::from(effort.to_string()));
            }
            EncoderSettings::Palette { .. } => {
                return Err(anyhow!("AVIF does not use palette settings"));
            }
        }
        // One job produces a different bitstream than two or more, so stay
        // at two or above to keep results independent of the thread budget.
        args.push(OsString::from("-j"));
        args.push(OsString::from(job.threads.max(2).to_string()));
        args.push(job.normalized_png.as_os_str().to_os_string());
        args.push(job.output_path.as_os_str().to_os_string());

        let start = Instant::now();
        run_command(path, &args, job.timeout, job.threads.max(2))?;
        artifact(job, "avifenc", start.elapsed(), Vec::new())
    }

    fn encode_jxl(&self, job: EncodeJob<'_>) -> Result<EncodedArtifact> {
        let Some(path) = &self.tools.cjxl.path else {
            return Err(anyhow!(EncodeError::Unavailable(Codec::Jxl)));
        };

        let mut args = Vec::new();
        match job.settings {
            EncoderSettings::Lossless => {
                // Without an explicit distance cjxl encodes lossily at d=1.
                args.push(OsString::from("-d"));
                args.push(OsString::from("0"));
                args.push(OsString::from("--lossless_jpeg=0"));
            }
            EncoderSettings::Quality { quality, effort } => {
                args.push(OsString::from("-q"));
                args.push(OsString::from(quality.to_string()));
                args.push(OsString::from("-e"));
                args.push(OsString::from(effort.to_string()));
                args.push(OsString::from("--lossless_jpeg=0"));
                args.push(OsString::from("-m"));
                args.push(OsString::from("0"));
            }
            EncoderSettings::Palette { .. } => {
                return Err(anyhow!("JXL does not use palette settings"));
            }
        }
        args.push(OsString::from(format!(
            "--num_threads={}",
            job.threads.max(1)
        )));
        args.push(job.normalized_png.as_os_str().to_os_string());
        args.push(job.output_path.as_os_str().to_os_string());

        let start = Instant::now();
        run_command(path, &args, job.timeout, job.threads)?;
        artifact(job, "cjxl", start.elapsed(), Vec::new())
    }

    fn attach_webp_metadata(
        &self,
        output_path: &Path,
        metadata: &MetadataPayload,
        timeout: Duration,
    ) -> Result<()> {
        let Some(webpmux) = self.tools.webpmux.path.clone() else {
            anyhow::bail!("webpmux not found; cannot preserve WebP metadata safely");
        };

        let temp = tempfile::tempdir()?;
        let mut current = output_path.to_path_buf();
        let mut stage_index = 0_u8;

        if let Some(icc_profile) = &metadata.icc_profile {
            let icc_path = temp.path().join("profile.icc");
            fs::write(&icc_path, icc_profile)
                .with_context(|| format!("writing {}", icc_path.display()))?;
            stage_index += 1;
            let staged = temp.path().join(format!("stage-{stage_index}.webp"));
            run_command(
                &webpmux,
                &[
                    OsString::from("-set"),
                    OsString::from("icc"),
                    icc_path.as_os_str().to_os_string(),
                    current.as_os_str().to_os_string(),
                    OsString::from("-o"),
                    staged.as_os_str().to_os_string(),
                ],
                timeout,
                1,
            )?;
            current = staged;
        }

        if let Some(exif_metadata) = &metadata.exif_metadata {
            let exif_path = temp.path().join("metadata.exif");
            fs::write(&exif_path, exif_metadata)
                .with_context(|| format!("writing {}", exif_path.display()))?;
            stage_index += 1;
            let staged = temp.path().join(format!("stage-{stage_index}.webp"));
            run_command(
                &webpmux,
                &[
                    OsString::from("-set"),
                    OsString::from("exif"),
                    exif_path.as_os_str().to_os_string(),
                    current.as_os_str().to_os_string(),
                    OsString::from("-o"),
                    staged.as_os_str().to_os_string(),
                ],
                timeout,
                1,
            )?;
            current = staged;
        }

        if current != output_path {
            fs::copy(&current, output_path).with_context(|| {
                format!("copying WebP metadata output to {}", output_path.display())
            })?;
        }
        Ok(())
    }
}

pub fn normalized_png(
    image: &DynamicImage,
    metadata: &MetadataPayload,
) -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("normalized-input.png");
    write_png(&path, image, metadata)?;
    Ok((dir, path))
}

fn write_png(path: &Path, image: &DynamicImage, metadata: &MetadataPayload) -> Result<()> {
    let file = fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = PngEncoder::new(file);
    apply_metadata(&mut encoder, metadata)?;
    image
        .write_with_encoder(encoder)
        .with_context(|| format!("writing PNG {}", path.display()))
}

fn apply_metadata<E: ImageEncoder>(encoder: &mut E, metadata: &MetadataPayload) -> Result<()> {
    if let Some(icc_profile) = &metadata.icc_profile {
        encoder
            .set_icc_profile(icc_profile.clone())
            .map_err(anyhow::Error::from)?;
    }
    if let Some(exif_metadata) = &metadata.exif_metadata {
        encoder
            .set_exif_metadata(exif_metadata.clone())
            .map_err(anyhow::Error::from)?;
    }
    Ok(())
}

fn artifact(
    job: EncodeJob<'_>,
    backend: &str,
    elapsed: Duration,
    warnings: Vec<String>,
) -> Result<EncodedArtifact> {
    let size_bytes = fs::metadata(job.output_path)
        .with_context(|| format!("reading metadata for {}", job.output_path.display()))?
        .len();
    Ok(EncodedArtifact {
        codec: job.codec,
        settings: job.settings,
        path: job.output_path.to_path_buf(),
        size_bytes,
        encode_ms: elapsed.as_millis() as u64,
        backend: backend.to_string(),
        warnings,
    })
}

fn detect_tool(name: &str, version_args: &[&str], install_hint: &str) -> ToolInfo {
    let path = find_in_path(name);
    let version = path.as_ref().and_then(|tool_path| {
        let args = version_args.iter().map(OsString::from).collect::<Vec<_>>();
        run_process(tool_path, &args, Duration::from_secs(5))
            .map(|output| {
                let text = if output.stdout.is_empty() {
                    String::from_utf8_lossy(&output.stderr).to_string()
                } else {
                    String::from_utf8_lossy(&output.stdout).to_string()
                };
                text.lines()
                    .next()
                    .unwrap_or("version unavailable")
                    .to_string()
            })
            .ok()
    });

    let enabled = path.is_some();
    ToolInfo {
        name: name.to_string(),
        enabled,
        path,
        version,
        install_hint: (!enabled).then(|| install_hint.to_string()),
    }
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let extensions: Vec<String> = if cfg!(windows) {
        std::env::var_os("PATHEXT")
            .map(|value| {
                std::env::split_paths(&value)
                    .filter_map(|path| path.to_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_else(|| vec![".EXE".to_string(), ".BAT".to_string(), ".CMD".to_string()])
    } else {
        vec!["".to_string()]
    };

    let mut search_dirs = std::env::var_os("PATH")
        .map(|path_var| absolute_path_dirs(&path_var))
        .unwrap_or_default();
    search_dirs.extend(project_tool_dirs());

    for dir in search_dirs {
        let direct = dir.join(name);
        if direct.is_file() {
            return Some(direct);
        }
        for ext in &extensions {
            let candidate = dir.join(format!("{name}{ext}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// The absolute entries of a PATH value. An empty or relative entry means the
/// working directory, which would run whatever `cjxl.exe` an untrusted folder
/// happened to contain.
fn absolute_path_dirs(path_var: &OsStr) -> Vec<PathBuf> {
    std::env::split_paths(path_var)
        .filter(|dir| dir.is_absolute())
        .collect()
}

/// Tool folders bundled with the program: those beside the executable and,
/// for a build inside this workspace, those of the workspace. Nothing else
/// above the executable is searched, since any user can create a `tools` folder
/// in the root of a drive.
fn project_tool_dirs() -> Vec<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(tool_dirs_for))
        .unwrap_or_default()
}

fn tool_dirs_for(exe_dir: &Path) -> Vec<PathBuf> {
    let mut roots = vec![exe_dir];
    // A development build lives in `<workspace>/target/...`.
    if let Some(workspace) = exe_dir.ancestors().find(|dir| {
        dir.join("Cargo.toml").is_file()
            && dir.join("tools").is_dir()
            && exe_dir
                .strip_prefix(dir)
                .is_ok_and(|rest| rest.starts_with("target"))
    }) {
        roots.push(workspace);
    }

    let mut dirs = Vec::new();
    for root in roots {
        for relative in [
            "tools",
            "tools/bin",
            "tools/libjxl/bin",
            "tools/libavif",
            "tools/libwebp/libwebp-1.6.0-windows-x64/bin",
            "tools/pngquant/pngquant",
        ] {
            let dir = root.join(relative);
            if dir.is_dir() && !dirs.contains(&dir) {
                dirs.push(dir);
            }
        }
    }
    dirs
}

const MAX_PROCESS_OUTPUT_BYTES: usize = 64 * 1024;

static CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);

pub fn request_cancel() {
    CANCEL_REQUESTED.store(true, Ordering::SeqCst);
    work_budget().wake();
}

pub fn clear_cancel() {
    CANCEL_REQUESTED.store(false, Ordering::SeqCst);
}

pub fn cancellation_requested() -> bool {
    CANCEL_REQUESTED.load(Ordering::SeqCst)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessFailureKind {
    Start,
    Poll,
    Timeout,
    Cancelled,
    Exit,
    Output,
}

#[derive(Debug, Error)]
#[error("{} {kind:?} after {elapsed_ms} ms{exit_suffix}: {message}", .executable.file_name().unwrap_or_default().to_string_lossy())]
pub struct ProcessFailure {
    pub executable: PathBuf,
    pub kind: ProcessFailureKind,
    pub exit_code: Option<i32>,
    pub message: String,
    pub elapsed_ms: u64,
    pub termination_requested: bool,
    exit_suffix: String,
}

impl ProcessFailure {
    fn new(
        executable: &Path,
        kind: ProcessFailureKind,
        exit_code: Option<i32>,
        message: String,
        elapsed_ms: u64,
        termination_requested: bool,
    ) -> Self {
        let exit_suffix = exit_code.map_or(String::new(), |code| format!(" (exit {code})"));
        Self {
            executable: executable.to_path_buf(),
            kind,
            exit_code,
            message,
            elapsed_ms,
            termination_requested,
            exit_suffix,
        }
    }
}

/// Why waiting for a budget ended without a grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum WaitEnded {
    #[error("cancelled while waiting for the work budget")]
    Cancelled,
    #[error("timed out while waiting for the work budget")]
    TimedOut,
}

#[derive(Debug)]
struct BudgetState {
    limit: u64,
    used: u64,
    /// Tickets of the requests that are waiting, oldest first.
    queue: VecDeque<u64>,
    next_ticket: u64,
}

/// A counted budget whose requests are served in arrival order, so a large
/// request is not starved by a stream of small ones. A request larger than the
/// whole budget is clamped to it and runs alone. Nothing is held while waiting.
///
/// The rule that keeps it deadlock free is the caller's: a permit is only held
/// by work that finishes without asking for another permit. File workers and
/// anything else that waits on other work hold none.
#[derive(Debug)]
pub struct Budget {
    state: Mutex<BudgetState>,
    changed: Condvar,
}

/// Waiters re-check cancellation and their deadline at least this often, so a
/// wake-up that nobody sent cannot strand them.
const WAIT_SLICE: Duration = Duration::from_millis(25);

impl Budget {
    pub const fn new(limit: u64) -> Self {
        Self {
            state: Mutex::new(BudgetState {
                limit: if limit == 0 { 1 } else { limit },
                used: 0,
                queue: VecDeque::new(),
                next_ticket: 0,
            }),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, BudgetState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn wait_slice<'a>(
        &self,
        state: MutexGuard<'a, BudgetState>,
        deadline: Option<Instant>,
    ) -> MutexGuard<'a, BudgetState> {
        let slice = deadline.map_or(WAIT_SLICE, |deadline| {
            deadline
                .saturating_duration_since(Instant::now())
                .min(WAIT_SLICE)
        });
        self.changed
            .wait_timeout(state, slice)
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .0
    }

    pub fn set_limit(&self, limit: u64) {
        self.lock().limit = limit.max(1);
        self.changed.notify_all();
    }

    pub fn limit(&self) -> u64 {
        self.lock().limit
    }

    pub fn used(&self) -> u64 {
        self.lock().used
    }

    /// Makes every waiter re-check cancellation, its deadline and its condition.
    pub fn wake(&self) {
        drop(self.lock());
        self.changed.notify_all();
    }

    /// Waits for `units` (at least one, at most the whole budget), until the
    /// deadline passes, or until cancellation is requested.
    pub fn acquire(&self, units: u64, deadline: Option<Instant>) -> Result<Permit<'_>, WaitEnded> {
        self.acquire_unless(units, deadline, cancellation_requested)
    }

    pub fn acquire_unless(
        &self,
        units: u64,
        deadline: Option<Instant>,
        stop: impl Fn() -> bool,
    ) -> Result<Permit<'_>, WaitEnded> {
        let mut state = self.lock();
        let ticket = state.next_ticket;
        state.next_ticket += 1;
        state.queue.push_back(ticket);
        loop {
            let ended = if stop() {
                Some(WaitEnded::Cancelled)
            } else if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                Some(WaitEnded::TimedOut)
            } else {
                None
            };
            let need = units.clamp(1, state.limit);
            if ended.is_none()
                && state.queue.front() == Some(&ticket)
                && state.limit.saturating_sub(state.used) >= need
            {
                state.queue.pop_front();
                state.used += need;
                // Room may be left for the request behind this one.
                self.changed.notify_all();
                return Ok(Permit {
                    budget: self,
                    units: need,
                });
            }
            if let Some(ended) = ended {
                state.queue.retain(|queued| *queued != ticket);
                self.changed.notify_all();
                return Err(ended);
            }
            state = self.wait_slice(state, deadline);
        }
    }

    /// Waits until `blocked` is false, for conditions that are not a count,
    /// such as temporary files in use. `wake` makes a waiter check sooner.
    pub fn wait_while(
        &self,
        deadline: Option<Instant>,
        blocked: impl Fn() -> bool,
    ) -> Result<(), WaitEnded> {
        let mut state = self.lock();
        loop {
            if cancellation_requested() {
                return Err(WaitEnded::Cancelled);
            }
            if !blocked() {
                return Ok(());
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(WaitEnded::TimedOut);
            }
            state = self.wait_slice(state, deadline);
        }
    }
}

/// Units taken from a [`Budget`], returned when dropped.
#[derive(Debug)]
pub struct Permit<'a> {
    budget: &'a Budget,
    units: u64,
}

impl Permit<'_> {
    pub fn units(&self) -> u64 {
        self.units
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut state = self.budget.lock();
        state.used = state.used.saturating_sub(self.units);
        drop(state);
        self.budget.changed.notify_all();
    }
}

/// The budget of threads of work shared by everything this process runs:
/// external tools take as many units as the threads they are told to use,
/// in-process encode, decode and metric work the threads they use, and a GPU
/// submission one.
fn work_budget() -> &'static Budget {
    static WORK: Budget = Budget::new(u64::MAX);
    &WORK
}

pub fn configure_work_budget(units: usize) {
    work_budget().set_limit(units as u64);
}

/// Takes `units` from the work budget for one piece of work that does not
/// itself wait for another permit.
pub fn acquire_work(units: usize, deadline: Option<Instant>) -> Result<Permit<'static>, WaitEnded> {
    work_budget().acquire(units as u64, deadline)
}

#[derive(Debug)]
struct CapturedStream {
    bytes: Vec<u8>,
    truncated: bool,
}

#[derive(Debug, Clone)]
pub struct ProcessOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub elapsed_ms: u64,
}

fn run_command(
    executable: &Path,
    args: &[OsString],
    timeout: Duration,
    threads: usize,
) -> Result<()> {
    run_process_threads(executable, args, timeout, threads).map(|_| ())
}

/// One unit of the work budget for an in-process step that runs on a single thread.
fn work_unit(timeout: Duration) -> Result<Permit<'static>> {
    Ok(acquire_work(1, Some(Instant::now() + timeout))?)
}

/// Runs a single-threaded tool.
pub fn run_process(
    executable: &Path,
    args: &[OsString],
    timeout: Duration,
) -> Result<ProcessOutput> {
    run_process_threads(executable, args, timeout, 1)
}

/// Runs a tool that has been told to use `threads` threads, which is what it
/// takes from the work budget. The wait for the budget is bounded by the same
/// timeout as the run.
pub fn run_process_threads(
    executable: &Path,
    args: &[OsString],
    timeout: Duration,
    threads: usize,
) -> Result<ProcessOutput> {
    let waiting = Instant::now();
    let _permit = acquire_work(threads, Some(waiting + timeout)).map_err(|ended| {
        anyhow!(ProcessFailure::new(
            executable,
            match ended {
                WaitEnded::Cancelled => ProcessFailureKind::Cancelled,
                WaitEnded::TimedOut => ProcessFailureKind::Timeout,
            },
            None,
            ended.to_string(),
            waiting.elapsed().as_millis() as u64,
            true,
        ))
    })?;
    let start = Instant::now();
    let mut command = Command::new(executable);
    command.args(args);
    // Tools get their own folder as working directory, not the caller's, so
    // a DLL they fail to find beside themselves is not picked up from there.
    // Every path in `args` is therefore absolute.
    if let Some(dir) = executable
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
    {
        command.current_dir(dir);
    }
    let mut child = quiet(&mut command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            anyhow!(ProcessFailure::new(
                executable,
                ProcessFailureKind::Start,
                None,
                error.to_string(),
                start.elapsed().as_millis() as u64,
                false,
            ))
        })?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("capturing stdout from {}", executable.display()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("capturing stderr from {}", executable.display()))?;
    let stdout_reader = spawn_capped_reader(stdout);
    let stderr_reader = spawn_capped_reader(stderr);

    let (status, timed_out) = loop {
        if cancellation_requested() {
            terminate_process_tree(child.id());
            let _ = child.kill();
            let status = child
                .wait()
                .with_context(|| format!("waiting for cancelled {}", executable.display()))?;
            let stdout = join_capped_reader(stdout_reader)
                .with_context(|| format!("reading stdout from {}", executable.display()))?;
            let stderr = join_capped_reader(stderr_reader)
                .with_context(|| format!("reading stderr from {}", executable.display()))?;
            return Err(anyhow!(ProcessFailure::new(
                executable,
                ProcessFailureKind::Cancelled,
                status.code(),
                useful_output(&stderr, &stdout),
                start.elapsed().as_millis() as u64,
                true,
            )));
        }
        match child.try_wait() {
            Ok(Some(status)) => break (status, false),
            // Poll quickly at first: on small images most tools exit within
            // a few milliseconds and a fixed wait would dominate.
            Ok(None) if start.elapsed() <= timeout => thread::sleep(Duration::from_millis(
                (1 + start.elapsed().as_millis() as u64 / 10).min(20),
            )),
            Ok(None) => {
                terminate_process_tree(child.id());
                let _ = child.kill();
                let status = child
                    .wait()
                    .with_context(|| format!("waiting for timed-out {}", executable.display()))?;
                break (status, true);
            }
            Err(error) => {
                terminate_process_tree(child.id());
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_capped_reader(stdout_reader);
                let _ = join_capped_reader(stderr_reader);
                return Err(anyhow!(ProcessFailure::new(
                    executable,
                    ProcessFailureKind::Poll,
                    None,
                    error.to_string(),
                    start.elapsed().as_millis() as u64,
                    true,
                )));
            }
        }
    };

    let stdout = join_capped_reader(stdout_reader)
        .with_context(|| format!("reading stdout from {}", executable.display()))?;
    let stderr = join_capped_reader(stderr_reader)
        .with_context(|| format!("reading stderr from {}", executable.display()))?;

    if timed_out {
        return Err(anyhow!(ProcessFailure::new(
            executable,
            ProcessFailureKind::Timeout,
            status.code(),
            useful_output(&stderr, &stdout),
            start.elapsed().as_millis() as u64,
            true,
        )));
    }
    if status.success() {
        return Ok(ProcessOutput {
            stdout: stdout.bytes,
            stderr: stderr.bytes,
            stdout_truncated: stdout.truncated,
            stderr_truncated: stderr.truncated,
            elapsed_ms: start.elapsed().as_millis() as u64,
        });
    }

    Err(anyhow!(ProcessFailure::new(
        executable,
        ProcessFailureKind::Exit,
        status.code(),
        useful_output(&stderr, &stdout),
        start.elapsed().as_millis() as u64,
        false,
    )))
}

/// Keeps console tools from opening a window of their own when the caller
/// has no console, as the windowed control panel does. Their output is piped
/// either way.
pub fn quiet(command: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

/// A program of the Windows folder by absolute path, so a file of the same
/// name in the working directory or on a `PATH` entry is never started. Only
/// meaningful on Windows, the only place it is called.
pub fn system_program(relative: &str) -> PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        std::env::var_os("SystemRoot")
            .map(PathBuf::from)
            .filter(|root| root.is_absolute())
            .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
    })
    .join(relative)
}

#[cfg(windows)]
fn terminate_process_tree(process_id: u32) {
    let _ = quiet(
        Command::new(system_program(r"System32\taskkill.exe")).args([
            "/PID",
            &process_id.to_string(),
            "/T",
            "/F",
        ]),
    )
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .status();
}

#[cfg(not(windows))]
fn terminate_process_tree(_process_id: u32) {}

fn spawn_capped_reader<R>(reader: R) -> JoinHandle<std::io::Result<CapturedStream>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || read_capped(reader, MAX_PROCESS_OUTPUT_BYTES))
}

fn join_capped_reader(
    reader: JoinHandle<std::io::Result<CapturedStream>>,
) -> Result<CapturedStream> {
    reader
        .join()
        .map_err(|_| anyhow!("process output reader panicked"))?
        .map_err(anyhow::Error::from)
}

fn read_capped(mut reader: impl Read, limit: usize) -> std::io::Result<CapturedStream> {
    let mut bytes = Vec::with_capacity(limit.min(8 * 1024));
    let mut buffer = [0_u8; 8 * 1024];
    let mut truncated = false;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let remaining = limit.saturating_sub(bytes.len());
        let retained = read.min(remaining);
        bytes.extend_from_slice(&buffer[..retained]);
        truncated |= retained < read;
    }
    Ok(CapturedStream { bytes, truncated })
}

fn output_suffix(stderr: &CapturedStream, stdout: &CapturedStream) -> String {
    let (message, truncated) = first_output_line(stderr)
        .map(|message| (message, stderr.truncated))
        .or_else(|| first_output_line(stdout).map(|message| (message, stdout.truncated)))
        .unwrap_or_else(|| {
            (
                "process failed".to_string(),
                stderr.truncated || stdout.truncated,
            )
        });
    if truncated {
        format!(": {message} [output truncated]")
    } else {
        format!(": {message}")
    }
}

fn useful_output(stderr: &CapturedStream, stdout: &CapturedStream) -> String {
    output_suffix(stderr, stdout)
        .strip_prefix(": ")
        .unwrap_or("process failed")
        .to_string()
}

fn first_output_line(output: &CapturedStream) -> Option<String> {
    String::from_utf8_lossy(&output.bytes)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn a_test_without_its_tools_skips_or_fails_when_tools_are_required() {
        let mut tools = ToolInventory::detect();
        tools.cwebp.enabled = true;
        tools.cjxl.enabled = false;
        tools.djxl.enabled = false;
        assert!(tools_or_skip_with(tools.clone(), &["cwebp"], true).is_some());
        assert!(tools_or_skip_with(tools.clone(), &[], true).is_some());
        assert!(tools_or_skip_with(tools.clone(), &["cwebp", "cjxl"], false).is_none());
        let failure = std::panic::catch_unwind(|| {
            let mut tools = ToolInventory::detect();
            tools.cjxl.enabled = false;
            tools.djxl.enabled = false;
            tools_or_skip_with(tools, &["cjxl", "djxl"], true)
        })
        .expect_err("a missing tool must fail the test when tools are required");
        let message = failure
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default();
        assert!(message.contains("missing cjxl, djxl"), "{message}");
        assert!(message.contains(REQUIRE_TOOLS_ENV), "{message}");
        assert_eq!(
            tool_names_for(&[Codec::Png, Codec::Webp, Codec::Avif, Codec::Jxl]),
            ["cwebp", "avifenc", "avifdec", "cjxl", "djxl"]
        );
    }

    #[test]
    fn a_format_needs_its_decoder_as_well_as_its_encoder() {
        let mut tools = ToolInventory::detect();
        for tool in [
            &mut tools.cwebp,
            &mut tools.avifenc,
            &mut tools.avifdec,
            &mut tools.cjxl,
            &mut tools.djxl,
        ] {
            tool.enabled = true;
        }
        assert!(
            [
                Codec::Png,
                Codec::Jpeg,
                Codec::Webp,
                Codec::Avif,
                Codec::Jxl
            ]
            .into_iter()
            .all(|codec| tools.available_for(codec))
        );

        tools.avifdec.enabled = false;
        assert!(!tools.available_for(Codec::Avif));
        assert_eq!(tools.missing_for(Codec::Avif), ["avifdec"]);
        tools.cjxl.enabled = false;
        tools.djxl.enabled = false;
        assert_eq!(tools.missing_for(Codec::Jxl), ["cjxl", "djxl"]);
        // The built-in formats and WebP, which is read back in process, are untouched.
        assert!(tools.available_for(Codec::Png) && tools.available_for(Codec::Jpeg));
        assert!(tools.available_for(Codec::Webp));
    }

    #[cfg(windows)]
    #[test]
    fn system_programs_are_started_by_absolute_path() {
        for relative in [
            r"System32\taskkill.exe",
            r"System32\cmd.exe",
            "explorer.exe",
        ] {
            let path = system_program(relative);
            assert!(path.is_absolute() && path.is_file(), "{}", path.display());
        }
    }

    #[test]
    fn a_process_failure_names_the_executable_without_its_folder() {
        let failure = ProcessFailure::new(
            Path::new("/home/someone/tools/cwebp.exe"),
            ProcessFailureKind::Exit,
            Some(1),
            "bad input".to_string(),
            5,
            false,
        );
        let text = failure.to_string();
        assert!(text.starts_with("cwebp.exe Exit"), "{text}");
        assert!(!text.contains("someone"));
        // The full path stays available to the code that runs the tool.
        assert!(failure.executable.starts_with("/home/someone"));
    }

    #[test]
    fn path_lookup_skips_empty_and_relative_entries() {
        let absolute = std::env::temp_dir();
        let joined = std::env::join_paths([
            PathBuf::new(),
            PathBuf::from("."),
            PathBuf::from("tools"),
            absolute.clone(),
        ])
        .unwrap();

        assert_eq!(absolute_path_dirs(&joined), vec![absolute]);
        // A trailing separator adds an empty entry.
        let mut trailing = std::env::temp_dir().into_os_string();
        trailing.push(if cfg!(windows) { ";" } else { ":" });
        assert_eq!(absolute_path_dirs(&trailing).len(), 1);
    }

    #[test]
    fn tool_dirs_stay_beside_the_executable_or_its_workspace() {
        let root = tempfile::tempdir().unwrap();
        let make = |path: &str| {
            let dir = root.path().join(path);
            fs::create_dir_all(&dir).unwrap();
            dir
        };

        // A release folder: tools beside the executable. A `tools` folder
        // further up must not be used.
        make("tools");
        let release = make("downloads/smart-compress/tools/libjxl/bin");
        let release_exe_dir = release.ancestors().nth(3).unwrap().to_path_buf();
        assert_eq!(
            tool_dirs_for(&release_exe_dir),
            vec![
                release_exe_dir.join("tools"),
                release_exe_dir.join("tools/libjxl/bin")
            ]
        );

        // A development build: tools in the workspace that holds `target`.
        make("work/tools/libavif");
        let workspace = root.path().join("work");
        fs::write(workspace.join("Cargo.toml"), "").unwrap();
        let target = make("work/target/release");
        assert_eq!(
            tool_dirs_for(&target),
            vec![workspace.join("tools"), workspace.join("tools/libavif")]
        );
        // The same workspace does not apply to an executable outside `target`.
        assert!(tool_dirs_for(&workspace.join("tools/libavif")).is_empty());
    }

    #[test]
    fn jpeg_does_not_support_alpha() {
        assert!(!Codec::Jpeg.supports_alpha());
        assert!(Codec::Webp.supports_alpha());
    }

    #[test]
    fn quality_setting_label_is_stable() {
        assert_eq!(
            EncoderSettings::Quality {
                quality: 82,
                effort: 4
            }
            .label(),
            "q82/e4"
        );
    }

    #[test]
    fn process_output_capture_drains_but_caps_retained_bytes() {
        let input = vec![b'x'; MAX_PROCESS_OUTPUT_BYTES + 8192];
        let captured = read_capped(Cursor::new(input), MAX_PROCESS_OUTPUT_BYTES).unwrap();

        assert_eq!(captured.bytes.len(), MAX_PROCESS_OUTPUT_BYTES);
        assert!(captured.truncated);
    }

    #[test]
    fn process_error_prefers_first_non_empty_stderr_line() {
        let stderr = CapturedStream {
            bytes: b"\nencoder failed\nmore detail".to_vec(),
            truncated: false,
        };
        let stdout = CapturedStream {
            bytes: b"progress".to_vec(),
            truncated: false,
        };

        assert_eq!(output_suffix(&stderr, &stdout), ": encoder failed");
    }

    #[test]
    fn process_output_exposes_capture_metadata() {
        let output = ProcessOutput {
            stdout: b"score".to_vec(),
            stderr: Vec::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            elapsed_ms: 7,
        };

        assert_eq!(output.stdout, b"score");
        assert!(!output.stdout_truncated);
        assert_eq!(output.elapsed_ms, 7);
    }

    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64},
        mpsc,
    };

    /// Runs `scenario` on its own thread and fails, instead of hanging, if it
    /// does not finish: a deadlock has to be a test failure.
    fn finishes_within(limit: Duration, scenario: impl FnOnce() + Send + 'static) {
        let (done, finished) = mpsc::channel();
        thread::spawn(move || {
            scenario();
            let _ = done.send(());
        });
        finished
            .recv_timeout(limit)
            .expect("the scenario deadlocked or panicked");
    }

    /// Small deterministic generator so the load is the same on every run.
    fn next_random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn a_budget_never_hands_out_more_than_its_size() {
        const LIMIT: u64 = 8;
        let budget = Arc::new(Budget::new(LIMIT));
        let in_use = Arc::new(AtomicU64::new(0));
        let peak = Arc::new(AtomicU64::new(0));
        finishes_within(Duration::from_secs(60), {
            let (budget, in_use, peak) = (budget.clone(), in_use.clone(), peak.clone());
            move || {
                let workers = (0..12_u64)
                    .map(|seed| {
                        let (budget, in_use, peak) = (budget.clone(), in_use.clone(), peak.clone());
                        thread::spawn(move || {
                            let mut state = 0x9e37_79b9_7f4a_7c15 ^ (seed + 1);
                            for _ in 0..200 {
                                // Requests up to larger than the whole budget.
                                let units = next_random(&mut state) % 12 + 1;
                                let permit = budget.acquire_unless(units, None, || false).unwrap();
                                let now = in_use.fetch_add(permit.units(), Ordering::SeqCst)
                                    + permit.units();
                                peak.fetch_max(now, Ordering::SeqCst);
                                assert!(budget.used() <= LIMIT);
                                thread::yield_now();
                                in_use.fetch_sub(permit.units(), Ordering::SeqCst);
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                for worker in workers {
                    worker.join().unwrap();
                }
            }
        });
        assert!(peak.load(Ordering::SeqCst) <= LIMIT);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn a_request_larger_than_the_budget_runs_alone() {
        let budget = Arc::new(Budget::new(4));
        let small = budget.acquire_unless(2, None, || false).unwrap();
        let (granted, wait_for_grant) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let oversize = {
            let budget = budget.clone();
            thread::spawn(move || {
                let permit = budget.acquire_unless(100, None, || false).unwrap();
                granted.send((permit.units(), budget.used())).unwrap();
                released.recv().unwrap();
            })
        };
        // It waits for the small holder instead of squeezing in beside it.
        assert!(
            wait_for_grant
                .recv_timeout(Duration::from_millis(150))
                .is_err()
        );
        drop(small);
        // Granted the whole budget, and nothing else is let in beside it.
        assert_eq!(
            wait_for_grant.recv_timeout(Duration::from_secs(5)).unwrap(),
            (4, 4)
        );
        let (second_granted, second_wait) = mpsc::channel();
        let second = {
            let budget = budget.clone();
            thread::spawn(move || {
                let _permit = budget.acquire_unless(1, None, || false).unwrap();
                second_granted.send(()).unwrap();
            })
        };
        assert!(
            second_wait
                .recv_timeout(Duration::from_millis(150))
                .is_err()
        );
        release.send(()).unwrap();
        second_wait.recv_timeout(Duration::from_secs(5)).unwrap();
        oversize.join().unwrap();
        second.join().unwrap();
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn a_large_request_is_not_starved_by_small_ones() {
        let budget = Arc::new(Budget::new(4));
        let order = Arc::new(Mutex::new(Vec::new()));
        let holder = budget.acquire_unless(3, None, || false).unwrap();
        let large = {
            let (budget, order) = (budget.clone(), order.clone());
            thread::spawn(move || {
                let _permit = budget.acquire_unless(4, None, || false).unwrap();
                order.lock().unwrap().push("large");
            })
        };
        while budget.lock().queue.is_empty() {
            thread::sleep(Duration::from_millis(5));
        }
        // One unit is free, but the large request is ahead in line.
        let small = {
            let (budget, order) = (budget.clone(), order.clone());
            thread::spawn(move || {
                let _permit = budget.acquire_unless(1, None, || false).unwrap();
                order.lock().unwrap().push("small");
            })
        };
        thread::sleep(Duration::from_millis(150));
        assert!(order.lock().unwrap().is_empty());
        drop(holder);
        large.join().unwrap();
        small.join().unwrap();
        assert_eq!(*order.lock().unwrap(), ["large", "small"]);
    }

    #[test]
    fn cancellation_wakes_a_waiter() {
        let budget = Arc::new(Budget::new(1));
        let holder = budget.acquire_unless(1, None, || false).unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (ended, wait_for_end) = mpsc::channel();
        let waiter = {
            let (budget, cancelled) = (budget.clone(), cancelled.clone());
            thread::spawn(move || {
                let result = budget
                    .acquire_unless(1, None, || cancelled.load(Ordering::SeqCst))
                    .map(|_| ());
                ended.send(result).unwrap();
            })
        };
        thread::sleep(Duration::from_millis(100));
        cancelled.store(true, Ordering::SeqCst);
        budget.wake();
        assert_eq!(
            wait_for_end.recv_timeout(Duration::from_secs(5)).unwrap(),
            Err(WaitEnded::Cancelled)
        );
        waiter.join().unwrap();
        // The cancelled request left the line, so the budget is still usable.
        drop(holder);
        assert!(budget.acquire_unless(1, None, || false).is_ok());
    }

    #[test]
    fn a_deadline_ends_a_wait() {
        let budget = Budget::new(1);
        let _holder = budget.acquire_unless(1, None, || false).unwrap();
        let start = Instant::now();
        let ended = budget
            .acquire_unless(1, Some(start + Duration::from_millis(80)), || false)
            .map(|_| ());
        assert_eq!(ended, Err(WaitEnded::TimedOut));
        assert!(start.elapsed() >= Duration::from_millis(80));
        assert_eq!(budget.lock().queue.len(), 0);
    }

    #[test]
    fn a_condition_wait_ends_when_the_condition_clears() {
        let gate = Arc::new(Budget::new(1));
        let blocked = Arc::new(AtomicBool::new(true));
        let waiter = {
            let (gate, blocked) = (gate.clone(), blocked.clone());
            thread::spawn(move || gate.wait_while(None, || blocked.load(Ordering::SeqCst)))
        };
        thread::sleep(Duration::from_millis(100));
        blocked.store(false, Ordering::SeqCst);
        gate.wake();
        assert_eq!(waiter.join().unwrap(), Ok(()));
        assert_eq!(
            gate.wait_while(Some(Instant::now() + Duration::from_millis(50)), || true),
            Err(WaitEnded::TimedOut)
        );
    }

    #[test]
    fn file_workers_and_their_candidates_do_not_deadlock() {
        // The shape of a batch: file workers reserve memory for their image,
        // do a step of their own, then wait on candidate threads that each ask
        // for several units. A file worker holds no work units while it waits.
        finishes_within(Duration::from_secs(60), || {
            let work = Arc::new(Budget::new(4));
            let memory = Arc::new(Budget::new(10));
            let files = (0..8_u64)
                .map(|file| {
                    let (work, memory) = (work.clone(), memory.clone());
                    thread::spawn(move || {
                        let mut state = 0xdead_beef_u64 ^ (file + 1);
                        let _image = memory.acquire_unless(4, None, || false).unwrap();
                        drop(work.acquire_unless(1, None, || false).unwrap());
                        for _ in 0..5 {
                            let candidates = (0..6)
                                .map(|_| {
                                    let work = work.clone();
                                    let units = next_random(&mut state) % 7 + 1;
                                    thread::spawn(move || {
                                        let _threads =
                                            work.acquire_unless(units, None, || false).unwrap();
                                        thread::sleep(Duration::from_millis(1));
                                    })
                                })
                                .collect::<Vec<_>>();
                            for candidate in candidates {
                                candidate.join().unwrap();
                            }
                        }
                        drop(work.acquire_unless(1, None, || false).unwrap());
                    })
                })
                .collect::<Vec<_>>();
            for file in files {
                file.join().unwrap();
            }
            assert_eq!(work.used(), 0);
            assert_eq!(memory.used(), 0);
        });
    }

    #[cfg(windows)]
    #[test]
    fn process_runner_drains_large_live_output() {
        let output = run_process(
            Path::new("powershell.exe"),
            &[
                OsString::from("-NoProfile"),
                OsString::from("-Command"),
                OsString::from("[Console]::Out.Write('x' * 100000)"),
            ],
            Duration::from_secs(10),
        )
        .unwrap();

        assert_eq!(output.stdout.len(), MAX_PROCESS_OUTPUT_BYTES);
        assert!(output.stdout_truncated);
    }

    #[cfg(windows)]
    #[test]
    fn process_runner_starts_tools_in_their_own_folder() {
        let system = PathBuf::from(std::env::var_os("SystemRoot").unwrap());
        let shell = system.join(r"System32\WindowsPowerShell\v1.0\powershell.exe");
        let output = run_process(
            &shell,
            &[
                OsString::from("-NoProfile"),
                OsString::from("-Command"),
                OsString::from("[Console]::Out.Write((Get-Location).Path)"),
            ],
            Duration::from_secs(10),
        )
        .unwrap();

        assert_eq!(
            String::from_utf8_lossy(&output.stdout).to_lowercase(),
            shell.parent().unwrap().to_string_lossy().to_lowercase()
        );
    }

    #[cfg(windows)]
    #[test]
    fn process_runner_returns_structured_exit_failure() {
        let error = run_process(
            Path::new("powershell.exe"),
            &[
                OsString::from("-NoProfile"),
                OsString::from("-Command"),
                OsString::from("[Console]::Error.WriteLine('useful failure'); exit 7"),
            ],
            Duration::from_secs(10),
        )
        .unwrap_err();
        let failure = error.downcast_ref::<ProcessFailure>().unwrap();

        assert_eq!(failure.kind, ProcessFailureKind::Exit);
        assert_eq!(failure.exit_code, Some(7));
        assert_eq!(failure.message, "useful failure");
        assert!(!failure.termination_requested);
    }

    #[cfg(windows)]
    #[test]
    fn process_runner_times_out_and_requests_tree_termination() {
        let error = run_process(
            Path::new("powershell.exe"),
            &[
                OsString::from("-NoProfile"),
                OsString::from("-Command"),
                OsString::from("Start-Sleep -Seconds 30"),
            ],
            Duration::from_millis(150),
        )
        .unwrap_err();
        let failure = error.downcast_ref::<ProcessFailure>().unwrap();

        assert_eq!(failure.kind, ProcessFailureKind::Timeout);
        assert!(failure.termination_requested);
        assert!(failure.elapsed_ms >= 150);
    }

    #[cfg(windows)]
    #[test]
    fn process_runner_terminates_spawned_children() {
        let script = "$child = Start-Process powershell.exe -WindowStyle Hidden -ArgumentList '-NoProfile -Command Start-Sleep -Seconds 30' -PassThru; [Console]::Out.WriteLine($child.Id); Start-Sleep -Seconds 30";
        let error = run_process(
            Path::new("powershell.exe"),
            &[
                OsString::from("-NoProfile"),
                OsString::from("-Command"),
                OsString::from(script),
            ],
            // Long enough for PowerShell to start and print the child id on a
            // slow machine; the script itself sleeps far longer.
            Duration::from_secs(8),
        )
        .unwrap_err();
        let failure = error.downcast_ref::<ProcessFailure>().unwrap();
        let child_id = failure.message.trim().parse::<u32>().unwrap();

        // Each check starts PowerShell, which alone can take over a second
        // on a loaded machine.
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let status = Command::new("powershell.exe")
                .args([
                    "-NoProfile",
                    "-Command",
                    &format!(
                        "if (Get-Process -Id {child_id} -ErrorAction SilentlyContinue) {{ exit 1 }}"
                    ),
                ])
                .status()
                .unwrap();
            if status.success() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "spawned child {child_id} survived timeout"
            );
            thread::sleep(Duration::from_millis(50));
        }
    }
}
