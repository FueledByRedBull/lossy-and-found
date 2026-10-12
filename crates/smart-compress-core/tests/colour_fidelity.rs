//! Colour differences must not create false quality wins.
//!
//! The metric compares channel values as if they were sRGB and nothing is
//! converted, so a candidate is only trustworthy when it keeps the colour
//! interpretation of the source: it carries the profile the pixels are in, or
//! the image is refused. Tests that need external tools skip themselves when
//! the tool is missing (the GitHub runners have none); they are marked below.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

use image::{
    DynamicImage, ExtendedColorType, GrayImage, ImageDecoder, ImageEncoder, ImageReader, RgbImage,
    codecs::png::PngEncoder,
};
use smart_compress_codecs::{
    Codec, EncodeJob, EncoderBackend, EncoderSettings, MetadataPayload, ToolInventory,
    icc_is_plain_srgb, normalized_png, tool_names_for, tools_or_skip,
};
use smart_compress_core::{
    AlphaPolicy, CliConfig, CodecSelection, ImageReport, MetadataPolicy, Optimizer,
    config::CacheConfig,
    image_probe::{ImageDecodeLimits, read_pam},
};

const POLICIES: [MetadataPolicy; 3] = [
    MetadataPolicy::Keep,
    MetadataPolicy::Safe,
    MetadataPolicy::Strip,
];

// D50-adapted colorants (red, green, blue) as XYZ.
const SRGB: [[f64; 3]; 3] = [
    [0.4361, 0.2225, 0.0139],
    [0.3851, 0.7169, 0.0971],
    [0.1431, 0.0606, 0.7141],
];
const DISPLAY_P3: [[f64; 3]; 3] = [
    [0.5151, 0.2412, -0.0011],
    [0.2920, 0.6922, 0.0419],
    [0.1571, 0.0666, 0.7841],
];
const ADOBE_RGB: [[f64; 3]; 3] = [
    [0.6097, 0.3111, 0.0195],
    [0.2053, 0.6257, 0.0609],
    [0.1492, 0.0632, 0.7446],
];

#[derive(Clone, Copy)]
enum Curve {
    Gamma(f64),
    /// The piecewise sRGB curve as an ICC parametric curve.
    Srgb,
}

fn s15(value: f64) -> [u8; 4] {
    ((value * 65536.0).round() as i32).to_be_bytes()
}

fn xyz(values: [f64; 3]) -> Vec<u8> {
    let mut data = b"XYZ \0\0\0\0".to_vec();
    values.iter().for_each(|v| data.extend_from_slice(&s15(*v)));
    data
}

fn curve(curve: Curve) -> Vec<u8> {
    match curve {
        Curve::Gamma(gamma) => {
            let mut data = b"curv\0\0\0\0\0\0\0\x01".to_vec();
            data.extend_from_slice(&((gamma * 256.0).round() as u16).to_be_bytes());
            data
        }
        Curve::Srgb => {
            let mut data = b"para\0\0\0\0\0\x03\0\0".to_vec();
            for value in [2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045] {
                data.extend_from_slice(&s15(value));
            }
            data
        }
    }
}

/// A well-formed ICC v2 profile made of the given tags.
fn profile(space: &[u8; 4], tags: Vec<(&[u8; 4], Vec<u8>)>) -> Vec<u8> {
    let mut directory = (tags.len() as u32).to_be_bytes().to_vec();
    let mut body = Vec::new();
    let body_start = 128 + 4 + 12 * tags.len();
    for (signature, data) in &tags {
        directory.extend_from_slice(*signature);
        directory.extend_from_slice(&((body_start + body.len()) as u32).to_be_bytes());
        directory.extend_from_slice(&(data.len() as u32).to_be_bytes());
        body.extend_from_slice(data);
        body.resize(body.len().next_multiple_of(4), 0);
    }
    let mut bytes = vec![0_u8; 128];
    bytes[0..4].copy_from_slice(&((body_start + body.len()) as u32).to_be_bytes());
    bytes[8..12].copy_from_slice(&[2, 0x10, 0, 0]);
    bytes[12..16].copy_from_slice(b"mntr");
    bytes[16..20].copy_from_slice(space);
    bytes[20..24].copy_from_slice(b"XYZ ");
    bytes[36..40].copy_from_slice(b"acsp");
    bytes[68..80].copy_from_slice(&[0, 0, 0xF6, 0xD6, 0, 1, 0, 0, 0, 0, 0xD3, 0x2D]);
    bytes.extend_from_slice(&directory);
    bytes.extend_from_slice(&body);
    bytes
}

fn description() -> Vec<u8> {
    let mut desc = b"desc\0\0\0\0".to_vec();
    desc.extend_from_slice(&5_u32.to_be_bytes());
    desc.extend_from_slice(b"test\0");
    desc.extend_from_slice(&[0; 1 + 4 + 4 + 2 + 1 + 67]);
    desc
}

fn rgb_profile(colorants: [[f64; 3]; 3], trc: Curve) -> Vec<u8> {
    let [red, green, blue] = colorants;
    profile(
        b"RGB ",
        vec![
            (b"desc", description()),
            (b"wtpt", xyz([0.9642, 1.0, 0.8249])),
            (b"rXYZ", xyz(red)),
            (b"gXYZ", xyz(green)),
            (b"bXYZ", xyz(blue)),
            (b"rTRC", curve(trc)),
            (b"gTRC", curve(trc)),
            (b"bTRC", curve(trc)),
        ],
    )
}

fn grey_profile(trc: Curve) -> Vec<u8> {
    profile(
        b"GRAY",
        vec![
            (b"desc", description()),
            (b"wtpt", xyz([0.9642, 1.0, 0.8249])),
            (b"kTRC", curve(trc)),
        ],
    )
}

fn display_p3() -> Vec<u8> {
    rgb_profile(DISPLAY_P3, Curve::Gamma(2.2))
}

/// Saturated sweeps with no hard edges, so lossy codecs stay close to the
/// source and a change of colour space shows up as a large difference.
fn smooth_colours() -> RgbImage {
    RgbImage::from_fn(96, 64, |x, y| {
        let across = (x * 255 / 95) as u8;
        let down = (y * 255 / 63) as u8;
        image::Rgb([across, down, 255 - across])
    })
}

fn png_with_profile(image: &DynamicImage, icc: Option<&[u8]>) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = PngEncoder::new(&mut bytes);
    if let Some(icc) = icc {
        encoder.set_icc_profile(icc.to_vec()).unwrap();
    }
    image.write_with_encoder(encoder).unwrap();
    bytes
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & 0_u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

type Chunk = (&'static [u8; 4], Vec<u8>);

/// `png` with the given chunks placed straight after IHDR.
fn with_chunks(png: &[u8], chunks: &[Chunk]) -> Vec<u8> {
    let mut inserted = Vec::new();
    for (kind, data) in chunks {
        inserted.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        inserted.extend_from_slice(&body);
        inserted.extend_from_slice(&crc32(&body).to_be_bytes());
    }
    let mut out = png.to_vec();
    out.splice(33..33, inserted);
    out
}

fn cicp(primaries: u8, transfer: u8) -> Vec<u8> {
    vec![primaries, transfer, 0, 1]
}

fn config(input: &Path, out_dir: &Path, codec: Codec, metadata: MetadataPolicy) -> CliConfig {
    let mut config = CliConfig {
        inputs: vec![input.to_path_buf()],
        out_dir: Some(out_dir.to_path_buf()),
        formats: CodecSelection::new([codec]),
        allow_larger: true,
        metadata,
        ..CliConfig::default()
    };
    config.gpu.enabled = false;
    config
}

fn run(config: CliConfig) -> ImageReport {
    let mut summary = Optimizer::new(config).run().unwrap();
    assert_eq!(summary.report.reports.len(), 1);
    summary.report.reports.remove(0)
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

fn icc_of(path: &Path) -> Option<Vec<u8>> {
    ImageReader::open(path)
        .unwrap()
        .with_guessed_format()
        .unwrap()
        .into_decoder()
        .unwrap()
        .icc_profile()
        .unwrap()
}

fn mean_difference(a: &DynamicImage, b: &DynamicImage) -> f64 {
    let (a, b) = (a.to_rgb8(), b.to_rgb8());
    assert_eq!(a.dimensions(), b.dimensions());
    let total: u64 = a
        .as_raw()
        .iter()
        .zip(b.as_raw())
        .map(|(x, y)| u64::from(x.abs_diff(*y)))
        .sum();
    total as f64 / a.as_raw().len() as f64
}

/// Mean difference that lossy candidates accepted by the search stay below on
/// the smooth test image; a change of colour space is more than twice this
/// (see `a_change_of_colour_space_is_far_above_coding_noise`).
const LOSSY_NOISE: f64 = 5.0;

fn tool_dir(tool: &Option<PathBuf>) -> Option<PathBuf> {
    tool.as_ref()?.parent().map(Path::to_path_buf)
}

/// The tools a candidate of `codec` needs to be made and read back.
struct Tools {
    inventory: ToolInventory,
    avifdec: Option<PathBuf>,
    jxlinfo: Option<PathBuf>,
}

impl Tools {
    fn detect() -> Self {
        let inventory = ToolInventory::detect();
        let sibling = |tool: &Option<PathBuf>, name: &str| {
            tool_dir(tool)
                .map(|dir| dir.join(name))
                .filter(|path| path.exists())
        };
        Self {
            avifdec: inventory.avifdec.path.clone(),
            jxlinfo: sibling(&inventory.djxl.path, "jxlinfo.exe"),
            inventory,
        }
    }

    /// Whether the tools of `codec` are installed; says what is missing if not,
    /// or fails when the tools are required.
    fn can_run(&self, codec: Codec) -> bool {
        let mut needed = tool_names_for(&[codec]);
        if codec == Codec::Webp {
            // The metadata of a WebP is written with webpmux.
            needed.push("webpmux");
        }
        tools_or_skip(&needed).is_some()
    }

    /// avifdec to an uncompressed PNG, as the pipeline runs it.
    fn avif_to_png(&self, path: &Path) -> PathBuf {
        let png = path.with_extension("decoded.png");
        let args = [
            "--png-compress".into(),
            "0".into(),
            path.into(),
            png.clone().into(),
        ];
        run_tool(self.avifdec.as_ref().unwrap(), &args);
        png
    }

    /// Decodes `path` the way the pipeline decodes a candidate.
    fn decode(&self, path: &Path) -> DynamicImage {
        match path.extension().and_then(|e| e.to_str()) {
            Some("avif") => image::open(self.avif_to_png(path)).unwrap(),
            Some("jxl") => {
                let pam = path.with_extension("decoded.pam");
                let args = ["--num_threads=2".into(), path.into(), pam.clone().into()];
                run_tool(self.inventory.djxl.path.as_ref().unwrap(), &args);
                read_pam(&pam, ImageDecodeLimits::default()).unwrap()
            }
            _ => image::open(path).unwrap(),
        }
    }

    /// Whether the file still carries `expected` (or, for JPEG XL, which can
    /// rewrite a profile as an equivalent description, something other than
    /// the sRGB it would assume without one).
    fn carries(&self, path: &Path, expected: &[u8]) -> bool {
        match path.extension().and_then(|e| e.to_str()) {
            Some("avif") => {
                let info = run_tool(
                    self.avifdec.as_ref().unwrap(),
                    &["--info".into(), path.into()],
                );
                info.contains("ICC Profile    : Present")
                    && icc_of(&self.avif_to_png(path)).as_deref() == Some(expected)
            }
            Some("jxl") => {
                let Some(jxlinfo) = &self.jxlinfo else {
                    println!("jxlinfo not found: JPEG XL colour description not checked");
                    return true;
                };
                let info = run_tool(jxlinfo, &["-v".into(), path.into()]);
                let line = info
                    .lines()
                    .find(|line| line.starts_with("Color space"))
                    .unwrap_or_default();
                !line.contains("sRGB primaries") && !line.contains("sRGB transfer")
            }
            _ => icc_of(path).as_deref() == Some(expected),
        }
    }
}

fn run_tool(tool: &Path, args: &[OsString]) -> String {
    let out = Command::new(tool).args(args).output().unwrap();
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn plain_srgb_is_recognised_conservatively() {
    let srgb = |trc| rgb_profile(SRGB, trc);
    assert!(icc_is_plain_srgb(&srgb(Curve::Srgb)));
    // The usual stand-in for sRGB.
    assert!(icc_is_plain_srgb(&srgb(Curve::Gamma(2.2))));
    assert!(icc_is_plain_srgb(&grey_profile(Curve::Srgb)));
    assert!(icc_is_plain_srgb(&grey_profile(Curve::Gamma(2.2))));

    // Different primaries, or the same primaries with a different curve.
    assert!(!icc_is_plain_srgb(&display_p3()));
    assert!(!icc_is_plain_srgb(&rgb_profile(DISPLAY_P3, Curve::Srgb)));
    assert!(!icc_is_plain_srgb(&rgb_profile(
        ADOBE_RGB,
        Curve::Gamma(2.2)
    )));
    assert!(!icc_is_plain_srgb(&srgb(Curve::Gamma(1.8))));
    assert!(!icc_is_plain_srgb(&srgb(Curve::Gamma(2.4))));
    assert!(!icc_is_plain_srgb(&srgb(Curve::Gamma(1.0))));
    assert!(!icc_is_plain_srgb(&grey_profile(Curve::Gamma(1.8))));
    assert!(!icc_is_plain_srgb(&grey_profile(Curve::Gamma(1.0))));

    // Anything that cannot be shown to be sRGB is kept: a header alone, junk,
    // and a profile with lookup tables.
    let mut header = vec![0_u8; 132];
    header[16..20].copy_from_slice(b"RGB ");
    header[36..40].copy_from_slice(b"acsp");
    assert!(!icc_is_plain_srgb(&header));
    assert!(!icc_is_plain_srgb(b"not a profile"));
    assert!(!icc_is_plain_srgb(&[]));
    let [red, green, blue] = SRGB;
    let lookup = profile(
        b"RGB ",
        vec![
            (b"rXYZ", xyz(red)),
            (b"gXYZ", xyz(green)),
            (b"bXYZ", xyz(blue)),
            (b"rTRC", curve(Curve::Srgb)),
            (b"gTRC", curve(Curve::Srgb)),
            (b"bTRC", curve(Curve::Srgb)),
            (b"A2B0", vec![0; 32]),
        ],
    );
    assert!(!icc_is_plain_srgb(&lookup));
    // The CMYK and Lab spaces have no sRGB reading.
    let mut cmyk = srgb(Curve::Srgb);
    cmyk[16..20].copy_from_slice(b"CMYK");
    assert!(!icc_is_plain_srgb(&cmyk));
}

/// (a) A profile that says what the pixels mean is colour data: no policy may
/// drop it, and the pixels must come through untouched. Native outputs only,
/// so this runs everywhere.
#[test]
fn non_srgb_profile_survives_every_policy_in_native_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let source = DynamicImage::ImageRgb8(smooth_colours());
    let wide_gamut = [
        ("display-p3", display_p3()),
        ("adobe-rgb", rgb_profile(ADOBE_RGB, Curve::Gamma(2.2))),
        (
            "srgb-primaries-gamma-1.8",
            rgb_profile(SRGB, Curve::Gamma(1.8)),
        ),
    ];
    for (name, icc) in wide_gamut {
        let input = write(
            dir.path(),
            &format!("{name}.png"),
            &png_with_profile(&source, Some(&icc)),
        );
        for policy in POLICIES {
            for codec in [Codec::Png, Codec::Jpeg] {
                let out = dir.path().join(format!("{name}-{codec:?}-{policy:?}"));
                let mut config = config(&input, &out, codec, policy);
                config.min_quality = Some(0.0);
                let report = run(config);
                let label = format!("{name} {codec:?} {policy:?}");
                assert!(report.errors.is_empty(), "{label}: {:?}", report.errors);
                let output = PathBuf::from(report.output.as_deref().expect(&label));
                assert_eq!(icc_of(&output).as_deref(), Some(icc.as_slice()), "{label}");
                if codec == Codec::Png {
                    assert_eq!(
                        image::open(&output).unwrap().to_rgb8(),
                        source.to_rgb8(),
                        "{label}"
                    );
                }
                // Dropping it silently is the thing the report must never hide.
                assert_eq!(
                    report
                        .warnings
                        .iter()
                        .any(|w| w.contains("kept under --metadata strip")),
                    policy == MetadataPolicy::Strip,
                    "{label}: {:?}",
                    report.warnings
                );
            }
        }
    }
}

/// What the other tests detect: the test picture read as Display P3 and shown
/// as sRGB differs by far more than coding noise, so a candidate decoded through
/// a path that converted the colours would not look like the original to any
/// metric that compares the numbers.
#[test]
fn a_change_of_colour_space_is_far_above_coding_noise() {
    let inverse = |m: [[f64; 3]; 3]| {
        let cofactor = |r: usize, c: usize| {
            let (r1, r2, c1, c2) = ((r + 1) % 3, (r + 2) % 3, (c + 1) % 3, (c + 2) % 3);
            m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1]
        };
        let det: f64 = (0..3).map(|c| m[0][c] * cofactor(0, c)).sum();
        std::array::from_fn::<_, 3, _>(|r| std::array::from_fn::<_, 3, _>(|c| cofactor(c, r) / det))
    };
    // Colorants are stored per primary; the matrices act on column vectors.
    let columns =
        |c: [[f64; 3]; 3]| std::array::from_fn::<_, 3, _>(|r| [c[0][r], c[1][r], c[2][r]]);
    let to_srgb = inverse(columns(SRGB));
    let p3 = columns(DISPLAY_P3);
    let source = smooth_colours();
    let converted = RgbImage::from_fn(source.width(), source.height(), |x, y| {
        let pixel = source.get_pixel(x, y);
        let linear = pixel.0.map(|v| (f64::from(v) / 255.0).powf(2.2));
        let xyz =
            std::array::from_fn::<_, 3, _>(|r| (0..3).map(|c| p3[r][c] * linear[c]).sum::<f64>());
        image::Rgb(std::array::from_fn(|r| {
            let value: f64 = (0..3).map(|c| to_srgb[r][c] * xyz[c]).sum();
            (value.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
        }))
    });
    let shift = mean_difference(
        &DynamicImage::ImageRgb8(source),
        &DynamicImage::ImageRgb8(converted),
    );
    assert!(shift > 2.0 * LOSSY_NOISE, "{shift}");
}

/// A profile that is plain sRGB may still be dropped by `strip`: viewers show
/// the same colours without it.
#[test]
fn plain_srgb_profile_is_dropped_only_by_strip() {
    let dir = tempfile::tempdir().unwrap();
    let source = DynamicImage::ImageRgb8(smooth_colours());
    for (name, icc) in [
        ("srgb-parametric", rgb_profile(SRGB, Curve::Srgb)),
        ("srgb-gamma-2.2", rgb_profile(SRGB, Curve::Gamma(2.2))),
    ] {
        let input = write(
            dir.path(),
            &format!("{name}.png"),
            &png_with_profile(&source, Some(&icc)),
        );
        for policy in POLICIES {
            let out = dir.path().join(format!("{name}-{policy:?}"));
            let mut config = config(&input, &out, Codec::Png, policy);
            config.min_quality = Some(0.0);
            let report = run(config);
            let output = PathBuf::from(report.output.as_deref().expect("an output"));
            let expected = (policy != MetadataPolicy::Strip).then_some(icc.clone());
            assert_eq!(icc_of(&output), expected, "{name} {policy:?}");
            assert!(
                report.warnings.is_empty(),
                "{name} {policy:?}: {:?}",
                report.warnings
            );
        }
    }
}

/// (g) The cache key holds the bytes the encoders read, which include the
/// profile the policy leaves in, so one policy cannot be served another's
/// artifact or score.
#[test]
fn cache_does_not_cross_metadata_policies() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache");
    let icc = rgb_profile(SRGB, Curve::Srgb);
    let input = write(
        dir.path(),
        "srgb.png",
        &png_with_profile(&DynamicImage::ImageRgb8(smooth_colours()), Some(&icc)),
    );
    let mut results = Vec::new();
    for policy in [
        MetadataPolicy::Safe,
        MetadataPolicy::Strip,
        MetadataPolicy::Safe,
    ] {
        let out = dir.path().join(format!("out-{}", results.len()));
        let mut config = config(&input, &out, Codec::Png, policy);
        config.min_quality = Some(0.0);
        config.cache = CacheConfig {
            enabled: true,
            dir: Some(cache.clone()),
            ..CacheConfig::default()
        };
        let report = run(config);
        let output = PathBuf::from(report.output.as_deref().expect("an output"));
        let hit = report
            .candidates
            .iter()
            .any(|c| c.warnings.iter().any(|w| w == "cache hit"));
        results.push((policy, icc_of(&output), hit));
    }
    assert_eq!(results[0], (MetadataPolicy::Safe, Some(icc.clone()), false));
    // Strip finds nothing of Safe's, and does not inherit its profile.
    assert_eq!(results[1], (MetadataPolicy::Strip, None, false));
    // The repeated Safe run is served from the cache and still has the profile.
    assert_eq!(results[2], (MetadataPolicy::Safe, Some(icc), true));
}

/// (e) Sixteen-bit pixels in a wide-gamut space keep both depth and profile.
#[test]
fn sixteen_bit_wide_gamut_png_keeps_profile_and_samples() {
    let dir = tempfile::tempdir().unwrap();
    let icc = display_p3();
    let samples: Vec<u8> = (0..96 * 64 * 3_u32)
        .flat_map(|i| ((i * 977) as u16).to_ne_bytes())
        .collect();
    let mut bytes = Vec::new();
    let mut encoder = PngEncoder::new(&mut bytes);
    encoder.set_icc_profile(icc.clone()).unwrap();
    encoder
        .write_image(&samples, 96, 64, ExtendedColorType::Rgb16)
        .unwrap();
    let input = write(dir.path(), "p3-16.png", &bytes);
    for policy in POLICIES {
        let out = dir.path().join(format!("{policy:?}"));
        let mut config = config(&input, &out, Codec::Png, policy);
        config.min_quality = Some(0.0);
        let report = run(config);
        let output = PathBuf::from(report.output.as_deref().expect("an output"));
        assert_eq!(icc_of(&output), Some(icc.clone()), "{policy:?}");
        let decoded = image::open(&output).unwrap();
        assert_eq!(decoded.color(), image::ColorType::Rgb16, "{policy:?}");
        assert_eq!(
            decoded.to_rgb16(),
            image::open(&input).unwrap().to_rgb16(),
            "{policy:?}"
        );
    }
}

/// (f) Flattening works on the pixel values in their own space and the profile
/// stays with them.
#[test]
fn alpha_flattening_keeps_the_profile() {
    let dir = tempfile::tempdir().unwrap();
    let icc = display_p3();
    let rgba = image::RgbaImage::from_fn(48, 32, |x, y| {
        image::Rgba([255, (x * 5) as u8, 0, (y * 8) as u8])
    });
    let input = write(
        dir.path(),
        "p3-alpha.png",
        &png_with_profile(&DynamicImage::ImageRgba8(rgba), Some(&icc)),
    );
    for policy in POLICIES {
        let out = dir.path().join(format!("{policy:?}"));
        let mut config = config(&input, &out, Codec::Png, policy);
        config.min_quality = Some(0.0);
        config.alpha = AlphaPolicy::Flatten {
            background: "#ffffff".to_string(),
        };
        let report = run(config);
        let output = PathBuf::from(report.output.as_deref().expect("an output"));
        assert_eq!(icc_of(&output), Some(icc.clone()), "{policy:?}");
        assert!(
            !image::open(&output).unwrap().color().has_alpha(),
            "{policy:?}"
        );
    }
}

/// (d) A CMYK JPEG with a CMYK profile is shown by viewers through that
/// profile; converting it unmanaged would change the picture, so it is refused.
/// Without a profile viewers convert the same unmanaged way and it is only
/// flagged.
#[test]
fn cmyk_jpeg_with_a_cmyk_profile_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/cmyk.jpg");
    let jpeg = std::fs::read(fixture).unwrap();
    let with_profile = |space: &[u8; 4]| {
        let mut icc = vec![0_u8; 132];
        icc[16..20].copy_from_slice(space);
        icc[36..40].copy_from_slice(b"acsp");
        let mut segment = b"ICC_PROFILE\0\x01\x01".to_vec();
        segment.extend_from_slice(&icc);
        let mut bytes = jpeg[..2].to_vec();
        bytes.extend_from_slice(&[0xFF, 0xE2]);
        bytes.extend_from_slice(&((segment.len() + 2) as u16).to_be_bytes());
        bytes.extend_from_slice(&segment);
        bytes.extend_from_slice(&jpeg[2..]);
        bytes
    };

    let cmyk = write(dir.path(), "cmyk-profile.jpg", &with_profile(b"CMYK"));
    for policy in POLICIES {
        let out = dir.path().join(format!("refused-{policy:?}"));
        let report = run(config(&cmyk, &out, Codec::Png, policy));
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("CMYK") && e.contains("colour management")),
            "{policy:?}: {:?}",
            report.errors
        );
        assert_eq!(report.output, None);
        assert!(report.candidates.is_empty());
    }

    // A profile for another space is dropped as a mismatch, as before.
    let rgb = write(dir.path(), "cmyk-rgb-profile.jpg", &with_profile(b"RGB "));
    let report = run(config(
        &rgb,
        &dir.path().join("rgb"),
        Codec::Png,
        MetadataPolicy::Safe,
    ));
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains("CMYK source converted"))
    );

    let plain = write(dir.path(), "cmyk.jpg", &jpeg);
    let report = run(config(
        &plain,
        &dir.path().join("plain"),
        Codec::Png,
        MetadataPolicy::Safe,
    ));
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains("CMYK source converted"))
    );
}

/// (a, e) Colour that a PNG declares without a profile (cICP, gAMA, cHRM) is
/// read by viewers but not carried by any encoder here, and avifdec writes it
/// for AVIF sources. Anything that is not sRGB is refused; sRGB is processed.
#[test]
fn png_colour_tags_other_than_srgb_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let plain = png_with_profile(&DynamicImage::ImageRgb8(smooth_colours()), None);
    let gamma = |value: u32| value.to_be_bytes().to_vec();
    let chromaticities = |values: [u32; 8]| {
        values
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect::<Vec<_>>()
    };
    let srgb_chrm = [31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000];
    let p3_chrm = [31270, 32900, 68000, 32000, 26500, 69000, 15000, 6000];

    let refused: Vec<(&str, Vec<Chunk>, &str)> = vec![
        ("cicp-p3", vec![(b"cICP", cicp(12, 13))], "cICP"),
        ("cicp-pq", vec![(b"cICP", cicp(9, 16))], "cICP"),
        ("cicp-hlg", vec![(b"cICP", cicp(9, 18))], "cICP"),
        ("gama-linear", vec![(b"gAMA", gamma(100_000))], "gAMA"),
        ("gama-1.8", vec![(b"gAMA", gamma(55_556))], "gAMA"),
        ("chrm-p3", vec![(b"cHRM", chromaticities(p3_chrm))], "cHRM"),
    ];
    for (name, chunks, needle) in refused {
        let input = write(
            dir.path(),
            &format!("{name}.png"),
            &with_chunks(&plain, &chunks),
        );
        for policy in POLICIES {
            let out = dir.path().join(format!("{name}-{policy:?}"));
            let report = run(config(&input, &out, Codec::Png, policy));
            assert!(
                report
                    .errors
                    .iter()
                    .any(|e| e.contains(needle) && e.contains("neither carried")),
                "{name} {policy:?}: {:?}",
                report.errors
            );
            assert_eq!(report.output, None, "{name}");
        }
    }

    let accepted: Vec<(&str, Vec<Chunk>)> = vec![
        ("cicp-srgb", vec![(b"cICP", cicp(1, 13))]),
        ("cicp-unspecified", vec![(b"cICP", cicp(2, 2))]),
        (
            "gama-srgb",
            vec![
                (b"gAMA", gamma(45_455)),
                (b"cHRM", chromaticities(srgb_chrm)),
            ],
        ),
        // The sRGB chunk outranks the others, as it does in viewers.
        (
            "srgb-chunk",
            vec![(b"sRGB", vec![0]), (b"gAMA", gamma(45_455))],
        ),
    ];
    for (name, chunks) in accepted {
        let input = write(
            dir.path(),
            &format!("{name}.png"),
            &with_chunks(&plain, &chunks),
        );
        let report = run(config(
            &input,
            &dir.path().join(name),
            Codec::Png,
            MetadataPolicy::Safe,
        ));
        assert!(report.errors.is_empty(), "{name}: {:?}", report.errors);
        assert!(report.output.is_some(), "{name}");
    }

    // BT.709 primaries with the BT.709 curve look the same as sRGB in viewers:
    // processed, with a warning that names the tag.
    let input = write(
        dir.path(),
        "cicp-bt709-curve.png",
        &with_chunks(&plain, &[(b"cICP", cicp(1, 1))]),
    );
    let report = run(config(
        &input,
        &dir.path().join("cicp-bt709-curve"),
        Codec::Png,
        MetadataPolicy::Safe,
    ));
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(report.output.is_some());
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains("cICP (primaries 1, transfer 1)") && w.contains("treated as sRGB")),
        "{:?}",
        report.warnings
    );

    // A real profile is carried, so a stray tag beside it does not refuse the image.
    let icc = display_p3();
    let tagged = png_with_profile(&DynamicImage::ImageRgb8(smooth_colours()), Some(&icc));
    let input = write(
        dir.path(),
        "profile-and-cicp.png",
        &with_chunks(&tagged, &[(b"cICP", cicp(12, 13))]),
    );
    let report = run(config(
        &input,
        &dir.path().join("profile-and-cicp"),
        Codec::Png,
        MetadataPolicy::Strip,
    ));
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(
        icc_of(Path::new(report.output.as_deref().unwrap())),
        Some(icc)
    );
}

/// Needs avifenc and avifdec. (b) An AVIF source whose colour is a CICP triple
/// is decoded by avifdec into a PNG that says so; non-sRGB triples are refused
/// and an ICC profile in the AVIF is carried through.
#[test]
fn avif_sources_keep_or_refuse_their_colour() {
    let tools = Tools::detect();
    if !tools.can_run(Codec::Avif) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let icc = display_p3();
    let source = DynamicImage::ImageRgb8(smooth_colours());
    let plain_png = write(dir.path(), "plain.png", &png_with_profile(&source, None));
    let icc_png = write(dir.path(), "p3.png", &png_with_profile(&source, Some(&icc)));
    let encode = |name: &str, png: &Path, extra: &[&str]| {
        let path = dir.path().join(name);
        let mut args: Vec<OsString> = ["-j", "2", "-q", "90", "--speed", "8"]
            .map(Into::into)
            .to_vec();
        args.extend(extra.iter().map(OsString::from));
        args.extend([png.as_os_str().to_owned(), path.as_os_str().to_owned()]);
        let status = Command::new(tools.inventory.avifenc.path.as_ref().unwrap())
            .args(args)
            .output()
            .unwrap();
        assert!(
            status.status.success(),
            "{}",
            String::from_utf8_lossy(&status.stderr)
        );
        path
    };

    for (name, cicp_arg) in [("p3.avif", "12/13/6"), ("pq.avif", "9/16/9")] {
        let input = encode(name, &plain_png, &["--cicp", cicp_arg]);
        for policy in POLICIES {
            let out = dir.path().join(format!("{name}-{policy:?}"));
            let report = run(config(&input, &out, Codec::Png, policy));
            assert!(
                report
                    .errors
                    .iter()
                    .any(|e| e.contains("cICP") && e.contains("neither carried")),
                "{name} {policy:?}: {:?}",
                report.errors
            );
            assert_eq!(report.output, None);
        }
    }

    let srgb = encode("srgb.avif", &plain_png, &["--cicp", "1/13/6"]);
    let report = run(config(
        &srgb,
        &dir.path().join("srgb"),
        Codec::Png,
        MetadataPolicy::Safe,
    ));
    assert!(report.errors.is_empty(), "{:?}", report.errors);

    let with_icc = encode("icc.avif", &icc_png, &[]);
    for policy in POLICIES {
        let out = dir.path().join(format!("icc-{policy:?}"));
        let report = run(config(&with_icc, &out, Codec::Png, policy));
        assert!(report.errors.is_empty(), "{policy:?}: {:?}", report.errors);
        let output = Path::new(report.output.as_deref().unwrap());
        assert_eq!(icc_of(output), Some(icc.clone()), "{policy:?}");
    }
}

/// Needs the tools of each codec (cwebp with webpmux, avifenc with avifdec,
/// cjxl with djxl; cjpegli is used for JPEG when present and the built-in
/// encoder otherwise; ssimulacra2 is used when present). A Display P3 source is run through
/// every output codec under every metadata policy with the settings the
/// pipeline would use. Each output must carry an equivalent profile and decode,
/// the way the pipeline decodes it, to colours close to the source.
#[test]
fn display_p3_through_every_codec_and_policy() {
    let tools = Tools::detect();
    let dir = tempfile::tempdir().unwrap();
    let icc = display_p3();
    let source = DynamicImage::ImageRgb8(smooth_colours());
    let input = write(dir.path(), "p3.png", &png_with_profile(&source, Some(&icc)));
    for codec in [
        Codec::Png,
        Codec::Jpeg,
        Codec::Webp,
        Codec::Avif,
        Codec::Jxl,
    ] {
        if !tools.can_run(codec) {
            continue;
        }
        for policy in POLICIES {
            let out = dir.path().join(format!("{codec:?}-{policy:?}"));
            let report = run(config(&input, &out, codec, policy));
            let label = format!("{codec:?} {policy:?}");
            assert!(report.errors.is_empty(), "{label}: {:?}", report.errors);
            // The floor is the preset's, so an output exists only if it scored.
            let path = PathBuf::from(report.output.as_deref().unwrap_or_else(|| {
                panic!(
                    "{label}: no accepted candidate: {:?}",
                    report
                        .candidates
                        .iter()
                        .map(|c| &c.status)
                        .collect::<Vec<_>>()
                )
            }));
            assert!(
                tools.carries(&path, &icc),
                "{label}: output does not carry the profile"
            );
            let difference = mean_difference(&source, &tools.decode(&path));
            assert!(
                difference < LOSSY_NOISE,
                "{label}: decoded colours differ by {difference:.2} on average"
            );
            let winner = &report.candidates[report.winner_index.unwrap()];
            println!(
                "{label:12} {:>6} B {:?} quality {:.1} mean difference {difference:.2}",
                winner.size_bytes.unwrap(),
                winner.settings,
                winner.quality.as_ref().unwrap().overall
            );
        }
    }
}

/// Needs cjxl, djxl, avifenc, avifdec, cwebp and webpmux (each part skips on
/// its own). (b) The encoders given the pipeline's flags keep the pixel values
/// and the profile: lossless candidates decode to the source bit for bit, and
/// lossy ones stay within coding noise, which a conversion between colour
/// spaces would exceed several times over.
#[test]
fn encoders_and_decoders_do_not_convert_colour() {
    let tools = Tools::detect();
    let backend = EncoderBackend::new(tools.inventory.clone());
    let dir = tempfile::tempdir().unwrap();
    let icc = display_p3();
    let source = DynamicImage::ImageRgb8(smooth_colours());
    let grey = DynamicImage::ImageLuma8(GrayImage::from_fn(96, 64, |x, y| {
        image::Luma([(x * 2 + y) as u8])
    }));
    let grey_icc = grey_profile(Curve::Gamma(1.0));
    let lossy = |quality| EncoderSettings::Quality { quality, effort: 6 };
    let runnable = [
        Codec::Png,
        Codec::Jpeg,
        Codec::Webp,
        Codec::Avif,
        Codec::Jxl,
    ]
    .into_iter()
    .filter(|codec| tools.can_run(*codec))
    .collect::<Vec<_>>();
    let cases = [
        (Codec::Png, EncoderSettings::Lossless, 0.0),
        (Codec::Jpeg, lossy(95), 3.0),
        (Codec::Webp, EncoderSettings::Lossless, 0.0),
        (Codec::Webp, lossy(95), 3.0),
        (Codec::Avif, EncoderSettings::Lossless, 0.0),
        (Codec::Avif, lossy(90), 3.0),
        (Codec::Jxl, EncoderSettings::Lossless, 0.0),
        (Codec::Jxl, lossy(90), 3.0),
    ];
    for (image, icc, space) in [
        (&source, &icc, "RGB P3"),
        (&grey, &grey_icc, "grey gamma 1"),
    ] {
        let payload = MetadataPayload {
            icc_profile: Some(icc.clone()),
            exif_metadata: None,
        };
        let (_keep, png) = normalized_png(image, &payload).unwrap();
        for (codec, settings, tolerance) in &cases {
            // WebP has no grey mode; the encoder refuses a grey profile it cannot carry.
            let grey_webp = *codec == Codec::Webp && icc.as_slice() == grey_icc;
            if !runnable.contains(codec) {
                continue;
            }
            let path = dir.path().join(format!(
                "{space}-{codec:?}-{}.{}",
                settings.label().replace('/', "_"),
                codec.extension()
            ));
            let result = backend.encode(EncodeJob {
                codec: *codec,
                settings: settings.clone(),
                normalized_png: &png,
                original: image,
                output_path: &path,
                timeout: std::time::Duration::from_secs(60),
                has_alpha: false,
                metadata: &payload,
                threads: 2,
            });
            if grey_webp {
                let error =
                    result.expect_err("a grey profile that is not sRGB cannot go into WebP");
                assert!(error.to_string().contains("cannot carry"), "{error}");
                continue;
            }
            let artifact = result.unwrap();
            let difference = mean_difference(image, &tools.decode(&path));
            let carried = tools.carries(&path, icc);
            println!(
                "{space:12} {codec:?} {:10} {:>6} B {:<22} carries profile: {carried:<5} mean difference {difference:.2}",
                settings.label(),
                artifact.size_bytes,
                artifact.backend
            );
            assert!(
                carried,
                "{space} {codec:?} {}: profile not carried",
                settings.label()
            );
            assert!(
                difference <= *tolerance,
                "{space} {codec:?} {}: difference {difference:.2}",
                settings.label()
            );
        }
    }
}

/// Needs cwebp. A grey profile cannot ride in a WebP. When it is plain sRGB the
/// viewer shows the same grey without it and WebP is made with a warning; any
/// other grey profile would change the grey levels, so the candidate is refused
/// with the reason in the report rather than accepted.
#[test]
fn webp_refuses_grey_profiles_it_cannot_carry() {
    let tools = Tools::detect();
    if !tools.can_run(Codec::Webp) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let grey = DynamicImage::ImageLuma8(GrayImage::from_fn(96, 64, |x, y| {
        image::Luma([(x * 2 + y) as u8])
    }));

    let srgb_grey = grey_profile(Curve::Gamma(2.2));
    let input = write(
        dir.path(),
        "srgb-grey.png",
        &png_with_profile(&grey, Some(&srgb_grey)),
    );
    let report = run(config(
        &input,
        &dir.path().join("srgb"),
        Codec::Webp,
        MetadataPolicy::Safe,
    ));
    assert!(report.output.is_some(), "{:?}", report.errors);
    let winner = &report.candidates[report.winner_index.unwrap()];
    assert!(
        winner
            .warnings
            .iter()
            .any(|w| w.contains("grey ICC profile not attached"))
    );

    let dark_grey = grey_profile(Curve::Gamma(1.0));
    let input = write(
        dir.path(),
        "linear-grey.png",
        &png_with_profile(&grey, Some(&dark_grey)),
    );
    for policy in POLICIES {
        let out = dir.path().join(format!("linear-{policy:?}"));
        let report = run(config(&input, &out, Codec::Webp, policy));
        assert_eq!(report.output, None, "{policy:?}");
        assert!(!report.candidates.is_empty());
        for candidate in &report.candidates {
            assert!(
                format!("{:?}", candidate.status).contains("cannot carry"),
                "{policy:?}: {:?}",
                candidate.status
            );
        }
    }
}

/// Needs ssimulacra2. (c) The external metric is handed two profile-free PNGs,
/// so it applies no conversion to either: a lossless P3 candidate scores as the
/// identical picture it is, where one profile-bearing and one bare file would
/// score badly.
#[test]
fn external_metric_sees_identical_pixels_in_a_wide_gamut_image() {
    if tools_or_skip(&["ssimulacra2"]).is_none() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = write(
        dir.path(),
        "p3.png",
        &png_with_profile(
            &DynamicImage::ImageRgb8(smooth_colours()),
            Some(&display_p3()),
        ),
    );
    for policy in POLICIES {
        let out = dir.path().join(format!("{policy:?}"));
        let report = run(config(&input, &out, Codec::Png, policy));
        let winner = &report.candidates[report.winner_index.expect("a winner")];
        let score = winner
            .quality
            .as_ref()
            .unwrap()
            .ssimulacra2
            .expect("an SSIMULACRA2 score");
        assert!(score > 99.0, "{policy:?}: {score}");
    }
}
