//! Colour-profile and colour-model fixtures: grey profiles, profiles that do
//! not match the pixels, and a four-component CMYK JPEG through the pipeline.

use std::path::{Path, PathBuf};

use image::{
    ColorType, DynamicImage, ExtendedColorType, GrayImage, ImageDecoder, ImageEncoder, ImageReader,
    RgbImage, codecs::png::PngEncoder,
};
use smart_compress_codecs::Codec;
use smart_compress_core::{
    CliConfig, CodecSelection, ImageReport, MetadataPolicy, Optimizer,
    image_probe::icc_profile_mismatch,
};

/// A header-only ICC profile for the given colour space signature.
fn icc_profile(space: &[u8; 4]) -> Vec<u8> {
    let mut profile = vec![0_u8; 132];
    profile[0..4].copy_from_slice(&132_u32.to_be_bytes());
    profile[8] = 2;
    profile[12..16].copy_from_slice(b"mntr");
    profile[16..20].copy_from_slice(space);
    profile[20..24].copy_from_slice(b"XYZ ");
    profile[36..40].copy_from_slice(b"acsp");
    profile
}

fn png(pixels: &[u8], width: u32, height: u32, color: ExtendedColorType, icc: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = PngEncoder::new(&mut bytes);
    encoder.set_icc_profile(icc.to_vec()).unwrap();
    encoder.write_image(pixels, width, height, color).unwrap();
    bytes
}

fn grey_png(icc: &[u8]) -> Vec<u8> {
    let image = GrayImage::from_fn(48, 32, |x, y| image::Luma([(x * 5 + y * 3) as u8]));
    png(image.as_raw(), 48, 32, ExtendedColorType::L8, icc)
}

fn colour_png(icc: &[u8]) -> Vec<u8> {
    let image = RgbImage::from_fn(48, 32, |x, y| {
        image::Rgb([(x * 5) as u8, (y * 7) as u8, ((x + y) * 3) as u8])
    });
    png(image.as_raw(), 48, 32, ExtendedColorType::Rgb8, icc)
}

struct Outcome {
    report: ImageReport,
    image: DynamicImage,
    icc: Option<Vec<u8>>,
}

fn optimize(input: &Path, out_dir: &Path, codec: Codec, policy: MetadataPolicy) -> Outcome {
    let mut config = CliConfig {
        inputs: vec![input.to_path_buf()],
        out_dir: Some(out_dir.to_path_buf()),
        formats: CodecSelection::new([codec]),
        allow_larger: true,
        // These tests are about colour handling, not about quality.
        min_quality: Some(0.0),
        metadata: policy,
        ..CliConfig::default()
    };
    config.gpu.enabled = false;
    let mut summary = Optimizer::new(config).run().unwrap();
    let report = summary.report.reports.remove(0);
    assert!(report.errors.is_empty(), "{codec:?}: {:?}", report.errors);
    let path = PathBuf::from(report.output.as_deref().expect("expected an output"));
    let mut decoder = ImageReader::open(&path)
        .unwrap()
        .with_guessed_format()
        .unwrap()
        .into_decoder()
        .unwrap();
    let icc = decoder.icc_profile().unwrap();
    let image = DynamicImage::from_decoder(decoder).unwrap();
    Outcome { report, image, icc }
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn grey_profile_is_kept_only_where_the_output_stays_grey() {
    let dir = tempfile::tempdir().unwrap();
    let profile = icc_profile(b"GRAY");
    let input = write(dir.path(), "grey.png", &grey_png(&profile));

    for policy in [MetadataPolicy::Keep, MetadataPolicy::Safe] {
        for codec in [Codec::Png, Codec::Jpeg] {
            let out = dir.path().join(format!("{codec:?}-{policy:?}"));
            let outcome = optimize(&input, &out, codec, policy);
            assert!(
                outcome.report.warnings.is_empty(),
                "{:?}",
                outcome.report.warnings
            );
            assert_eq!(
                outcome.report.probe.as_ref().unwrap().icc_profile_size,
                Some(profile.len() as u64)
            );
            assert_eq!(
                outcome.icc.as_deref(),
                Some(profile.as_slice()),
                "{codec:?} {policy:?}"
            );
            // A profile we write must describe the pixels we write.
            assert_eq!(
                icc_profile_mismatch(&profile, outcome.image.color()),
                None,
                "{codec:?} {policy:?} wrote {:?} via {:?}",
                outcome.image.color(),
                outcome.report.candidates[outcome.report.winner_index.unwrap()]
            );
        }
    }

    // A header-only profile is not provably sRGB grey, so `strip` keeps it too.
    let stripped = optimize(
        &input,
        &dir.path().join("strip"),
        Codec::Png,
        MetadataPolicy::Strip,
    );
    assert_eq!(stripped.icc.as_deref(), Some(profile.as_slice()));

    // WebP has no grey mode, so it cannot carry this grey profile, and dropping
    // one that is not plain sRGB would change the greys: no WebP is produced.
    // (A plain sRGB grey profile is dropped with a warning; see colour_fidelity.)
    if smart_compress_codecs::tools_or_skip(&["cwebp"]).is_some() {
        let mut config = CliConfig {
            inputs: vec![input.clone()],
            out_dir: Some(dir.path().join("webp")),
            formats: CodecSelection::new([Codec::Webp]),
            allow_larger: true,
            min_quality: Some(0.0),
            metadata: MetadataPolicy::Keep,
            ..CliConfig::default()
        };
        config.gpu.enabled = false;
        let mut summary = Optimizer::new(config).run().unwrap();
        let report = summary.report.reports.remove(0);
        assert_eq!(report.output, None);
        assert!(
            report
                .candidates
                .iter()
                .all(|candidate| format!("{:?}", candidate.status).contains("cannot carry")),
            "{:?}",
            report
                .candidates
                .iter()
                .map(|c| &c.status)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn profiles_that_do_not_match_the_pixels_are_dropped_with_a_warning() {
    let dir = tempfile::tempdir().unwrap();
    let cases = [
        (
            "grey-profile-on-colour.png",
            colour_png(&icc_profile(b"GRAY")),
            "GRAY",
        ),
        (
            "colour-profile-on-grey.png",
            grey_png(&icc_profile(b"RGB ")),
            "RGB",
        ),
        (
            "cmyk-profile-on-colour.png",
            colour_png(&icc_profile(b"CMYK")),
            "CMYK",
        ),
        (
            "lab-profile-on-colour.png",
            colour_png(&icc_profile(b"Lab ")),
            "Lab",
        ),
    ];
    for (name, bytes, space) in cases {
        let input = write(dir.path(), name, &bytes);
        for policy in [MetadataPolicy::Keep, MetadataPolicy::Safe] {
            let out = dir.path().join(format!("{name}-{policy:?}"));
            let outcome = optimize(&input, &out, Codec::Png, policy);
            assert_eq!(outcome.icc, None, "{name} {policy:?}");
            assert!(
                outcome
                    .report
                    .warnings
                    .iter()
                    .any(|warning| warning.contains("ICC profile dropped")
                        && warning.contains(space)),
                "{name} {policy:?}: {:?}",
                outcome.report.warnings
            );
        }
    }

    // A matching profile is left alone.
    let input = write(
        dir.path(),
        "matching.png",
        &colour_png(&icc_profile(b"RGB ")),
    );
    let outcome = optimize(
        &input,
        &dir.path().join("matching"),
        Codec::Png,
        MetadataPolicy::Safe,
    );
    assert!(outcome.report.warnings.is_empty());
    assert_eq!(outcome.icc, Some(icc_profile(b"RGB ")));
}

#[test]
fn mismatch_rule_reads_only_the_profile_header() {
    assert_eq!(
        icc_profile_mismatch(&icc_profile(b"RGB "), ColorType::Rgb8),
        None
    );
    assert_eq!(
        icc_profile_mismatch(&icc_profile(b"RGB "), ColorType::Rgba16),
        None
    );
    assert_eq!(
        icc_profile_mismatch(&icc_profile(b"GRAY"), ColorType::La8),
        None
    );
    assert!(icc_profile_mismatch(&icc_profile(b"GRAY"), ColorType::Rgb8).is_some());
    assert!(icc_profile_mismatch(b"not a profile", ColorType::Rgb8).is_some());
}

fn cmyk_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/cmyk.jpg")
        .canonicalize()
        .unwrap()
}

#[test]
fn cmyk_jpeg_is_normalized_to_rgb_and_optimized() {
    let dir = tempfile::tempdir().unwrap();
    let input = write(
        dir.path(),
        "cmyk.jpg",
        &std::fs::read(cmyk_fixture()).unwrap(),
    );
    let reference = image::open(&input).unwrap();
    assert_eq!(reference.color(), ColorType::Rgb8);

    // Lossless output must carry exactly the pixels the decoder produced.
    let lossless = optimize(
        &input,
        &dir.path().join("png"),
        Codec::Png,
        MetadataPolicy::Safe,
    );
    let probe = lossless.report.probe.as_ref().unwrap();
    assert!(
        probe
            .original_color_type
            .to_ascii_lowercase()
            .contains("cmyk"),
        "{}",
        probe.original_color_type
    );
    assert_eq!(probe.color_type, "Rgb8");
    assert_eq!(lossless.image.to_rgb8(), reference.to_rgb8());
    assert!(
        lossless
            .report
            .warnings
            .iter()
            .any(|warning| warning.contains("CMYK source converted to RGB")),
        "{:?}",
        lossless.report.warnings
    );

    // Regression pin for the decoder's CMYK to RGB conversion. The benchmark
    // corpus tests check the same file against its source pixels.
    let pixels = reference.to_rgb8();
    assert_eq!(
        blake3::hash(pixels.as_raw()).to_hex().as_str(),
        include_str!("../../../tests/fixtures/cmyk.rgb.blake3").trim(),
        "CMYK to RGB conversion changed"
    );

    // A lossy re-encode is a three-component image again.
    let lossy = optimize(
        &input,
        &dir.path().join("jpeg"),
        Codec::Jpeg,
        MetadataPolicy::Safe,
    );
    assert_eq!(lossy.image.color(), ColorType::Rgb8);
    assert_eq!(
        (lossy.image.width(), lossy.image.height()),
        (reference.width(), reference.height())
    );
}
