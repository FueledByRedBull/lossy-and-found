use std::path::{Path, PathBuf};

use image::{
    DynamicImage, ExtendedColorType, ImageEncoder, Rgb, RgbImage, codecs::png::PngEncoder,
    metadata::Orientation,
};
use smart_compress_codecs::Codec;
use smart_compress_core::{
    AlphaPolicy, CliConfig, CodecSelection, Preset, ReportFormat, TargetPolicy,
    image_probe::{ContentClass, apply_orientation, decode_and_probe, decode_source},
    scoring::{CpuRayonBackend, GpuWgpuBackend, gpu_or_skip},
};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn fixture(name: &str) -> PathBuf {
    fixture_dir().join(name)
}

/// The 199.8 MP source for the ignored large-image test, named by
/// `SMART_COMPRESS_LARGE_IMAGE`. The image is not part of the repository.
fn known_large_image() -> PathBuf {
    std::env::var_os("SMART_COMPRESS_LARGE_IMAGE")
        .map(PathBuf::from)
        .expect("set SMART_COMPRESS_LARGE_IMAGE to the path of a large JPEG (199.8 MP) to run this test")
}

fn exif_chunk(orientation: Orientation) -> Vec<u8> {
    let value = orientation.to_exif() as u16;
    let mut exif = Vec::new();
    exif.extend_from_slice(b"II*\0");
    exif.extend_from_slice(&8_u32.to_le_bytes());
    exif.extend_from_slice(&1_u16.to_le_bytes());
    exif.extend_from_slice(&0x0112_u16.to_le_bytes());
    exif.extend_from_slice(&3_u16.to_le_bytes());
    exif.extend_from_slice(&1_u32.to_le_bytes());
    exif.extend_from_slice(&value.to_le_bytes());
    exif.extend_from_slice(&0_u16.to_le_bytes());
    exif.extend_from_slice(&0_u32.to_le_bytes());
    exif
}

/// An RGB ICC header followed by a distinguishing tail; profiles that do not
/// claim RGB data are dropped for RGB images.
fn test_icc(tail: &[u8]) -> Vec<u8> {
    let mut profile = vec![0_u8; 128];
    profile[16..20].copy_from_slice(b"RGB ");
    profile[36..40].copy_from_slice(b"acsp");
    profile.extend_from_slice(tail);
    profile
}

fn write_png_with_metadata(path: &Path, image: &RgbImage, icc: &[u8], exif: &[u8]) {
    let file = std::fs::File::create(path).unwrap();
    let mut encoder = PngEncoder::new(file);
    encoder.set_icc_profile(icc.to_vec()).unwrap();
    encoder.set_exif_metadata(exif.to_vec()).unwrap();
    DynamicImage::ImageRgb8(image.clone())
        .write_with_encoder(encoder)
        .unwrap();
}

#[test]
fn checked_in_fixture_corpus_decodes_and_probes() {
    let fixtures = [
        ("photo-like.jpg", "Jpeg", false),
        ("screenshot-like.png", "Png", false),
        ("flat-icon.png", "Png", false),
        ("gradient.png", "Png", false),
        ("noisy.png", "Png", false),
        ("transparent.png", "Png", true),
        ("sample.webp", "WebP", false),
        ("cmyk.jpg", "Jpeg", false),
    ];

    for (name, expected_format, expected_alpha) in fixtures {
        let path = fixture(name);
        assert!(path.is_file(), "missing fixture {}", path.display());
        let (_image, probe) = decode_and_probe(&path, 10_000_000).unwrap();
        assert_eq!(probe.format.as_deref(), Some(expected_format), "{name}");
        assert_eq!(probe.has_alpha, expected_alpha, "{name}");
        assert!(probe.width > 0 && probe.height > 0, "{name}");
        assert!(!probe.original_color_type.is_empty(), "{name}");
    }
}

#[test]
fn transparent_fixture_is_classified_as_transparent_asset() {
    let (_image, probe) = decode_and_probe(&fixture("transparent.png"), 10_000_000).unwrap();
    assert_eq!(probe.content_class, ContentClass::TransparentAsset);
    assert!(probe.alpha_coverage > 0.0);
}

/// Every checked-in fixture is under the 128 x 128 px floor below which the
/// colour statistics are not trusted, so only alpha sets their class. The real
/// photographs and the synthetic graphics cover the classifier itself.
#[test]
fn content_class_matches_labelled_images() {
    use ContentClass::{Photo, Screenshot, SimpleGraphic, TransparentAsset, Unknown};
    let class_of = |path: &Path| decode_and_probe(path, 100_000_000).unwrap().1.content_class;

    for (name, expected) in [
        ("cmyk.jpg", Unknown),
        ("flat-icon.png", Unknown),
        ("gradient.png", Unknown),
        ("noisy.png", Unknown),
        ("photo-like.jpg", Unknown),
        ("sample.webp", Unknown),
        ("screenshot-like.png", Unknown),
        ("tiny.png", TransparentAsset),
        ("transparent.png", TransparentAsset),
    ] {
        assert_eq!(class_of(&fixture(name)), expected, "{name}");
    }

    let photos = fixture_dir().join("../../samples/photos");
    let mut count = 0;
    for entry in std::fs::read_dir(photos).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|extension| extension == "jpg") {
            assert_eq!(class_of(&path), Photo, "{}", path.display());
            count += 1;
        }
    }
    assert_eq!(count, 10, "samples/photos should hold ten photographs");

    let temp = tempfile::tempdir().unwrap();
    let synthetic = |name: &str, pixel: fn(u32, u32) -> [u8; 3]| {
        let path = temp.path().join(name);
        RgbImage::from_fn(512, 512, |x, y| Rgb(pixel(x, y)))
            .save(&path)
            .unwrap();
        class_of(&path)
    };
    let flat = |x: u32, y: u32| match (x < 256, y < 256) {
        (true, true) => [255, 0, 0],
        (false, true) => [0, 255, 0],
        (true, false) => [0, 0, 255],
        (false, false) => [255, 255, 255],
    };
    assert_eq!(synthetic("flat.png", flat), SimpleGraphic);
    let stripes = |x: u32, _y: u32| {
        if x.is_multiple_of(2) {
            [0; 3]
        } else {
            [255; 3]
        }
    };
    assert_eq!(synthetic("stripes.png", stripes), Screenshot);
    let ramp = |x: u32, y: u32| [(x / 2) as u8, (y / 2) as u8, 128];
    assert_eq!(synthetic("ramp.png", ramp), Unknown);
}

#[test]
fn unsupported_legacy_input_families_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let out_dir = temp.path().join("out");
    for name in ["sample.bmp", "sample.gif", "sample.tiff"] {
        let config = CliConfig {
            inputs: vec![fixture(name)],
            out_dir: Some(out_dir.clone()),
            ..CliConfig::default()
        };
        let summary = smart_compress_core::Optimizer::new(config).run().unwrap();
        assert_eq!(summary.report.failures, 1, "{name}");
        assert!(summary.report.reports[0].errors.iter().any(|error| {
            error.contains("unsupported input format")
                || error.contains("unsupported input extension")
        }));
    }
}

#[test]
fn cpu_only_and_gpu_default_choose_same_png_winner_for_checked_in_fixtures_when_available() {
    if gpu_or_skip(GpuWgpuBackend::new()).is_none() {
        return;
    }

    for name in [
        "photo-like.jpg",
        "screenshot-like.png",
        "transparent.png",
        "sample.webp",
    ] {
        let cpu = png_only_winner_size(name, false);
        let gpu = png_only_winner_size(name, true);
        assert_eq!(cpu, gpu, "{name}");
    }
}

#[test]
fn probe_reports_original_color_type_and_icc_profile_size() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("with-icc.jpg");
    let image = RgbImage::from_pixel(2, 2, Rgb([10, 20, 30]));
    let icc = test_icc(b"smart-compress-test-icc-profile");
    let mut bytes = Vec::new();
    {
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 90);
        encoder.set_icc_profile(icc.clone()).unwrap();
        encoder
            .write_image(image.as_raw(), 2, 2, ExtendedColorType::Rgb8)
            .unwrap();
    }
    std::fs::write(&input, bytes).unwrap();

    let (_image, probe) = decode_and_probe(&input, 10_000_000).unwrap();
    assert_eq!(probe.original_color_type, "Rgb8");
    assert_eq!(probe.icc_profile_size, Some(icc.len() as u64));
    assert_eq!(probe.metadata_size, icc.len() as u64);
}

#[test]
fn probe_metadata_size_counts_embedded_icc_and_exif_not_the_file() {
    let temp = tempfile::tempdir().unwrap();
    let image = RgbImage::from_pixel(8, 6, Rgb([12, 34, 56]));
    let icc = test_icc(b"smart-compress-metadata-size");
    let exif = exif_chunk(Orientation::Rotate90);

    let tagged = temp.path().join("tagged.png");
    write_png_with_metadata(&tagged, &image, &icc, &exif);
    let (_image, probe) = decode_and_probe(&tagged, 10_000_000).unwrap();
    assert_eq!(probe.metadata_size, (icc.len() + exif.len()) as u64);
    assert!(probe.metadata_size < std::fs::metadata(&tagged).unwrap().len());

    let plain = temp.path().join("plain.png");
    image.save(&plain).unwrap();
    let (_image, probe) = decode_and_probe(&plain, 10_000_000).unwrap();
    assert_eq!(probe.metadata_size, 0);
}

#[test]
fn metadata_policy_safe_preserves_icc_in_png_output() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("safe-input.png");
    let image = RgbImage::from_pixel(8, 6, Rgb([12, 34, 56]));
    let icc = test_icc(b"smart-compress-safe-icc");
    write_png_with_metadata(&input, &image, &icc, &exif_chunk(Orientation::Rotate90));

    let metadata = optimize_metadata_fixture(
        &input,
        Codec::Png,
        smart_compress_core::MetadataPolicy::Safe,
    );

    assert_eq!(metadata.icc_profile, Some(icc));
    assert!(metadata.exif_metadata.is_none());
    assert_eq!(metadata.orientation, Orientation::NoTransforms);
}

#[test]
fn metadata_policy_keep_preserves_icc_and_clears_orientation_in_png_output() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("keep-input.png");
    let image = RgbImage::from_pixel(9, 7, Rgb([90, 45, 30]));
    let icc = test_icc(b"smart-compress-keep-icc");
    write_png_with_metadata(&input, &image, &icc, &exif_chunk(Orientation::Rotate90));

    let metadata = optimize_metadata_fixture(
        &input,
        Codec::Png,
        smart_compress_core::MetadataPolicy::Keep,
    );

    assert_eq!(metadata.icc_profile, Some(icc));
    let exif = metadata.exif_metadata.expect("expected EXIF metadata");
    assert_eq!(
        Orientation::from_exif_chunk(&exif),
        Some(Orientation::NoTransforms)
    );
    assert_eq!(metadata.orientation, Orientation::NoTransforms);
}

#[test]
fn metadata_policy_keep_preserves_icc_and_clears_orientation_in_jpeg_output() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("keep-input.png");
    let image = RgbImage::from_pixel(12, 10, Rgb([25, 100, 200]));
    let icc = test_icc(b"smart-compress-jpeg-icc");
    write_png_with_metadata(&input, &image, &icc, &exif_chunk(Orientation::Rotate90));

    let metadata = optimize_metadata_fixture(
        &input,
        Codec::Jpeg,
        smart_compress_core::MetadataPolicy::Keep,
    );

    assert_eq!(metadata.icc_profile, Some(icc));
    let exif = metadata.exif_metadata.expect("expected EXIF metadata");
    assert_eq!(
        Orientation::from_exif_chunk(&exif),
        Some(Orientation::NoTransforms)
    );
    assert_eq!(metadata.orientation, Orientation::NoTransforms);
}

#[test]
fn metadata_policy_keep_preserves_icc_and_clears_orientation_in_webp_output() {
    if smart_compress_codecs::tools_or_skip(&["cwebp"]).is_none() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("keep-input.png");
    let image = RgbImage::from_pixel(11, 9, Rgb([70, 40, 180]));
    let icc = test_icc(b"smart-compress-webp-icc");
    write_png_with_metadata(&input, &image, &icc, &exif_chunk(Orientation::Rotate90));

    let metadata = optimize_metadata_fixture(
        &input,
        Codec::Webp,
        smart_compress_core::MetadataPolicy::Keep,
    );

    assert_eq!(metadata.icc_profile, Some(icc));
    let exif = metadata.exif_metadata.expect("expected EXIF metadata");
    assert_eq!(
        Orientation::from_exif_chunk(&exif),
        Some(Orientation::NoTransforms)
    );
    assert_eq!(metadata.orientation, Orientation::NoTransforms);
}

#[test]
fn malformed_image_reports_failure_without_panicking() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("not-an-image.jpg");
    std::fs::write(&input, b"this is not a jpeg").unwrap();

    let config = CliConfig {
        inputs: vec![input],
        dry_run: true,
        report: ReportFormat::Json,
        ..CliConfig::default()
    };

    let summary = smart_compress_core::Optimizer::new(config).run().unwrap();
    assert_eq!(summary.report.failures, 1);
    assert_eq!(summary.report.reports.len(), 1);
    assert!(!summary.report.reports[0].errors.is_empty());
}

#[test]
fn gpu_and_cpu_scores_match_checked_in_fixture_corpus_when_available() {
    let Some(gpu) = gpu_or_skip(GpuWgpuBackend::new()) else {
        return;
    };
    let cpu = CpuRayonBackend;
    let fixtures = [
        "photo-like.jpg",
        "screenshot-like.png",
        "flat-icon.png",
        "gradient.png",
        "noisy.png",
        "transparent.png",
        "sample.webp",
    ];

    for name in fixtures {
        let (original, probe) = decode_and_probe(&fixture(name), 10_000_000).unwrap();
        let candidate = perturbed_candidate(&original);
        let cpu_score = cpu
            .score_candidate(&original, &candidate, probe.has_alpha)
            .unwrap();
        let gpu_score = gpu
            .score_candidate(&original, &candidate, probe.has_alpha)
            .unwrap();

        assert_close(name, "rgb", cpu_score.rgb, gpu_score.rgb);
        assert_close(name, "edge", cpu_score.edge, gpu_score.edge);
        assert_close(name, "overall", cpu_score.overall, gpu_score.overall);
        assert_optional_close(name, "alpha", cpu_score.alpha, gpu_score.alpha);
        assert_optional_close(
            name,
            "composite",
            cpu_score.composite_min,
            gpu_score.composite_min,
        );
    }
}

#[test]
#[ignore = "uses the local 199.8 MP image"]
fn gpu_prepared_scores_match_known_large_image() {
    // The test is ignored, so running it is deliberate: a missing input fails.
    let path = known_large_image();
    assert!(path.is_file(), "missing large image {}", path.display());

    let Some(gpu) = gpu_or_skip(GpuWgpuBackend::new()) else {
        return;
    };
    let cpu = CpuRayonBackend;
    let mut decoded = decode_source(&path, 250_000_000).unwrap();
    let prepared = gpu
        .prepare_original(&decoded.image, decoded.metadata.orientation)
        .unwrap();
    apply_orientation(&mut decoded.image, decoded.metadata.orientation);
    let original = decoded.image;
    let candidate = perturbed_candidate(&original);

    let cpu_score = cpu.score_candidate(&original, &candidate, false).unwrap();
    let gpu_score = gpu
        .score_candidate_prepared_with_timings(&prepared, &candidate, false)
        .unwrap()
        .0;

    assert_close("known-large", "rgb", cpu_score.rgb, gpu_score.rgb);
    assert_close("known-large", "edge", cpu_score.edge, gpu_score.edge);
    assert_close(
        "known-large",
        "overall",
        cpu_score.overall,
        gpu_score.overall,
    );
}

fn assert_output_exists(output: Option<&str>) {
    let output = output.expect("expected output path");
    assert!(Path::new(output).is_file(), "missing output {output}");
}

fn optimize_metadata_fixture(
    input: &Path,
    codec: Codec,
    metadata: smart_compress_core::MetadataPolicy,
) -> smart_compress_core::image_probe::SourceMetadata {
    let temp = tempfile::tempdir().unwrap();
    let config = CliConfig {
        inputs: vec![input.to_path_buf()],
        out_dir: Some(temp.path().join("out")),
        preset: Preset::Balanced,
        target: TargetPolicy::MaxCompat,
        formats: CodecSelection::new([codec]),
        report: ReportFormat::Json,
        alpha: AlphaPolicy::Preserve,
        allow_larger: true,
        max_pixels: 10_000_000,
        metadata,
        ..CliConfig::default()
    };

    let summary = smart_compress_core::Optimizer::new(config).run().unwrap();
    assert_eq!(summary.report.failures, 0, "{:?}", input);
    let report = &summary.report.reports[0];
    let output = report.output.as_ref().expect("expected output").clone();
    let output_path = PathBuf::from(output);
    assert!(
        output_path.is_file(),
        "missing output {}",
        output_path.display()
    );
    decode_source(&output_path, 10_000_000).unwrap().metadata
}

fn png_only_winner_size(name: &str, gpu_enabled: bool) -> u64 {
    let temp = tempfile::tempdir().unwrap();
    let mut config = CliConfig {
        inputs: vec![fixture(name)],
        out_dir: Some(temp.path().join("out")),
        preset: Preset::Balanced,
        target: TargetPolicy::MaxCompat,
        formats: CodecSelection::new([Codec::Png]),
        report: ReportFormat::Json,
        alpha: AlphaPolicy::Preserve,
        allow_larger: true,
        max_pixels: 10_000_000,
        ..CliConfig::default()
    };
    config.gpu.enabled = gpu_enabled;

    let summary = smart_compress_core::Optimizer::new(config).run().unwrap();
    assert_eq!(summary.report.failures, 0, "{name}");
    let report = &summary.report.reports[0];
    let winner = report
        .winner_index
        .and_then(|index| report.candidates.get(index))
        .expect("expected winner");
    assert_eq!(winner.codec, Codec::Png, "{name}");
    assert_output_exists(report.output.as_deref());
    winner.size_bytes.expect("winner size")
}

fn perturbed_candidate(image: &DynamicImage) -> DynamicImage {
    let mut rgba = image.to_rgba8();
    if rgba.width() == 0 || rgba.height() == 0 {
        return DynamicImage::ImageRgba8(rgba);
    }

    let points = [
        (0, 0),
        (rgba.width() / 2, rgba.height() / 2),
        (rgba.width() - 1, rgba.height() - 1),
    ];
    for (x, y) in points {
        let pixel = rgba.get_pixel_mut(x, y);
        pixel[0] = pixel[0].saturating_add(3);
        pixel[1] = pixel[1].saturating_sub(2);
        pixel[2] = pixel[2].saturating_add(1);
        pixel[3] = pixel[3].saturating_sub(1);
    }

    DynamicImage::ImageRgba8(rgba)
}

fn assert_optional_close(name: &str, label: &str, cpu: Option<f32>, gpu: Option<f32>) {
    match (cpu, gpu) {
        (Some(cpu), Some(gpu)) => assert_close(name, label, cpu, gpu),
        (None, None) => {}
        _ => panic!("{name} {label} option mismatch: cpu={cpu:?} gpu={gpu:?}"),
    }
}

fn assert_close(name: &str, label: &str, cpu: f32, gpu: f32) {
    let delta = (cpu - gpu).abs();
    assert!(
        delta <= 0.05,
        "{name} {label} differs: cpu={cpu:.6} gpu={gpu:.6} delta={delta:.6}"
    );
}
