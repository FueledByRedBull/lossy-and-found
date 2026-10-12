//! Input, limit and metadata policy fixtures. Every fixture is generated at
//! test time; nothing here depends on quality scores or on which codec wins.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use image::{
    DynamicImage, ExtendedColorType, ImageDecoder, ImageEncoder, ImageFormat, ImageReader, Rgb,
    RgbImage,
    codecs::{jpeg::JpegEncoder, png::PngEncoder},
    metadata::Orientation,
};
use smart_compress_codecs::Codec;
use smart_compress_core::{CliConfig, CodecSelection, ImageReport, MetadataPolicy, Optimizer};

/// ASCII payload of the GPS date tag in `exif_with_orientation_and_gps`.
const GPS_MARKER: &[u8] = b"1999:12:31";

fn config(input: &Path, out_dir: &Path, codec: Codec) -> CliConfig {
    let mut config = CliConfig {
        inputs: vec![input.to_path_buf()],
        out_dir: Some(out_dir.to_path_buf()),
        formats: CodecSelection::new([codec]),
        allow_larger: true,
        // Accept any decodable candidate; these tests are about policy.
        min_quality: Some(0.0),
        ..CliConfig::default()
    };
    config.gpu.enabled = false;
    config
}

fn run(config: CliConfig) -> ImageReport {
    let mut summary = Optimizer::new(config)
        .run()
        .expect("a bad input must be reported, not fail the run");
    assert_eq!(summary.report.reports.len(), 1);
    summary.report.reports.remove(0)
}

/// Asserts a clean rejection whose message contains every needle (lowercase).
fn assert_rejected(report: &ImageReport, out_dir: &Path, needles: &[&str]) {
    let errors = report.errors.join("; ").to_lowercase();
    assert!(!errors.is_empty(), "{} was not rejected", report.input);
    for needle in needles {
        assert!(
            errors.contains(needle),
            "{}: error does not mention '{needle}': {errors}",
            report.input
        );
    }
    assert_eq!(report.output, None, "{}", report.input);
    assert!(report.candidates.is_empty(), "{}", report.input);
    let published = std::fs::read_dir(out_dir).map_or(0, Iterator::count);
    assert_eq!(published, 0, "{} left files in the output", report.input);
}

fn reject_png_run(input: &Path, adjust: impl FnOnce(&mut CliConfig), needles: &[&str]) {
    let out = tempfile::tempdir().unwrap();
    let mut config = config(input, out.path(), Codec::Png);
    adjust(&mut config);
    assert_rejected(&run(config), out.path(), needles);
}

/// 48x32 gradient with a red block in the top-left corner.
fn test_image() -> RgbImage {
    RgbImage::from_fn(48, 32, |x, y| {
        if x < 8 && y < 8 {
            Rgb([255, 0, 0])
        } else {
            Rgb([(x * 5) as u8, (y * 7) as u8, ((x + y) * 3) as u8])
        }
    })
}

fn png_bytes(image: &RgbImage, icc: Option<&[u8]>) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = PngEncoder::new(&mut bytes);
    if let Some(icc) = icc {
        encoder.set_icc_profile(icc.to_vec()).unwrap();
    }
    encoder
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ExtendedColorType::Rgb8,
        )
        .unwrap();
    bytes
}

fn jpeg_bytes(image: &RgbImage, exif: Option<&[u8]>) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut bytes, 90);
    if let Some(exif) = exif {
        encoder.set_exif_metadata(exif.to_vec()).unwrap();
    }
    encoder
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ExtendedColorType::Rgb8,
        )
        .unwrap();
    bytes
}

/// Stamps an RGB ICC header onto the bytes; profiles that do not claim RGB
/// data are dropped for RGB images.
fn rgb_profile(mut bytes: Vec<u8>) -> Vec<u8> {
    bytes[16..20].copy_from_slice(b"RGB ");
    bytes[36..40].copy_from_slice(b"acsp");
    bytes
}

/// Deterministic incompressible bytes (xorshift64).
fn pseudo_random_bytes(length: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    (0..length)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 32) as u8
        })
        .collect()
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

fn png_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut chunk = (data.len() as u32).to_be_bytes().to_vec();
    chunk.extend_from_slice(kind);
    chunk.extend_from_slice(data);
    chunk.extend_from_slice(&crc32(&chunk[4..]).to_be_bytes());
    chunk
}

/// Little-endian TIFF block: IFD0 holds orientation and a GPS IFD pointer, the
/// GPS IFD holds a latitude reference and a date stamp.
fn exif_with_orientation_and_gps(orientation: u16) -> Vec<u8> {
    let mut exif = b"II*\0".to_vec();
    let entry = |exif: &mut Vec<u8>, tag: u16, kind: u16, count: u32, value: [u8; 4]| {
        exif.extend_from_slice(&tag.to_le_bytes());
        exif.extend_from_slice(&kind.to_le_bytes());
        exif.extend_from_slice(&count.to_le_bytes());
        exif.extend_from_slice(&value);
    };
    exif.extend_from_slice(&8_u32.to_le_bytes());
    exif.extend_from_slice(&2_u16.to_le_bytes());
    entry(
        &mut exif,
        0x0112,
        3,
        1,
        u32::from(orientation).to_le_bytes(),
    );
    entry(&mut exif, 0x8825, 4, 1, 38_u32.to_le_bytes());
    exif.extend_from_slice(&0_u32.to_le_bytes());
    assert_eq!(exif.len(), 38);
    exif.extend_from_slice(&2_u16.to_le_bytes());
    entry(&mut exif, 0x0001, 2, 2, *b"N\0\0\0");
    entry(&mut exif, 0x001D, 2, 11, 68_u32.to_le_bytes());
    exif.extend_from_slice(&0_u32.to_le_bytes());
    assert_eq!(exif.len(), 68);
    exif.extend_from_slice(GPS_MARKER);
    exif.push(0);
    exif
}

struct Output {
    bytes: Vec<u8>,
    image: DynamicImage,
    icc: Option<Vec<u8>>,
    exif: Option<Vec<u8>>,
    orientation: Orientation,
}

/// Reads the published output back with the image crate's own decoder.
fn read_output(report: &ImageReport, extension: &str) -> Output {
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    let path = Path::new(report.output.as_deref().expect("expected an output"));
    assert_eq!(
        path.extension().and_then(|ext| ext.to_str()),
        Some(extension)
    );
    let mut decoder = ImageReader::open(path)
        .unwrap()
        .with_guessed_format()
        .unwrap()
        .into_decoder()
        .unwrap();
    let icc = decoder.icc_profile().unwrap();
    let exif = decoder.exif_metadata().unwrap();
    let orientation = decoder.orientation().unwrap();
    Output {
        bytes: std::fs::read(path).unwrap(),
        image: DynamicImage::from_decoder(decoder).unwrap(),
        icc,
        exif,
        orientation,
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

// (a) Malformed, spoofed and unsupported inputs.

#[test]
fn truncated_png_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("truncated.png");
    let bytes = png_bytes(
        &RgbImage::from_raw(64, 64, pseudo_random_bytes(64 * 64 * 3)).unwrap(),
        None,
    );
    std::fs::write(&input, &bytes[..bytes.len() / 2]).unwrap();
    reject_png_run(&input, |_| {}, &[]);
}

#[test]
fn truncated_jpeg_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("truncated.jpg");
    let bytes = jpeg_bytes(
        &RgbImage::from_raw(64, 64, pseudo_random_bytes(64 * 64 * 3)).unwrap(),
        None,
    );
    std::fs::write(&input, &bytes[..bytes.len() / 2]).unwrap();
    reject_png_run(&input, |_| {}, &[]);
}

#[test]
fn zero_byte_file_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("empty.png");
    std::fs::write(&input, b"").unwrap();
    reject_png_run(&input, |_| {}, &[]);
}

#[test]
fn random_bytes_with_png_extension_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("random.png");
    std::fs::write(&input, pseudo_random_bytes(4096)).unwrap();
    reject_png_run(&input, |_| {}, &[]);
}

#[test]
fn png_renamed_to_jpg_is_rejected_as_spoofed() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("spoofed.jpg");
    std::fs::write(&input, png_bytes(&test_image(), None)).unwrap();
    reject_png_run(&input, |_| {}, &["file extension indicates"]);
}

#[test]
fn gif_and_bmp_inputs_are_rejected_as_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    for (name, format) in [
        ("input.gif", ImageFormat::Gif),
        ("input.bmp", ImageFormat::Bmp),
    ] {
        let input = dir.path().join(name);
        DynamicImage::ImageRgb8(test_image())
            .save_with_format(&input, format)
            .unwrap();
        reject_png_run(&input, |_| {}, &["unsupported input"]);
    }
}

// (b) Decode limits.

#[test]
fn inputs_over_a_configured_limit_are_rejected_naming_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("limits.png");
    let bytes = png_bytes(&test_image(), None);
    std::fs::write(&input, &bytes).unwrap();

    reject_png_run(
        &input,
        |config| config.max_pixels = 48 * 32 - 1,
        &["pixel limit"],
    );
    reject_png_run(&input, |config| config.max_width = 47, &["limit"]);
    reject_png_run(&input, |config| config.max_height = 31, &["limit"]);
    reject_png_run(
        &input,
        |config| config.max_input_bytes = bytes.len() as u64 - 1,
        &["byte limit"],
    );
}

#[test]
fn width_and_height_rejections_name_the_dimension() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("limits.png");
    std::fs::write(&input, png_bytes(&test_image(), None)).unwrap();

    reject_png_run(&input, |config| config.max_width = 47, &["width limit"]);
    reject_png_run(&input, |config| config.max_height = 31, &["height limit"]);
}

#[test]
fn limits_equal_to_the_input_are_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let input = dir.path().join("exact.png");
    let bytes = png_bytes(&test_image(), None);
    std::fs::write(&input, &bytes).unwrap();

    let mut config = config(&input, out.path(), Codec::Png);
    config.max_pixels = 48 * 32;
    config.max_width = 48;
    config.max_height = 32;
    config.max_input_bytes = bytes.len() as u64;
    let output = read_output(&run(config), "png");
    assert_eq!((output.image.width(), output.image.height()), (48, 32));
}

/// A valid PNG header claiming 60000x60000 in front of an empty zlib stream.
fn bomb_png() -> Vec<u8> {
    let mut header = Vec::new();
    header.extend_from_slice(&60_000_u32.to_be_bytes());
    header.extend_from_slice(&60_000_u32.to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    let empty_zlib = [
        0x78, 0x01, 0x01, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01,
    ];
    [
        b"\x89PNG\r\n\x1a\n".to_vec(),
        png_chunk(b"IHDR", &header),
        png_chunk(b"IDAT", &empty_zlib),
        png_chunk(b"IEND", &[]),
    ]
    .concat()
}

#[test]
fn decompression_bomb_header_is_rejected_before_allocation() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("bomb.png");
    std::fs::write(&input, bomb_png()).unwrap();

    // Once by the pixel limit, once by the decoded-byte limit alone.
    type Adjust = fn(&mut CliConfig);
    let cases: [(Adjust, &str); 2] = [
        (|_| {}, "pixel limit"),
        (|config| config.max_pixels = u64::MAX, "byte limit"),
    ];
    for (adjust, needle) in cases {
        let out = tempfile::tempdir().unwrap();
        let mut config = config(&input, out.path(), Codec::Png);
        adjust(&mut config);
        let optimizer = Optimizer::new(config);
        let start = Instant::now();
        let mut summary = optimizer.run().unwrap();
        let elapsed = start.elapsed();
        assert_rejected(&summary.report.reports.remove(0), out.path(), &[needle]);
        // Decoding 10.8 GB of pixels could not finish in this time.
        assert!(elapsed < Duration::from_secs(2), "took {elapsed:?}");
    }
}

// (c) ICC profiles.

#[test]
fn icc_profile_follows_the_metadata_policy() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("icc.png");
    let icc = rgb_profile(pseudo_random_bytes(3144));
    std::fs::write(&input, png_bytes(&test_image(), Some(&icc))).unwrap();

    for (policy, expected) in [
        (MetadataPolicy::Keep, Some(&icc)),
        (MetadataPolicy::Safe, Some(&icc)),
        // Not provably sRGB, so it is colour data and `strip` leaves it (the
        // colour_fidelity tests cover a plain sRGB profile, which it drops).
        (MetadataPolicy::Strip, Some(&icc)),
    ] {
        let out = tempfile::tempdir().unwrap();
        let mut config = config(&input, out.path(), Codec::Png);
        config.metadata = policy;
        let report = run(config);
        let probe = report.probe.as_ref().expect("expected a probe");
        assert_eq!(probe.icc_profile_size, Some(icc.len() as u64), "{policy:?}");
        let output = read_output(&report, "png");
        assert_eq!(output.icc.as_ref(), expected, "{policy:?}");
        assert_eq!(output.image.to_rgb8(), test_image(), "{policy:?}");
    }
}

#[test]
fn garbage_iccp_payload_does_not_crash_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let input = dir.path().join("garbage-icc.png");
    let mut bytes = png_bytes(&test_image(), None);
    // Signature plus IHDR is 33 bytes; iCCP must precede IDAT. The payload
    // after the name and method byte is not a zlib stream.
    let payload = [b"bad\0\0".as_slice(), &pseudo_random_bytes(64)].concat();
    bytes.splice(33..33, png_chunk(b"iCCP", &payload));
    std::fs::write(&input, bytes).unwrap();

    let report = run(config(&input, out.path(), Codec::Png));
    if report.errors.is_empty() {
        println!("garbage iCCP: processed, the broken profile is dropped");
        let output = read_output(&report, "png");
        assert_eq!(output.icc, None);
        assert_eq!(output.image.to_rgb8(), test_image());
    } else {
        println!("garbage iCCP: rejected: {:?}", report.errors);
        assert_rejected(&report, out.path(), &[]);
    }
}

#[test]
fn profile_heavy_png_is_kept_within_the_metadata_limit_and_rejected_over_it() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("heavy-icc.png");
    let icc = rgb_profile(pseudo_random_bytes(1024 * 1024));
    std::fs::write(&input, png_bytes(&test_image(), Some(&icc))).unwrap();

    let out = tempfile::tempdir().unwrap();
    let mut accepted = config(&input, out.path(), Codec::Png);
    accepted.metadata = MetadataPolicy::Keep;
    accepted.max_metadata_bytes = icc.len() as u64;
    let report = run(accepted);
    assert_eq!(
        report.probe.as_ref().unwrap().icc_profile_size,
        Some(icc.len() as u64)
    );
    assert_eq!(read_output(&report, "png").icc, Some(icc.clone()));

    reject_png_run(
        &input,
        |config| config.max_metadata_bytes = icc.len() as u64 - 1,
        &["metadata", "byte limit"],
    );
}

// (d) EXIF orientation and GPS.

fn oriented_jpeg(dir: &Path) -> std::path::PathBuf {
    let input = dir.join("oriented.jpg");
    let exif = exif_with_orientation_and_gps(6);
    assert_eq!(
        Orientation::from_exif_chunk(&exif),
        Some(Orientation::Rotate90)
    );
    std::fs::write(&input, jpeg_bytes(&test_image(), Some(&exif))).unwrap();
    input
}

fn run_oriented(codec: Codec, policy: MetadataPolicy) -> (ImageReport, Output, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let mut config = config(&oriented_jpeg(dir.path()), out.path(), codec);
    config.metadata = policy;
    let report = run(config);
    let output = read_output(&report, codec.extension());
    (report, output, out)
}

#[test]
fn orientation_6_is_applied_and_reported() {
    let (report, output, _out) = run_oriented(Codec::Png, MetadataPolicy::Safe);
    let probe = report.probe.as_ref().expect("expected a probe");
    assert!(probe.orientation_applied);
    assert_eq!((probe.width, probe.height), (32, 48));
    assert_eq!((output.image.width(), output.image.height()), (32, 48));
    assert_eq!(output.orientation, Orientation::NoTransforms);
    // Rotating 90 degrees clockwise moves the red top-left block to the top right.
    let corner = output.image.to_rgb8()[(31, 0)];
    assert!(
        corner[0] > 200 && corner[1] < 60 && corner[2] < 60,
        "{corner:?}"
    );
}

#[test]
fn safe_policy_drops_exif_and_gps_from_png_and_jpeg_outputs() {
    for codec in [Codec::Png, Codec::Jpeg] {
        let (_report, output, _out) = run_oriented(codec, MetadataPolicy::Safe);
        assert_eq!(output.exif, None, "{codec:?}");
        assert_eq!(output.orientation, Orientation::NoTransforms, "{codec:?}");
        assert_eq!(
            (output.image.width(), output.image.height()),
            (32, 48),
            "{codec:?}"
        );
        for marker in [GPS_MARKER, b"Exif\0\0", b"eXIf"] {
            assert!(
                !contains(&output.bytes, marker),
                "{codec:?} output still contains {:?}",
                String::from_utf8_lossy(marker)
            );
        }
    }
}

#[test]
fn keep_policy_retains_exif_without_the_baked_orientation() {
    for codec in [Codec::Png, Codec::Jpeg] {
        let (_report, output, _out) = run_oriented(codec, MetadataPolicy::Keep);
        let exif = output.exif.as_ref().expect("keep must retain EXIF");
        assert_eq!(
            Orientation::from_exif_chunk(exif),
            Some(Orientation::NoTransforms),
            "{codec:?}"
        );
        assert_eq!(output.orientation, Orientation::NoTransforms, "{codec:?}");
        assert_eq!(
            (output.image.width(), output.image.height()),
            (32, 48),
            "{codec:?}"
        );
    }
}

// (e) WebP and AVIF, the other advertised input families.

fn webp_fixture() -> Vec<u8> {
    std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/sample.webp"))
        .unwrap()
}

#[test]
fn corrupt_and_spoofed_webp_inputs_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let webp = webp_fixture();

    let truncated = dir.path().join("truncated.webp");
    std::fs::write(&truncated, &webp[..webp.len() / 2]).unwrap();
    reject_png_run(&truncated, |_| {}, &[]);

    let random = dir.path().join("random.webp");
    std::fs::write(&random, pseudo_random_bytes(4096)).unwrap();
    reject_png_run(&random, |_| {}, &[]);

    let webp_as_png = dir.path().join("webp-inside.png");
    std::fs::write(&webp_as_png, &webp).unwrap();
    reject_png_run(&webp_as_png, |_| {}, &["file extension indicates"]);

    let png_as_webp = dir.path().join("png-inside.webp");
    std::fs::write(&png_as_webp, png_bytes(&test_image(), None)).unwrap();
    reject_png_run(&png_as_webp, |_| {}, &["file extension indicates"]);
}

#[test]
fn corrupt_spoofed_and_animated_avif_inputs_are_rejected() {
    let dir = tempfile::tempdir().unwrap();

    let random = dir.path().join("random.avif");
    std::fs::write(&random, pseudo_random_bytes(4096)).unwrap();
    reject_png_run(&random, |_| {}, &[]);

    let png_as_avif = dir.path().join("png-inside.avif");
    std::fs::write(&png_as_avif, png_bytes(&test_image(), None)).unwrap();
    reject_png_run(&png_as_avif, |_| {}, &[]);

    // The remaining cases need a real AVIF, which needs the encoder.
    let Some(avifenc) =
        smart_compress_codecs::tools_or_skip(&["avifenc"]).and_then(|tools| tools.avifenc.path)
    else {
        return;
    };
    let frame = dir.path().join("frame.png");
    std::fs::write(&frame, png_bytes(&test_image(), None)).unwrap();
    let encode = |frames: &[&Path], output: &Path| {
        let status = std::process::Command::new(&avifenc)
            .args(frames)
            .arg(output)
            .output()
            .unwrap()
            .status;
        assert!(status.success(), "avifenc failed for {}", output.display());
    };

    let still = dir.path().join("still.avif");
    encode(&[&frame], &still);
    let accepted_out = tempfile::tempdir().unwrap();
    let report = run(config(&still, accepted_out.path(), Codec::Png));
    assert!(report.errors.is_empty(), "{:?}", report.errors);

    let truncated = dir.path().join("truncated.avif");
    let bytes = std::fs::read(&still).unwrap();
    std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
    reject_png_run(&truncated, |_| {}, &[]);

    let animated = dir.path().join("animated.avif");
    encode(&[&frame, &frame], &animated);
    reject_png_run(&animated, |_| {}, &["animated avif"]);
}
