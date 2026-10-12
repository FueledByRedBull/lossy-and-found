//! Cross-codec decisions on real encoder output: lossless winners, lossy
//! winners, and alpha handling across the quality range. Codecs whose tools
//! are not installed are left out of each run, unless
//! `SMART_COMPRESS_REQUIRE_TOOLS` is set, which fails the test instead.

use std::path::{Path, PathBuf};

use image::{DynamicImage, Rgb, RgbImage, Rgba, RgbaImage};
use smart_compress_codecs::{Codec, EncoderSettings, ToolInventory, tool_names_for, tools_or_skip};
use smart_compress_core::{
    CliConfig, CodecSelection, ImageReport, Optimizer, TargetPolicy, candidate::CandidateStatus,
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn available(codecs: &[Codec]) -> Vec<Codec> {
    let tools = ToolInventory::detect();
    let present = codecs
        .iter()
        .copied()
        .filter(|codec| tools.available_for(*codec))
        .collect::<Vec<_>>();
    if present.len() < codecs.len() {
        // Says what is left out, or fails when the tools are required.
        let _ = tools_or_skip(&tool_names_for(codecs));
    }
    present
}

fn optimize(
    input: &Path,
    out_dir: &Path,
    codecs: &[Codec],
    adjust: impl FnOnce(&mut CliConfig),
) -> ImageReport {
    let mut config = CliConfig {
        inputs: vec![input.to_path_buf()],
        out_dir: Some(out_dir.to_path_buf()),
        formats: CodecSelection::new(codecs.iter().copied()),
        // Size alone decides between formats in these tests.
        target: TargetPolicy::Smallest,
        ..CliConfig::default()
    };
    config.gpu.enabled = false;
    adjust(&mut config);
    let mut summary = Optimizer::new(config).run().unwrap();
    summary.report.reports.remove(0)
}

fn accepted(
    report: &ImageReport,
) -> impl Iterator<Item = &smart_compress_core::candidate::CandidateReport> {
    report
        .candidates
        .iter()
        .filter(|candidate| matches!(candidate.status, CandidateStatus::Accepted))
}

#[test]
fn lossless_winner_keeps_every_pixel_across_png_webp_and_jxl() {
    let codecs = available(&[Codec::Png, Codec::Webp, Codec::Jxl]);
    for name in ["flat-icon.png", "screenshot-like.png", "gradient.png"] {
        let dir = tempfile::tempdir().unwrap();
        let report = optimize(&fixture(name), dir.path(), &codecs, |config| {
            // Only a lossless candidate can reach this floor.
            config.min_quality = Some(99.99);
            config.allow_larger = true;
        });
        assert!(report.errors.is_empty(), "{name}: {:?}", report.errors);
        let winner = &report.candidates[report.winner_index.unwrap()];
        assert!(
            matches!(
                winner.settings,
                EncoderSettings::Lossless | EncoderSettings::Palette { .. }
            ),
            "{name}: {:?}",
            winner.settings
        );
        // Every candidate labelled lossless must score as an exact copy.
        for candidate in &report.candidates {
            if let (EncoderSettings::Lossless, Some(quality)) =
                (&candidate.settings, candidate.quality)
            {
                assert_eq!(quality.overall, 100.0, "{name} {:?}", candidate.codec);
            }
        }
        // No accepted candidate is smaller than the winner.
        let smallest = accepted(&report)
            .filter_map(|candidate| candidate.size_bytes)
            .min();
        assert_eq!(winner.size_bytes, smallest, "{name}");
        // Formats the image crate can read back are compared pixel by pixel.
        if matches!(winner.codec, Codec::Png | Codec::Webp) {
            let output = image::open(report.output.as_deref().unwrap()).unwrap();
            let source = image::open(fixture(name)).unwrap();
            assert_eq!(output.to_rgba8(), source.to_rgba8(), "{name}");
        }
    }
}

/// Smooth colour with fine texture: cheap for lossy codecs, costly losslessly.
fn textured_photo() -> RgbImage {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    RgbImage::from_fn(256, 256, |x, y| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let noise = (state % 24) as u32;
        Rgb([
            (x * 2 / 3 + noise) as u8,
            (y * 2 / 3 + noise) as u8,
            ((x + y) / 3 + noise) as u8,
        ])
    })
}

#[test]
fn lossy_winner_is_the_smallest_candidate_that_passes() {
    // With JPEG alone nothing need pass the floor, so this needs every format.
    let codecs = [Codec::Jpeg, Codec::Webp, Codec::Avif, Codec::Jxl];
    if tools_or_skip(&tool_names_for(&codecs)).is_none() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("photo.png");
    textured_photo().save(&input).unwrap();
    let report = optimize(&input, &dir.path().join("out"), &codecs, |_| {});
    assert!(report.errors.is_empty(), "{:?}", report.errors);

    let floor = report.min_quality.unwrap();
    let winner = &report.candidates[report.winner_index.unwrap()];
    assert!(
        matches!(winner.settings, EncoderSettings::Quality { .. }),
        "{:?} {:?}",
        winner.codec,
        winner.settings
    );
    assert!(winner.quality.unwrap().overall >= floor);
    assert!(winner.size_bytes.unwrap() < report.original_size);
    assert_eq!(
        winner.size_bytes,
        accepted(&report)
            .filter_map(|candidate| candidate.size_bytes)
            .min()
    );
    // Every codec in the run produced at least one scored lossy candidate.
    for codec in &codecs {
        assert!(
            report.candidates.iter().any(|candidate| {
                candidate.codec == *codec
                    && candidate.quality.is_some()
                    && matches!(candidate.settings, EncoderSettings::Quality { .. })
            }),
            "{codec:?} was never scored"
        );
    }
    // Verdicts follow the floor exactly.
    for candidate in &report.candidates {
        match (&candidate.status, candidate.quality) {
            (CandidateStatus::Accepted, Some(quality)) => assert!(quality.overall >= floor),
            (CandidateStatus::Rejected(_), Some(quality)) => {
                assert!(quality.overall < floor);
            }
            _ => {}
        }
    }
}

#[test]
fn lossy_alpha_is_checked_at_every_quality() {
    let codecs = available(&[Codec::Webp, Codec::Avif]);
    if codecs.is_empty() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("soft-alpha.png");
    // A disc with a soft, anti-aliased edge over texture.
    RgbaImage::from_fn(96, 96, |x, y| {
        let distance = (((x as f32 - 48.0).powi(2) + (y as f32 - 48.0).powi(2)).sqrt() - 30.0)
            .clamp(0.0, 12.0);
        let alpha = (255.0 * (1.0 - distance / 12.0)) as u8;
        Rgba([
            (x * 2) as u8,
            (y * 2) as u8,
            ((x * 7 + y * 3) % 256) as u8,
            alpha,
        ])
    })
    .save(&input)
    .unwrap();
    let report = optimize(&input, &dir.path().join("out"), &codecs, |config| {
        // Score every candidate, including ones that cannot win.
        config.verbose = true;
        config.allow_larger = true;
    });
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(report.probe.as_ref().unwrap().has_alpha);

    let floor = report.min_quality.unwrap();
    let mut scored = 0;
    for candidate in &report.candidates {
        let Some(quality) = candidate.quality else {
            continue;
        };
        scored += 1;
        // Transparency is always measured, never assumed.
        let alpha = quality.alpha.expect("alpha score for a transparent image");
        if matches!(candidate.status, CandidateStatus::Accepted) {
            assert!(
                alpha >= floor,
                "{:?} {:?}",
                candidate.codec,
                candidate.settings
            );
        }
        assert!(quality.overall <= alpha);
    }
    assert!(
        scored >= codecs.len() * 3,
        "only {scored} candidates were scored"
    );

    // The published file still carries transparency where it can be read back.
    let winner = &report.candidates[report.winner_index.unwrap()];
    if winner.codec == Codec::Webp {
        let output: DynamicImage = image::open(report.output.as_deref().unwrap()).unwrap();
        assert!(output.color().has_alpha());
        assert_eq!(output.to_rgba8().get_pixel(0, 0)[3], 0);
        assert_eq!(output.to_rgba8().get_pixel(48, 48)[3], 255);
    }
}
