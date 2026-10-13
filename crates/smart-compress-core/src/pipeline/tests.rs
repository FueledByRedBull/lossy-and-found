use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Instant,
};

use anyhow::{Result, anyhow};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use smart_compress_codecs::{Codec, EncodeError, EncoderSettings, ToolInventory, WaitEnded};
use smart_compress_metrics::QualityScore;

use super::*;
use super::{
    evaluate::encode_failure,
    plan::{lossless_gate, stage_plan},
    predict::{Outcome, Step, Track, predicted_quality, readings},
    score::{RemoveOnDrop, bounded_metric_image, write_metric_png},
    search::{
        Inversion, InversionKind, coarse_followup, held_back_to_run, inversions, neighbour_probes,
        quality_bracket,
    },
};
use crate::{
    candidate::{
        ActualScoreBackend, CandidateInfluence, CandidateReport, CandidateStatus, QualitySource,
    },
    config::{
        AlphaPolicy, CacheConfig, CliConfig, CodecSelection, MetadataPolicy, Preset, PrimaryMetric,
        ReportFormat, TargetPolicy,
    },
    decision::{AcceptedCandidate, DecisionPolicy},
    image_probe::{ContentClass, ImageProbe},
    progress::{NoopProgress, ProgressObserver},
    report::{ImageOutcome, ImageReport},
    scoring::{GpuWgpuBackend, gpu_or_skip},
};

#[test]
fn dry_run_reports_probe_without_candidates() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 16, Rgb([120, 40, 10])));
    image
        .save_with_format(&input, image::ImageFormat::Png)
        .unwrap();

    let config = CliConfig {
        inputs: vec![input],
        dry_run: true,
        report: ReportFormat::Json,
        ..CliConfig::default()
    };
    let summary = Optimizer::new(config).run().unwrap();
    assert_eq!(summary.report.failures, 0);
    assert!(summary.report.reports[0].probe.is_some());
    assert!(summary.report.reports[0].candidates.is_empty());
}

#[test]
fn native_png_pipeline_writes_atomic_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    let out_dir = dir.path().join("out");
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([10, 20, 30])));
    image
        .save_with_format(&input, image::ImageFormat::Png)
        .unwrap();

    let config = CliConfig {
        inputs: vec![input],
        out_dir: Some(out_dir.clone()),
        preset: Preset::Balanced,
        target: TargetPolicy::MaxCompat,
        formats: CodecSelection::new([Codec::Png]),
        allow_larger: true,
        max_pixels: 100_000_000,
        ..CliConfig::default()
    };
    let summary = Optimizer::new(config).run().unwrap();
    assert_eq!(summary.report.failures, 0);
    let report = &summary.report.reports[0];
    assert!(report.winner_index.is_some());
    assert!(
        report
            .output
            .as_ref()
            .is_some_and(|path| path.ends_with("input.png"))
    );
    assert!(out_dir.join("input.png").exists());
}

#[test]
fn quality_bracket_finds_lowest_accepted_and_nearest_rejected() {
    let candidates = vec![
        quality_candidate(Codec::Jxl, 74, CandidateStatus::Accepted),
        quality_candidate(
            Codec::Jxl,
            68,
            CandidateStatus::Rejected("quality 89.0 below 90.0".to_string()),
        ),
        quality_candidate(Codec::Jxl, 86, CandidateStatus::Accepted),
        quality_candidate(
            Codec::Webp,
            50,
            CandidateStatus::Rejected("quality 80.0 below 90.0".to_string()),
        ),
    ];

    let mut bracket = quality_bracket(&candidates, Codec::Jxl).unwrap();
    assert_eq!(bracket.low.quality, 68);
    assert_eq!(bracket.high.quality, 74);
    assert!(bracket.seen.contains(&68));
    assert!(bracket.seen.contains(&74));

    // Scores of 76 at q68 and 82 at q74 cross a floor of 80 two thirds
    // of the way up; sizes interpolate geometrically.
    bracket.low.score = Some(76.0);
    bracket.high.score = Some(82.0);
    bracket.low.size = Some(1_000);
    bracket.high.size = Some(4_000);
    assert!((bracket.crossing(80.0) - 72.0).abs() < 0.01);
    assert!((bracket.estimated_size(71.0).unwrap() - 2_000.0).abs() < 1.0);
    bracket.low.score = None;
    assert_eq!(bracket.crossing(80.0), 71.0);
}

fn lossy(codec: Codec, quality: u8, size: u64, status: CandidateStatus) -> CandidateReport {
    let mut candidate = quality_candidate(codec, quality, status);
    candidate.size_bytes = Some(size);
    candidate
}

fn failed() -> CandidateStatus {
    CandidateStatus::Rejected("quality 70.0 below 80.0".to_string())
}

fn smallest_policy() -> DecisionPolicy {
    DecisionPolicy {
        min_quality: 80.0,
        quality_slack: 0.0,
        allow_larger: false,
        original_size: 10_000,
        prefer_compatibility_delta: 0.05,
        target: crate::config::TargetPolicy::Smallest,
    }
}

fn best_of(codec: Codec, size_bytes: u64) -> AcceptedCandidate {
    AcceptedCandidate {
        codec,
        size_bytes,
        quality: QualityScore::lossless(),
    }
}

#[test]
fn staged_plan_holds_back_all_but_the_middle_quality() {
    let quality = |quality| EncoderSettings::Quality { quality, effort: 0 };
    let plan = vec![
        (Codec::Webp, EncoderSettings::Lossless),
        (Codec::Webp, quality(92)),
        (Codec::Webp, quality(78)),
        (Codec::Webp, quality(60)),
        (Codec::Jpeg, quality(88)),
        (Codec::Jpeg, quality(70)),
        (Codec::Png, EncoderSettings::Palette { colors: 64 }),
    ];

    let staged = stage_plan(plan.clone(), false);

    assert_eq!(
        staged.first,
        vec![(Codec::Webp, quality(78)), (Codec::Jpeg, quality(70))]
    );
    assert_eq!(staged.held.len(), 5);
    assert_eq!(staged.held[0], (Codec::Webp, EncoderSettings::Lossless));
    assert_eq!(staged.mids[&Codec::Webp], 78);
    assert_eq!(staged.mids[&Codec::Jpeg], 70);
    assert!(!staged.mids.contains_key(&Codec::Png));

    // The lossless probe joins the first stage, after the lossy ones.
    let gated = stage_plan(plan, true);
    assert_eq!(
        gated.first,
        vec![
            (Codec::Webp, quality(78)),
            (Codec::Jpeg, quality(70)),
            (Codec::Webp, EncoderSettings::Lossless),
        ]
    );
    assert_eq!(gated.held.len(), 4);
}

fn lossless_candidate(codec: Codec, size: Option<u64>) -> CandidateReport {
    let mut candidate = quality_candidate(codec, 0, CandidateStatus::Accepted);
    candidate.settings = EncoderSettings::Lossless;
    candidate.size_bytes = size;
    candidate
}

#[test]
fn lossless_gate_compares_the_probe_with_the_largest_winning_size() {
    let policy = smallest_policy();
    let gate =
        |probe: Option<u64>, codec, settings: EncoderSettings, best: Option<&AcceptedCandidate>| {
            let candidates = [lossless_candidate(Codec::Webp, probe)];
            lossless_gate(&candidates, codec, &settings, &policy, best)
        };
    let lossless = || EncoderSettings::Lossless;

    // Against the original (10_000 bytes): the margin is 3x and exclusive.
    assert_eq!(gate(Some(30_000), Codec::Jxl, lossless(), None), None);
    let reason = gate(Some(30_001), Codec::Jxl, lossless(), None).unwrap();
    assert!(
        reason.starts_with("cannot win: lossless WebP is 30001 bytes"),
        "{reason}"
    );
    // Against a best accepted candidate that is smaller than the original.
    let best = best_of(Codec::Jpeg, 2_000);
    assert_eq!(
        gate(Some(6_000), Codec::Avif, lossless(), Some(&best)),
        None
    );
    assert!(gate(Some(6_001), Codec::Avif, lossless(), Some(&best)).is_some());
    // The probe, lossy candidates and a probe without a size are never gated.
    assert_eq!(gate(Some(99_999), Codec::Webp, lossless(), None), None);
    let lossy = EncoderSettings::Quality {
        quality: 70,
        effort: 0,
    };
    assert_eq!(gate(Some(99_999), Codec::Avif, lossy, None), None);
    assert_eq!(gate(None, Codec::Jxl, lossless(), None), None);
    // Nothing bounds the winner when larger outputs are allowed and none passed yet.
    let unbounded = DecisionPolicy {
        allow_larger: true,
        ..smallest_policy()
    };
    let candidates = [lossless_candidate(Codec::Webp, Some(99_999))];
    assert_eq!(
        lossless_gate(
            &candidates,
            Codec::Jxl,
            &EncoderSettings::Lossless,
            &unbounded,
            None
        ),
        None
    );
}

/// Counts progress the way the terminal bar and the GUI do.
#[derive(Default)]
struct ProgressTrace {
    planned: AtomicUsize,
    total: AtomicUsize,
    finished: AtomicUsize,
    at_file_end: Mutex<Option<(usize, usize)>>,
}

impl ProgressObserver for ProgressTrace {
    fn candidates_planned(&self, _path: &Path, total: usize) {
        self.planned.store(total, Ordering::SeqCst);
        self.total.store(total, Ordering::SeqCst);
    }
    fn candidate_started(&self, _: &Path, _: usize, total: usize, _: Codec, _: &EncoderSettings) {
        self.total.fetch_max(total, Ordering::SeqCst);
    }
    fn candidate_finished(&self, _path: &Path, _index: usize, _total: usize) {
        self.finished.fetch_add(1, Ordering::SeqCst);
    }
    fn candidates_settled(&self, _path: &Path) {
        self.total
            .store(self.finished.load(Ordering::SeqCst), Ordering::SeqCst);
    }
    fn file_finished(&self, _path: &Path) {
        *self.at_file_end.lock().unwrap() = Some((
            self.finished.load(Ordering::SeqCst),
            self.total.load(Ordering::SeqCst),
        ));
    }
}

#[test]
fn a_finished_image_has_completed_every_candidate_it_counts() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, y| {
        Rgb([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8])
    }))
    .save_with_format(&input, image::ImageFormat::Png)
    .unwrap();
    let run = |label: &str, adjust: fn(&mut CliConfig)| {
        let mut config = CliConfig {
            inputs: vec![input.clone()],
            out_dir: Some(dir.path().join(label)),
            formats: CodecSelection::new([Codec::Jpeg]),
            allow_larger: true,
            min_quality: Some(0.0),
            workers: 2,
            gpu: crate::config::GpuConfig {
                enabled: false,
                ..crate::config::GpuConfig::default()
            },
            ..CliConfig::default()
        };
        adjust(&mut config);
        let trace = Arc::new(ProgressTrace::default());
        Optimizer::with_progress(config, trace.clone())
            .run()
            .unwrap();
        let (finished, total) = trace.at_file_end.lock().unwrap().unwrap();
        let planned = trace.planned.load(Ordering::SeqCst);
        println!("{label}: planned {planned}, finished {finished}, total at the end {total}");
        assert_eq!(finished, total, "{label}");
        assert!(finished > 0, "{label}");
        (finished, planned)
    };

    // The higher quality is assumed to pass, so fewer candidates finish than were planned.
    let (finished, planned) = run("staged", |config| {
        config.speed = crate::SpeedPreset::Fast;
    });
    assert!(finished < planned);
    // A budget that stops the search early leaves the rest unstarted.
    let (finished, planned) = run("budget", |config| config.max_candidate_attempts = Some(1));
    assert!(finished < planned);
    // The one-at-a-time path skips a settled format's higher qualities.
    let (finished, planned) = run("serial", |config| {
        config.speed = crate::SpeedPreset::Fast;
        config.max_temp_bytes = Some(u64::MAX);
    });
    assert!(finished < planned);
}

fn noisy_source(dir: &Path, name: &str) -> PathBuf {
    // Smooth colour under noise: tiny as a JPEG, large to store exactly.
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % 16) as u32
    };
    let path = dir.join(name);
    RgbImage::from_fn(96, 96, |x, y| {
        Rgb([
            (x + noise()) as u8,
            (y + noise()) as u8,
            ((x + y) / 2 + noise()) as u8,
        ])
    })
    .save(&path)
    .unwrap();
    path
}

/// The lossless candidates of a report: (codec, size if encoded, skip reason if gated).
fn lossless_outcomes(report: &ImageReport) -> Vec<(Codec, Option<u64>, Option<String>)> {
    report
        .candidates
        .iter()
        .filter(|candidate| candidate.settings == EncoderSettings::Lossless)
        .map(|candidate| {
            let gated = match &candidate.status {
                CandidateStatus::Skipped(reason) if reason.starts_with("cannot win: lossless") => {
                    Some(reason.clone())
                }
                _ => None,
            };
            (candidate.codec, candidate.size_bytes, gated)
        })
        .collect()
}

#[test]
fn lossless_encoders_are_gated_on_webp_for_jpeg_sources_only() {
    let tools = ToolInventory::detect();
    let every_codec = [
        Codec::Jpeg,
        Codec::Webp,
        Codec::Avif,
        Codec::Jxl,
        Codec::Png,
    ];
    let codecs = every_codec
        .into_iter()
        .filter(|codec| tools.available_for(*codec))
        .collect::<Vec<_>>();
    if codecs.len() < every_codec.len() {
        // Says what is missing, or fails when the tools are required.
        let _ = smart_compress_codecs::tools_or_skip(&smart_compress_codecs::tool_names_for(
            &every_codec,
        ));
    }
    if !codecs.contains(&Codec::Webp) || codecs.len() < 4 {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let jpeg = noisy_source(dir.path(), "source.jpg");
    let png = noisy_source(dir.path(), "source.png");
    let run = |input: &Path, label: &str, formats: &[Codec], adjust: fn(&mut CliConfig)| {
        let mut config = CliConfig {
            inputs: vec![input.to_path_buf()],
            out_dir: Some(dir.path().join(label)),
            formats: CodecSelection::new(formats.iter().copied()),
            workers: 4,
            gpu: crate::config::GpuConfig {
                enabled: false,
                ..crate::config::GpuConfig::default()
            },
            ..CliConfig::default()
        };
        adjust(&mut config);
        let mut summary = Optimizer::new(config).run().unwrap();
        summary.report.reports.remove(0)
    };
    let others = |outcomes: &[(Codec, Option<u64>, Option<String>)]| {
        outcomes
            .iter()
            .filter(|(codec, ..)| *codec != Codec::Webp)
            .cloned()
            .collect::<Vec<_>>()
    };
    let all_gated = |report: &ImageReport| {
        let outcomes = lossless_outcomes(report);
        assert!(outcomes.len() >= 3, "{outcomes:?}");
        // The probe is encoded, and every other lossless encoder is skipped unencoded.
        let probe = outcomes
            .iter()
            .find(|(codec, ..)| *codec == Codec::Webp)
            .unwrap();
        assert!(probe.1.is_some() && probe.2.is_none(), "{outcomes:?}");
        assert!(
            probe.1.unwrap() > 3 * report.original_size,
            "fixture too compressible"
        );
        for (codec, size, gated) in others(&outcomes) {
            assert!(
                size.is_none() && gated.is_some(),
                "{codec:?}: {size:?} {gated:?}"
            );
        }
        assert!(report.winner_index.is_some(), "{:?}", report.errors);
        let winner = &report.candidates[report.winner_index.unwrap()];
        assert_ne!(winner.settings, EncoderSettings::Lossless);
    };
    let none_gated = |report: &ImageReport| {
        let outcomes = lossless_outcomes(report);
        assert!(outcomes.len() >= 3, "{outcomes:?}");
        for (codec, size, gated) in outcomes {
            assert!(
                size.is_some() && gated.is_none(),
                "{codec:?}: {size:?} {gated:?}"
            );
        }
    };

    // Concurrent search, one-at-a-time search.
    all_gated(&run(&jpeg, "staged", &codecs, |_| {}));
    all_gated(&run(&jpeg, "serial", &codecs, |config| {
        config.max_temp_bytes = Some(u64::MAX);
    }));
    // Verbose mode evaluates everything.
    none_gated(&run(&jpeg, "verbose", &codecs, |config| {
        config.verbose = true
    }));
    // A source that may already be lossless runs every lossless encoder.
    none_gated(&run(&png, "png", &codecs, |_| {}));
    // With larger outputs allowed and nothing passing a high floor, lossless may be the only choice.
    let strict = run(&jpeg, "strict", &codecs, |config| {
        config.allow_larger = true;
        config.min_quality = Some(99.99);
    });
    none_gated(&strict);
    assert_eq!(
        strict.candidates[strict.winner_index.unwrap()].settings,
        EncoderSettings::Lossless
    );
    // Without a second lossless encoder there is nothing to gate.
    let alone = run(&jpeg, "alone", &[Codec::Jpeg, Codec::Webp], |_| {});
    let outcomes = lossless_outcomes(&alone);
    assert_eq!(outcomes.len(), 1);
    assert!(outcomes[0].1.is_some() && outcomes[0].2.is_none());
}

#[test]
fn coarse_followup_skips_the_side_the_middle_quality_rules_out() {
    let policy = smallest_policy();
    let followup = |candidates: &[CandidateReport], best: Option<&AcceptedCandidate>, q| {
        coarse_followup(candidates, &policy, best, Codec::Jpeg, 78, q)
    };

    // A passing middle quality makes the higher one pointless and the lower one necessary.
    let passed = [lossy(Codec::Jpeg, 78, 500, CandidateStatus::Accepted)];
    assert!(
        followup(&passed, None, 92)
            .unwrap()
            .starts_with("assumed to pass")
    );
    assert_eq!(followup(&passed, None, 60), None);

    // A failing one needs the higher quality and rules the lower one out.
    let failing = [lossy(Codec::Jpeg, 78, 500, failed())];
    assert!(
        followup(&failing, None, 60)
            .unwrap()
            .starts_with("assumed to fail")
    );
    assert_eq!(followup(&failing, None, 92), None);
    // ... unless it is already no smaller than the best, so the format cannot win.
    let best = best_of(Codec::Avif, 400);
    assert!(
        followup(&failing, Some(&best), 92)
            .unwrap()
            .starts_with("cannot win")
    );
    let best = best_of(Codec::Avif, 600);
    assert_eq!(followup(&failing, Some(&best), 92), None);

    // Without a verdict on the middle quality, nothing is assumed.
    let unknown = [lossy(
        Codec::Jpeg,
        78,
        500,
        CandidateStatus::Skipped("encoder crashed".to_string()),
    )];
    assert_eq!(followup(&unknown, None, 92), None);
    assert_eq!(followup(&unknown, None, 60), None);
}

#[test]
fn a_candidate_that_failed_to_run_has_no_verdict_whatever_its_message_says() {
    let policy = smallest_policy();
    // A timeout, a decode failure, and a metric error that begins like a
    // quality verdict: none of them says anything about the quality.
    for message in [
        "tools/ssimulacra2.exe Timeout after 120000 ms: terminated",
        "decode failed: opening candidate",
        "quality metric failed: GPU is required",
        "quality 12.0 below 80.0",
    ] {
        let broken = [lossy(
            Codec::Jpeg,
            78,
            500,
            CandidateStatus::Failed(message.to_string()),
        )];
        assert!(!broken[0].failed_quality(), "{message}");
        for quality in [92, 60] {
            assert_eq!(
                coarse_followup(&broken, &policy, None, Codec::Jpeg, 78, quality),
                None,
                "{message}, q{quality}"
            );
        }
        // It is not an end of the bracket either: the upper end is the next one up.
        let candidates = [
            broken[0].clone(),
            lossy(Codec::Jpeg, 92, 900, CandidateStatus::Accepted),
        ];
        let bracket = quality_bracket(&candidates, Codec::Jpeg).unwrap();
        assert_eq!((bracket.low.quality, bracket.high.quality), (1, 92));
        assert!(inversions(&candidates, Codec::Jpeg).is_empty());
    }
    // A rejection for size is a verdict: the format is known to be too large.
    let too_large = lossy(
        Codec::Jpeg,
        78,
        20_000,
        CandidateStatus::Rejected("not smaller than original".to_string()),
    );
    assert!(!too_large.failed_quality());
    assert!(
        coarse_followup(&[too_large], &policy, None, Codec::Jpeg, 78, 92)
            .unwrap()
            .starts_with("assumed to pass")
    );
}

#[test]
fn a_dry_run_report_is_not_resumed_by_a_real_run_and_a_missing_output_is_redone() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    gradient(&input);
    let out = dir.path().join("out");
    let config = CliConfig {
        inputs: vec![input],
        out_dir: Some(out.clone()),
        formats: CodecSelection::new([Codec::Png]),
        report: ReportFormat::Json,
        ..base_config()
    };

    // A dry run leaves a report; a real run asked to resume it still runs.
    let dry_report = dir.path().join("dry.json");
    Optimizer::new(CliConfig {
        dry_run: true,
        report_out: Some(dry_report.clone()),
        ..config.clone()
    })
    .run()
    .unwrap();
    let real_report = dir.path().join("real.json");
    let real = Optimizer::new(CliConfig {
        report_out: Some(real_report.clone()),
        resume_report: Some(dry_report),
        ..config.clone()
    })
    .run()
    .unwrap();
    assert!(out.join("input.png").is_file());
    assert_eq!(real.report.reports[0].outcome, ImageOutcome::Optimized);
    assert!(
        real.report
            .manifest
            .unwrap()
            .resume_decision
            .unwrap()
            .contains("does not match")
    );

    // The same run resumed finds its output and reuses the image ...
    let resumed = Optimizer::new(CliConfig {
        report_out: Some(dir.path().join("again.json")),
        resume_report: Some(real_report.clone()),
        ..config.clone()
    })
    .run()
    .unwrap();
    assert_eq!(
        resumed.report.manifest.unwrap().resume_decision.as_deref(),
        Some("resumed 1 compatible completed input(s)")
    );
    // ... unless the output has gone, which makes it do the work again.
    fs::remove_file(out.join("input.png")).unwrap();
    let redone = Optimizer::new(CliConfig {
        report_out: Some(dir.path().join("third.json")),
        resume_report: Some(real_report),
        ..config
    })
    .run()
    .unwrap();
    assert_eq!(
        redone.report.manifest.unwrap().resume_decision.as_deref(),
        Some("resumed 0 compatible completed input(s)")
    );
    assert!(out.join("input.png").is_file());
}

#[test]
fn the_resume_fingerprint_covers_where_and_whether_outputs_are_written() {
    let fingerprint = |adjust: fn(&mut CliConfig)| {
        let mut config = CliConfig::default();
        adjust(&mut config);
        Optimizer::new(config).config_fingerprint
    };
    let base = fingerprint(|_| {});
    for (label, changed) in [
        ("dry_run", fingerprint(|config| config.dry_run = true)),
        (
            "out_dir",
            fingerprint(|config| config.out_dir = Some(PathBuf::from("elsewhere"))),
        ),
        (
            "out",
            fingerprint(|config| config.out = Some(PathBuf::from("one.webp"))),
        ),
        ("overwrite", fingerprint(|config| config.overwrite = true)),
        (
            "protect_inputs",
            fingerprint(|config| config.protect_inputs = false),
        ),
    ] {
        assert_ne!(base, changed, "{label}");
    }
}

#[test]
fn a_second_run_skips_what_is_already_there_without_searching_or_failing() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    gradient(&input);
    let config = CliConfig {
        inputs: vec![input],
        formats: CodecSelection::new([Codec::Png]),
        ..base_config()
    };

    let first = Optimizer::new(config.clone()).run().unwrap();
    assert_eq!(first.report.reports[0].outcome, ImageOutcome::Optimized);
    let written = fs::read(dir.path().join("optimized").join("input.png")).unwrap();

    let second = Optimizer::new(config.clone()).run().unwrap();
    let report = &second.report.reports[0];
    assert_eq!(second.report.failures, 0);
    assert_eq!(report.outcome, ImageOutcome::Skipped);
    assert!(report.candidates.is_empty() && report.errors.is_empty());
    assert!(report.warnings[0].starts_with("output already exists"));
    assert!(
        second
            .report
            .outcome_summary()
            .starts_with("0 improved, 0 kept as they were, 1 skipped, 0 failed")
    );
    assert_eq!(
        fs::read(dir.path().join("optimized").join("input.png")).unwrap(),
        written
    );

    // With overwriting on, the image is searched again and the file replaced.
    let third = Optimizer::new(CliConfig {
        overwrite: true,
        ..config
    })
    .run()
    .unwrap();
    assert_eq!(third.report.reports[0].outcome, ImageOutcome::Optimized);
}

#[test]
fn a_walk_includes_hidden_and_ignored_images_once_and_leaves_its_own_outputs_alone() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    for path in [
        "visible.png",
        ".hidden/inside.png",
        "ignored/skipped.png",
        "optimized/earlier-output.png",
        "nested/optimized/earlier-output.png",
        "nested/deeper/kept.png",
    ] {
        let path = root.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        gradient(&path);
    }
    fs::write(
        root.join(".gitignore"),
        "ignored/
",
    )
    .unwrap();
    fs::write(
        root.join(".ignore"),
        "nested/deeper/
",
    )
    .unwrap();
    let names = |config: CliConfig| {
        let (files, warnings) = Optimizer::new(config).discover_inputs().unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        let mut names = files
            .iter()
            .map(|file| {
                file.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect::<Vec<_>>();
        names.sort();
        names
    };
    let walk = CliConfig {
        // The same folder twice, and a file in it that the folder also finds.
        inputs: vec![
            root.to_path_buf(),
            root.to_path_buf(),
            root.join("visible.png"),
        ],
        recursive: true,
        ..base_config()
    };

    assert_eq!(
        names(walk.clone()),
        [
            ".hidden/inside.png",
            "ignored/skipped.png",
            "nested/deeper/kept.png",
            "visible.png"
        ]
    );
    // With an output folder named, a folder called "optimized" is a user's
    // folder and only the output folder itself is left out.
    let with_out_dir = names(CliConfig {
        out_dir: Some(root.join("optimized")),
        ..walk
    });
    assert!(with_out_dir.contains(&"nested/optimized/earlier-output.png".to_string()));
    assert!(!with_out_dir.contains(&"optimized/earlier-output.png".to_string()));
}

#[test]
fn out_names_one_file_and_its_extension_decides_the_format() {
    let dir = tempfile::tempdir().unwrap();
    let input = noisy_source(dir.path(), "source.png");
    let run = |out: &str| {
        Optimizer::new(CliConfig {
            inputs: vec![input.clone()],
            out: Some(dir.path().join(out)),
            formats: CodecSelection::new([Codec::Png, Codec::Jpeg]),
            min_quality: Some(0.0),
            ..base_config()
        })
        .run()
    };

    // A JPEG name gets a JPEG: only that format is tried.
    let summary = run("named.jpg").unwrap();
    let report = &summary.report.reports[0];
    assert!(report.candidates.iter().all(|c| c.codec == Codec::Jpeg));
    assert_eq!(
        &fs::read(dir.path().join("named.jpg")).unwrap()[..2],
        &[0xff, 0xd8]
    );
    assert_eq!(report.outcome, ImageOutcome::Optimized);

    // A name that names no format gets the winner's extension added.
    let summary = run("named.bin").unwrap();
    let report = &summary.report.reports[0];
    let output = report.output.as_ref().unwrap();
    assert!(
        output.ends_with("named.bin.png") || output.ends_with("named.bin.jpg"),
        "{output}"
    );
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.starts_with("--out has no image extension"))
    );
    assert!(!dir.path().join("named.bin").exists());

    // A format that is not selected is refused before any work.
    let error = Optimizer::new(CliConfig {
        inputs: vec![input.clone()],
        out: Some(dir.path().join("x.webp")),
        formats: CodecSelection::new([Codec::Png]),
        ..base_config()
    })
    .run()
    .unwrap_err();
    assert!(
        error.to_string().contains("names the webp format"),
        "{error}"
    );

    // So is one file for several images.
    let second = noisy_source(dir.path(), "second.png");
    let error = Optimizer::new(CliConfig {
        inputs: vec![input, second],
        out: Some(dir.path().join("only.png")),
        ..base_config()
    })
    .run()
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("--out names one file, but 2 input images"),
        "{error}"
    );
    let folder = Optimizer::new(CliConfig {
        inputs: vec![dir.path().to_path_buf()],
        recursive: true,
        out: Some(dir.path().join("only.png")),
        ..base_config()
    })
    .run()
    .unwrap_err();
    assert!(
        folder.to_string().contains("--out names one file"),
        "{folder}"
    );
}

#[test]
fn only_a_failed_image_is_a_failure_of_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let fine = noisy_source(dir.path(), "fine.png");
    let broken = dir.path().join("broken.png");
    fs::write(&broken, b"this is not an image").unwrap();
    let transparent = dir.path().join("transparent.png");
    image::RgbaImage::from_pixel(16, 16, image::Rgba([1, 2, 3, 100]))
        .save(&transparent)
        .unwrap();
    let run = |inputs: Vec<PathBuf>, adjust: fn(&mut CliConfig)| {
        let mut config = CliConfig {
            inputs,
            out_dir: Some(dir.path().join("out")),
            formats: CodecSelection::new([Codec::Jpeg]),
            allow_larger: true,
            ..base_config()
        };
        adjust(&mut config);
        Optimizer::new(config).run().unwrap().report
    };

    let improved = run(vec![fine.clone()], |config| config.min_quality = Some(0.0));
    assert_eq!(
        (improved.failures, improved.reports[0].outcome),
        (0, ImageOutcome::Optimized)
    );

    // Nothing passes the floor: the original is kept, which is no failure.
    let kept = run(vec![fine.clone()], |config| {
        config.min_quality = Some(1000.0);
        config.out_dir = Some(config.out_dir.clone().unwrap().join("kept"));
    });
    assert_eq!(
        (kept.failures, kept.reports[0].outcome),
        (0, ImageOutcome::Kept)
    );
    assert!(kept.reports[0].errors.is_empty() && kept.reports[0].winner_index.is_none());
    assert!(
        kept.reports[0]
            .warnings
            .iter()
            .any(|w| w.starts_with("nothing scored at least 1000.0"))
    );
    assert!(
        kept.outcome_summary()
            .starts_with("0 improved, 1 kept as they were, 0 skipped, 0 failed")
    );

    // An image that cannot be decoded fails, and so does one with no candidate to try.
    let report = run(vec![broken], |_| {});
    assert_eq!(
        (report.failures, report.reports[0].outcome),
        (1, ImageOutcome::Failed)
    );
    let report = run(vec![transparent], |_| {});
    assert_eq!(
        (report.failures, report.reports[0].outcome),
        (1, ImageOutcome::Failed)
    );
    assert!(
        report.reports[0].errors[0].starts_with(
            "no candidate could be planned for this image: jpeg cannot store transparency"
        ),
        "{:?}",
        report.reports[0].errors
    );
}

#[test]
fn a_budget_that_stops_the_search_is_a_warning() {
    let dir = tempfile::tempdir().unwrap();
    let input = noisy_source(dir.path(), "source.png");
    let report = Optimizer::new(CliConfig {
        inputs: vec![input],
        out_dir: Some(dir.path().join("out")),
        formats: CodecSelection::new([Codec::Jpeg]),
        min_quality: Some(0.0),
        max_candidate_attempts: Some(1),
        ..base_config()
    })
    .run()
    .unwrap()
    .report;
    let image = &report.reports[0];
    assert_eq!(
        image.search_stop_reason.as_deref(),
        Some("candidate attempt budget of 1 reached")
    );
    assert!(image.warnings.iter().any(|warning| {
        warning.starts_with("the search stopped early (candidate attempt budget of 1 reached)")
    }));
    assert!(
        report
            .to_text()
            .contains("Search stopped early: candidate attempt budget of 1 reached")
    );
    assert!(
        report
            .to_markdown()
            .contains("Search stopped early: candidate attempt budget of 1 reached")
    );
}

#[test]
fn an_encoder_that_declines_is_a_skip_and_any_other_error_is_a_failure() {
    let declined = anyhow!(EncodeError::Declined("pngquant made no file".to_string()))
        .context("encoding palette-64");
    let (status, influence) = encode_failure(&declined);
    assert!(
        matches!(&status, CandidateStatus::Skipped(reason) if reason.contains("pngquant made no file")),
        "{status:?}"
    );
    assert_eq!(influence, CandidateInfluence::Skipped);

    let crashed = anyhow!("cwebp exited with 1");
    let (status, influence) = encode_failure(&crashed);
    assert!(matches!(status, CandidateStatus::Failed(_)), "{status:?}");
    assert_eq!(influence, CandidateInfluence::Failed);
}

#[test]
fn scratch_files_are_deleted_on_success_and_on_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metric-candidate-0.png");
    let attempt = |fail: bool| -> Result<()> {
        fs::write(&path, b"png")?;
        let _scratch = RemoveOnDrop(&path);
        anyhow::ensure!(!fail, "scoring failed");
        Ok(())
    };
    attempt(false).unwrap();
    assert!(!path.exists());
    attempt(true).unwrap_err();
    assert!(!path.exists());
}

#[test]
fn only_images_with_a_side_of_8_px_are_scored_by_ssimulacra2() {
    let optimizer = Optimizer::new(CliConfig {
        metrics: crate::config::MetricsConfig {
            primary: PrimaryMetric::Ssimulacra2,
            ..Default::default()
        },
        ..base_config()
    });
    assert!(optimizer.scores_ssimulacra2(8, 8));
    assert!(!optimizer.scores_ssimulacra2(7, 100));
    assert!(!optimizer.scores_ssimulacra2(100, 7));
    // The floor of the built-in score is lower than the perceptual one.
    assert_eq!(quality_floor(&optimizer.config, true), 80.0);
    assert_eq!(quality_floor(&optimizer.config, false), 73.0);
    let builtin = Optimizer::new(CliConfig {
        metrics: crate::config::MetricsConfig {
            primary: PrimaryMetric::Builtin,
            ..Default::default()
        },
        ..base_config()
    });
    assert!(!builtin.scores_ssimulacra2(1000, 1000));
}

#[test]
fn a_tiny_image_reports_the_metric_and_floor_it_was_scored_with() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("tiny.png");
    DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 4, Rgb([9, 99, 199])))
        .save(&input)
        .unwrap();
    let report = Optimizer::new(CliConfig {
        inputs: vec![input],
        out_dir: Some(dir.path().join("out")),
        formats: CodecSelection::new([Codec::Png]),
        metrics: crate::config::MetricsConfig {
            primary: PrimaryMetric::Ssimulacra2,
            ..Default::default()
        },
        ..base_config()
    })
    .run()
    .unwrap()
    .report;
    let image = &report.reports[0];
    assert_eq!(image.quality_metric, "builtin");
    assert_eq!(image.min_quality, Some(73.0));
    assert!(image.warnings.iter().any(|w| w.contains("under 8 px")));
}

#[test]
fn a_lowered_input_limit_does_not_apply_to_the_pipelines_own_files() {
    let dir = tempfile::tempdir().unwrap();
    let candidate = dir.path().join("candidate.png");
    let original = DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, y| {
        Rgb([(x * 4) as u8, (y * 4) as u8, 7])
    }));
    original.save(&candidate).unwrap();
    let optimizer = Optimizer::new(CliConfig {
        max_input_bytes: 10,
        ..base_config()
    });

    assert_eq!(optimizer.pipeline_decode_limits().max_input_bytes, u64::MAX);
    assert_eq!(optimizer.image_decode_limits().max_input_bytes, 10);
    let decoded = optimizer
        .decode_candidate(Codec::Png, &candidate, &original, false, 1)
        .unwrap();
    assert_eq!(decoded.image.dimensions(), (64, 64));
    // A user's input of that size is still refused.
    assert!(optimizer.decode_input(&candidate).is_err());
}

#[test]
fn the_icc_profile_is_judged_against_the_pixels_that_are_written() {
    // The smallest thing that says it is a grey ICC profile.
    let mut profile = vec![0_u8; 128];
    profile[16..20].copy_from_slice(b"GRAY");
    profile[36..40].copy_from_slice(b"acsp");
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("grey.png");
    {
        let mut encoder = image::codecs::png::PngEncoder::new(fs::File::create(&input).unwrap());
        image::ImageEncoder::set_icc_profile(&mut encoder, profile).unwrap();
        DynamicImage::ImageLumaA8(image::GrayAlphaImage::from_pixel(
            16,
            16,
            image::LumaA([90, 128]),
        ))
        .write_with_encoder(encoder)
        .unwrap();
    }
    let warnings = |alpha: AlphaPolicy, label: &str| {
        let report = Optimizer::new(CliConfig {
            inputs: vec![input.clone()],
            out_dir: Some(dir.path().join(label)),
            formats: CodecSelection::new([Codec::Png]),
            alpha,
            metadata: MetadataPolicy::Keep,
            ..base_config()
        })
        .run()
        .unwrap()
        .report;
        assert!(
            report.reports[0].errors.is_empty(),
            "{:?}",
            report.reports[0].errors
        );
        report.reports[0].warnings.clone()
    };

    // Grey stays grey under the profile ...
    assert!(
        warnings(AlphaPolicy::Preserve, "preserve")
            .iter()
            .all(|w| !w.contains("ICC"))
    );
    // ... but flattening makes RGB, and a grey profile no longer fits it.
    assert!(
        warnings(
            AlphaPolicy::Flatten {
                background: "#ffffff".to_string()
            },
            "flatten"
        )
        .iter()
        .any(|w| w.starts_with("embedded ICC profile dropped"))
    );
}

#[test]
fn inversions_are_found_in_score_and_in_size_but_not_in_monotonic_results() {
    let accepted = || CandidateStatus::Accepted;
    let monotonic = [
        lossy(Codec::Jpeg, 60, 300, failed()),
        lossy(Codec::Jpeg, 78, 500, accepted()),
        lossy(Codec::Jpeg, 92, 900, accepted()),
    ];
    assert!(inversions(&monotonic, Codec::Jpeg).is_empty());
    assert!(neighbour_probes(&monotonic, Codec::Jpeg, None, &smallest_policy()).is_none());
    assert!(held_back_to_run(&monotonic, &[], true).0.is_empty());

    // q60 passes although q78 fails.
    let score = [
        lossy(Codec::Jpeg, 60, 300, accepted()),
        lossy(Codec::Jpeg, 78, 500, failed()),
    ];
    let found = inversions(&score, Codec::Jpeg);
    assert_eq!(
        found,
        vec![Inversion {
            kind: InversionKind::Score,
            lower: 60,
            higher: 78,
            lower_size: Some(300)
        }]
    );
    let (qualities, warning) =
        neighbour_probes(&score, Codec::Jpeg, None, &smallest_policy()).unwrap();
    assert_eq!(qualities, vec![59, 61]);
    assert!(
        warning.contains("non-monotonic behaviour detected for jpeg: q60 passes while q78 fails")
    );
    assert!(warning.contains("q59, q61"));

    // q60 encodes larger than q78.
    let size = [
        lossy(Codec::Jpeg, 60, 700, accepted()),
        lossy(Codec::Jpeg, 78, 500, accepted()),
    ];
    let found = inversions(&size, Codec::Jpeg);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].kind, InversionKind::Size);
    assert_eq!(
        neighbour_probes(&size, Codec::Jpeg, None, &smallest_policy())
            .unwrap()
            .0,
        vec![59, 61]
    );
    // A neighbour that was already tried is not tried again.
    let mut tried = size.to_vec();
    tried.push(lossy(Codec::Jpeg, 61, 480, accepted()));
    assert_eq!(
        neighbour_probes(&tried, Codec::Jpeg, None, &smallest_policy())
            .unwrap()
            .0,
        vec![59]
    );
    // A format that cannot beat the best is not probed.
    let best = best_of(Codec::Avif, 200);
    assert!(neighbour_probes(&size, Codec::Jpeg, Some(&best), &smallest_policy()).is_none());
    // Other formats and other efforts are not compared.
    assert!(inversions(&size, Codec::Webp).is_empty());
    let mut other_effort = size.to_vec();
    other_effort[1].settings = EncoderSettings::Quality {
        quality: 78,
        effort: 9,
    };
    assert!(inversions(&other_effort, Codec::Jpeg).is_empty());
}

#[test]
fn held_back_qualities_run_after_an_inversion_or_for_an_unscored_upper_end() {
    let held = |quality| {
        CandidateReport::unevaluated(
            Codec::Jpeg,
            EncoderSettings::Quality { quality, effort: 3 },
            "assumed".to_string(),
        )
    };
    let held = [held(92), held(60), held(48)];
    let scored = |quality, overall| {
        let mut candidate = lossy(Codec::Jpeg, quality, 500, failed());
        candidate.quality = Some(QualityScore {
            overall,
            ..QualityScore::lossless()
        });
        candidate
    };

    // q78 failed and q92 is too large to be scored: the crossing needs q60 too.
    let oversize = lossy(
        Codec::Jpeg,
        92,
        20_000,
        CandidateStatus::Rejected("not smaller than original".to_string()),
    );
    let candidates = [scored(78, 70.0), oversize];
    assert_eq!(
        held_back_to_run(&candidates, &held, true),
        (vec![1], vec![])
    );
    assert!(held_back_to_run(&candidates, &held, false).0.is_empty());

    // q60 passing under a failing q78 is an inversion: everything held back runs.
    let inverted = [
        lossy(Codec::Jpeg, 60, 300, CandidateStatus::Accepted),
        scored(78, 70.0),
    ];
    let (picked, warnings) = held_back_to_run(&inverted, &held, false);
    assert_eq!(picked, vec![0, 1, 2]);
    assert!(warnings[0].contains("non-monotonic behaviour detected for jpeg"));
    assert!(warnings[0].contains("q92, q60, q48"));
}

#[test]
fn staged_search_skips_what_the_middle_quality_rules_out() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, y| {
        Rgb([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8])
    }))
    .save_with_format(&input, image::ImageFormat::Png)
    .unwrap();
    let reasons = |min_quality: f32| {
        let summary = Optimizer::new(CliConfig {
            inputs: vec![input.clone()],
            out_dir: Some(dir.path().join(format!("out-{min_quality}"))),
            formats: CodecSelection::new([Codec::Jpeg]),
            allow_larger: true,
            min_quality: Some(min_quality),
            workers: 2,
            gpu: crate::config::GpuConfig {
                enabled: false,
                ..crate::config::GpuConfig::default()
            },
            ..CliConfig::default()
        })
        .run()
        .unwrap();
        let report = &summary.report.reports[0];
        assert!(report.warnings.iter().all(|w| !w.contains("non-monotonic")));
        report
            .candidates
            .iter()
            .take(3)
            .map(|candidate| match &candidate.status {
                CandidateStatus::Skipped(reason) => reason.clone(),
                _ => "evaluated".to_string(),
            })
            .collect::<Vec<_>>()
    };

    // Everything passes: q78 first, then q60; q92 is assumed to pass.
    let passing = reasons(0.0);
    assert_eq!(passing[0], "evaluated");
    assert_eq!(passing[1], "evaluated");
    assert!(passing[2].starts_with("assumed to pass"), "{passing:?}");
    // Nothing passes: q78 first, then q92; q60 is assumed to fail.
    let failing = reasons(1000.0);
    assert_eq!(failing[0], "evaluated");
    assert_eq!(failing[1], "evaluated");
    assert!(failing[2].starts_with("assumed to fail"), "{failing:?}");
}

#[test]
fn candidate_cache_reuses_encoded_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    let cache_dir = dir.path().join("cache");
    let first_out = dir.path().join("first");
    let second_out = dir.path().join("second");
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([50, 80, 110])));
    image
        .save_with_format(&input, image::ImageFormat::Png)
        .unwrap();

    let base_config = CliConfig {
        inputs: vec![input],
        preset: Preset::Balanced,
        target: TargetPolicy::MaxCompat,
        formats: CodecSelection::new([Codec::Png]),
        allow_larger: true,
        max_pixels: 100_000_000,
        cache: CacheConfig {
            enabled: true,
            dir: Some(cache_dir),
            ..CacheConfig::default()
        },
        ..CliConfig::default()
    };

    let first = Optimizer::new(CliConfig {
        out_dir: Some(first_out),
        ..base_config.clone()
    })
    .run()
    .unwrap();
    assert_eq!(first.report.failures, 0);

    let second = Optimizer::new(CliConfig {
        out_dir: Some(second_out),
        ..base_config
    })
    .run()
    .unwrap();
    assert_eq!(second.report.failures, 0);
    let candidate = &second.report.reports[0].candidates[0];
    assert_eq!(candidate.backend.as_deref(), Some("cache"));
    assert_eq!(candidate.encode_ms, Some(0));
    assert!(
        candidate
            .warnings
            .iter()
            .any(|warning| warning == "cache hit")
    );
    assert!(
        candidate
            .warnings
            .iter()
            .any(|warning| warning.starts_with("quality cache hit"))
    );
    assert!(candidate.quality.is_some());
    assert!(candidate.score_backend.is_some());
    assert!(candidate.score_ms.is_some());
    assert!(candidate.total_ms.is_some());
}

#[test]
fn a_candidate_of_the_wrong_size_is_rejected_before_any_decoder_runs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("candidate.jxl");
    // The start of a bare JPEG XL codestream for a 37x21 image.
    fs::write(
        &path,
        [
            0xff, 0x0a, 0xa0, 0xd0, 0x01, 0x00, 0x14, 0x08, 0x02, 0x00, 0xa4, 0x02,
        ],
    )
    .unwrap();
    let original = DynamicImage::ImageRgb8(RgbImage::new(64, 48));

    let error = Optimizer::new(CliConfig::default())
        .decode_candidate(Codec::Jxl, &path, &original, false, 1)
        .err()
        .expect("the size mismatch must fail the candidate");

    // The message comes from the header check, not from djxl, and the
    // decoder's output file was never created.
    assert!(error.to_string().contains("37x21 but 64x48"), "{error}");
    assert!(!path.with_extension("decoded.pam").exists());
}

#[test]
fn cpu_and_gpu_backends_reach_the_same_decisions() {
    if gpu_or_skip(GpuWgpuBackend::new()).is_none() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    // Smooth colour with fine texture, so lossy candidates land on both
    // sides of the floor.
    DynamicImage::ImageRgb8(RgbImage::from_fn(640, 400, |x, y| {
        let texture = ((x * 7 + y * 13) % 23) as u8;
        Rgb([
            (x / 3) as u8 + texture,
            (y / 2) as u8 + texture / 2,
            ((x + y) / 5) as u8,
        ])
    }))
    .save_with_format(&input, image::ImageFormat::Png)
    .unwrap();
    let run = |gpu_enabled: bool, name: &str| {
        let mut config = CliConfig {
            inputs: vec![input.clone()],
            out_dir: Some(dir.path().join(name)),
            formats: CodecSelection::new([Codec::Jpeg, Codec::Png]),
            metadata: MetadataPolicy::Keep,
            allow_larger: true,
            // The built-in score is the one the backends compute.
            metrics: crate::config::MetricsConfig {
                primary: PrimaryMetric::Builtin,
                ..Default::default()
            },
            ..CliConfig::default()
        };
        config.gpu.enabled = gpu_enabled;
        config.gpu.min_pixels_for_gpu_metrics = 0;
        let mut summary = Optimizer::new(config).run().unwrap();
        summary.report.reports.remove(0)
    };
    let (cpu, gpu) = (run(false, "cpu"), run(true, "gpu"));
    assert!(
        gpu.candidates
            .iter()
            .any(|candidate| candidate.score_backend == Some(ActualScoreBackend::Gpu))
    );
    assert_eq!(cpu.winner_index, gpu.winner_index);
    assert_eq!(cpu.candidates.len(), gpu.candidates.len());
    let mut verdicts = HashSet::new();
    for (cpu, gpu) in cpu.candidates.iter().zip(&gpu.candidates) {
        assert_eq!((cpu.codec, &cpu.settings), (gpu.codec, &gpu.settings));
        assert_eq!(cpu.size_bytes, gpu.size_bytes);
        // Same verdict, and scores too close to move a candidate across the floor.
        assert_eq!(
            std::mem::discriminant(&cpu.status),
            std::mem::discriminant(&gpu.status),
            "{:?} vs {:?}",
            cpu.status,
            gpu.status
        );
        if let (Some(cpu), Some(gpu)) = (cpu.quality, gpu.quality) {
            assert!((cpu.overall - gpu.overall).abs() <= 0.01, "{cpu:?} {gpu:?}");
        }
        verdicts.insert(std::mem::discriminant(&cpu.status));
    }
    assert!(
        verdicts.len() > 1,
        "the fixture should produce mixed verdicts"
    );
}

#[test]
fn large_images_are_scored_on_full_resolution_samples() {
    let large = DynamicImage::ImageRgb8(RgbImage::from_fn(5000, 4000, |x, y| {
        Rgb([(x % 256) as u8, (y % 256) as u8, 7])
    }));
    let metric = bounded_metric_image(&large);
    assert_eq!(metric.dimensions(), (3072, 3072));
    assert!(!metric.color().has_alpha());
    // The first tile is the centre of the top-left grid cell, unscaled.
    let (x, y) = ((5000 / 3 - 1024) / 2, (4000 / 3 - 1024) / 2);
    assert_eq!(
        metric.to_rgb8().get_pixel(0, 0),
        &Rgb([(x % 256) as u8, (y % 256) as u8, 7])
    );

    let small = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
    assert_eq!(bounded_metric_image(&small).dimensions(), (64, 64));
}

fn fixture(name: &str) -> DynamicImage {
    image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name),
    )
    .unwrap()
}

#[test]
fn in_process_ssimulacra2_scores_identical_images_100_and_degraded_ones_lower() {
    let original = DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, y| {
        Rgb([(x * 4) as u8, (y * 4) as u8, ((x * y) % 256) as u8])
    }));
    let reference = inprocess::build_reference(&original).unwrap();
    let same = inprocess::score(&reference, &original).unwrap();
    let mild = inprocess::score(&reference, &original.blur(1.0)).unwrap();
    let heavy = inprocess::score(&reference, &original.blur(4.0)).unwrap();
    assert!(same > 99.99, "{same}");
    assert!(same > mild && mild > heavy, "{same} {mild} {heavy}");
}

#[test]
fn in_process_ssimulacra2_agrees_with_the_external_tool() {
    /// The two differ by under 0.1 on these cases; the bound leaves room for
    /// the SIMD rounding of other machines.
    const TOLERANCE: f32 = 0.25;
    let Some(tools) = smart_compress_codecs::tools_or_skip(&["ssimulacra2"]) else {
        return;
    };
    let tool = tools.ssimulacra2.path.unwrap();
    let dir = tempfile::tempdir().unwrap();
    // Opaque, transparent, flat and busy images.
    for name in [
        "photo-like.jpg",
        "screenshot-like.png",
        "transparent.png",
        "noisy.png",
        "gradient.png",
    ] {
        let original = fixture(name);
        let original_png = dir.path().join("original.png");
        write_metric_png(&original, &original_png).unwrap();
        let reference = inprocess::build_reference(&original).unwrap();
        for candidate in [
            original.blur(1.0),
            original.blur(3.0),
            original.brighten(12),
        ] {
            let candidate_png = dir.path().join("candidate.png");
            write_metric_png(&candidate, &candidate_png).unwrap();
            let output = smart_compress_codecs::run_process(
                &tool,
                &[original_png.clone().into(), candidate_png.into()],
                std::time::Duration::from_secs(60),
            )
            .unwrap();
            let external: f32 = String::from_utf8_lossy(&output.stdout)
                .split_whitespace()
                .next()
                .unwrap()
                .parse()
                .unwrap();
            let in_process = inprocess::score(&reference, &candidate).unwrap();
            assert!(
                (external - in_process).abs() <= TOLERANCE,
                "{name}: external {external}, in-process {in_process}"
            );
        }
    }
}

#[test]
fn in_process_ssimulacra2_scores_a_run_without_the_external_tool() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    // Half transparent, so the alpha checks run beside the perceptual score.
    DynamicImage::ImageRgba8(image::RgbaImage::from_fn(48, 48, |x, y| {
        image::Rgba([
            (x * 5) as u8,
            (y * 5) as u8,
            90,
            if x < 24 { 255 } else { 120 },
        ])
    }))
    .save(&input)
    .unwrap();
    let mut optimizer = Optimizer::new(CliConfig {
        inputs: vec![input],
        out_dir: Some(dir.path().join("out")),
        formats: CodecSelection::new([Codec::Png]),
        metrics: crate::config::MetricsConfig {
            primary: PrimaryMetric::Ssimulacra2Inprocess,
            ..Default::default()
        },
        ..base_config()
    });
    let mut tools = ToolInventory::detect();
    tools.ssimulacra2.enabled = false;
    tools.ssimulacra2.path = None;
    optimizer.backend = Arc::new(smart_compress_codecs::EncoderBackend::new(tools));
    let report = optimizer.run().unwrap().report;
    let image = &report.reports[0];
    assert!(image.errors.is_empty(), "{:?}", image.errors);
    assert_eq!(image.quality_metric, "ssimulacra2");
    let scored = image
        .candidates
        .iter()
        .filter(|candidate| candidate.quality.is_some())
        .collect::<Vec<_>>();
    assert!(!scored.is_empty());
    for candidate in scored {
        assert_eq!(
            candidate.score_backend,
            Some(ActualScoreBackend::Ssimulacra2Inprocess)
        );
        assert!(candidate.quality.unwrap().ssimulacra2.is_some());
    }
}

#[test]
fn the_scoring_signature_names_the_ssimulacra2_implementation() {
    let signature = |primary| {
        Optimizer::new(CliConfig {
            metrics: crate::config::MetricsConfig {
                primary,
                ..Default::default()
            },
            ..base_config()
        })
        .scoring_signature()
    };
    let external = signature(PrimaryMetric::Ssimulacra2);
    let in_process = signature(PrimaryMetric::Ssimulacra2Inprocess);
    assert!(external.contains(";ssimulacra2=true;"), "{external}");
    assert!(in_process.contains(inprocess::SIGNATURE), "{in_process}");
    assert_ne!(external, in_process);
}

#[test]
fn one_at_a_time_search_stops_a_format_once_it_is_settled() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, y| {
        Rgb([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8])
    }))
    .save_with_format(&input, image::ImageFormat::Png)
    .unwrap();
    let summary = Optimizer::new(CliConfig {
        inputs: vec![input],
        out_dir: Some(dir.path().join("out")),
        formats: CodecSelection::new([Codec::Jpeg]),
        allow_larger: true,
        min_quality: Some(0.0),
        // A temporary-disk budget selects the one-at-a-time path.
        max_temp_bytes: Some(u64::MAX),
        gpu: crate::config::GpuConfig {
            enabled: false,
            ..crate::config::GpuConfig::default()
        },
        ..CliConfig::default()
    })
    .run()
    .unwrap();

    let report = &summary.report.reports[0];
    assert_eq!(summary.report.failures, 0);
    let settled = report
        .candidates
        .iter()
        .filter(|candidate| {
            matches!(&candidate.status, CandidateStatus::Skipped(reason) if reason.contains("settled"))
        })
        .count();
    // The lowest grid quality passes a floor of zero; the rest are never encoded.
    assert_eq!(settled, crate::SpeedPreset::Balanced.qualities().len() - 1);
    assert!(report.candidates.iter().all(|candidate| {
        candidate.size_bytes.is_some() || matches!(candidate.status, CandidateStatus::Skipped(_))
    }));
}

#[test]
fn flatten_composites_alpha_over_background() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    let mut rgba = image::RgbaImage::from_pixel(8, 8, image::Rgba([0, 0, 255, 255]));
    rgba.put_pixel(0, 0, image::Rgba([0, 0, 255, 0]));
    rgba.save(&input).unwrap();
    let summary = Optimizer::new(CliConfig {
        inputs: vec![input],
        out_dir: Some(dir.path().join("out")),
        formats: CodecSelection::new([Codec::Png]),
        allow_larger: true,
        alpha: AlphaPolicy::Flatten {
            background: "#ff0000".to_string(),
        },
        gpu: crate::config::GpuConfig {
            enabled: false,
            ..crate::config::GpuConfig::default()
        },
        ..CliConfig::default()
    })
    .run()
    .unwrap();

    assert_eq!(summary.report.failures, 0);
    let output = image::open(dir.path().join("out").join("input.png")).unwrap();
    assert!(!output.color().has_alpha());
    assert_eq!(output.to_rgb8().get_pixel(0, 0), &Rgb([255, 0, 0]));
    assert_eq!(output.to_rgb8().get_pixel(1, 1), &Rgb([0, 0, 255]));
}

#[test]
fn same_named_inputs_publish_distinct_outputs_and_protect_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let mut inputs = Vec::new();
    for (folder, shade) in [("a", 10), ("b", 200)] {
        let folder = dir.path().join(folder);
        fs::create_dir(&folder).unwrap();
        let input = folder.join("x.png");
        DynamicImage::ImageRgb8(RgbImage::from_pixel(8, 8, Rgb([shade, shade, shade])))
            .save_with_format(&input, image::ImageFormat::Png)
            .unwrap();
        inputs.push(input);
    }
    let base = CliConfig {
        formats: CodecSelection::new([Codec::Png]),
        allow_larger: true,
        gpu: crate::config::GpuConfig {
            enabled: false,
            ..crate::config::GpuConfig::default()
        },
        ..CliConfig::default()
    };
    let out_dir = dir.path().join("out");
    let summary = Optimizer::new(CliConfig {
        inputs: inputs.clone(),
        out_dir: Some(out_dir.clone()),
        ..base.clone()
    })
    .run()
    .unwrap();
    assert_eq!(summary.report.failures, 0);
    assert_eq!(fs::read_dir(&out_dir).unwrap().count(), 2);

    // Writing next to the input with overwrite enabled must not replace it.
    let original_bytes = fs::read(&inputs[0]).unwrap();
    let summary = Optimizer::new(CliConfig {
        inputs: vec![inputs[0].clone()],
        out_dir: Some(dir.path().join("a")),
        overwrite: true,
        ..base
    })
    .run()
    .unwrap();
    assert_eq!(summary.report.failures, 1);
    assert_eq!(fs::read(&inputs[0]).unwrap(), original_bytes);
}

fn base_config() -> CliConfig {
    CliConfig {
        allow_larger: true,
        gpu: crate::config::GpuConfig {
            enabled: false,
            ..crate::config::GpuConfig::default()
        },
        ..CliConfig::default()
    }
}

fn gradient(path: &Path) {
    DynamicImage::ImageRgb8(RgbImage::from_fn(32, 32, |x, y| {
        Rgb([x as u8 * 8, y as u8 * 8, 90])
    }))
    .save(path)
    .unwrap();
}

#[test]
fn a_file_that_was_already_in_the_input_folder_is_not_replaced() {
    let dir = tempfile::tempdir().unwrap();
    gradient(&dir.path().join("a.png"));
    let unrelated = dir.path().join("a.jpg");
    fs::write(&unrelated, b"someone else's picture").unwrap();

    let summary = Optimizer::new(CliConfig {
        inputs: vec![dir.path().join("a.png")],
        out_dir: Some(dir.path().to_path_buf()),
        formats: CodecSelection::new([Codec::Jpeg]),
        min_quality: Some(0.0),
        overwrite: true,
        ..base_config()
    })
    .run()
    .unwrap();

    let report = &summary.report.reports[0];
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.contains("already there")),
        "{:?}",
        report.errors
    );
    assert_eq!(fs::read(&unrelated).unwrap(), b"someone else's picture");

    // Without protect-inputs the same run replaces it, as asked.
    let summary = Optimizer::new(CliConfig {
        inputs: vec![dir.path().join("a.png")],
        out_dir: Some(dir.path().to_path_buf()),
        formats: CodecSelection::new([Codec::Jpeg]),
        min_quality: Some(0.0),
        overwrite: true,
        protect_inputs: false,
        ..base_config()
    })
    .run()
    .unwrap();
    assert_eq!(summary.report.failures, 0);
    assert_ne!(fs::read(&unrelated).unwrap(), b"someone else's picture");
}

#[test]
fn a_panic_in_one_image_is_an_error_entry_and_the_batch_goes_on() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = ["panic.png", "fine.png"].map(|name| dir.path().join(name));
    inputs.iter().for_each(|input| gradient(input));
    let optimizer = Optimizer::new(CliConfig {
        inputs: inputs.to_vec(),
        out_dir: Some(dir.path().join("out")),
        formats: CodecSelection::new([Codec::Png]),
        report_out: Some(dir.path().join("report.json")),
        report: ReportFormat::Json,
        ..base_config()
    });

    let summary = optimizer.run().unwrap();

    let reports = &summary.report.reports;
    assert_eq!(reports.len(), 2);
    assert!(
        reports[0].errors[0].starts_with("internal error: injected panic"),
        "{:?}",
        reports[0].errors
    );
    assert!(reports[1].errors.is_empty() && reports[1].winner_index.is_some());
    assert_eq!(summary.report.failures, 1);
    // What the panicking image held was given back.
    assert_eq!(optimizer.file_memory.used(), 0);
    assert_eq!(optimizer.active_files.load(Ordering::SeqCst), 0);
    assert!(dir.path().join("report.json").exists());
}

#[test]
fn checkpointed_runs_keep_input_order_and_resume_by_path() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = ["c.png", "a.png", "b.png"].map(|name| dir.path().join(name));
    inputs.iter().for_each(|input| gradient(input));
    let first_report = dir.path().join("first.json");
    let config = CliConfig {
        inputs: inputs.to_vec(),
        dry_run: true,
        report: ReportFormat::Json,
        report_out: Some(first_report.clone()),
        ..base_config()
    };
    let first = Optimizer::new(config.clone()).run().unwrap();
    let names = |summary: &BatchSummary| {
        summary
            .report
            .reports
            .iter()
            .map(|report| report.input.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        names(&first),
        inputs.map(|input| input.display().to_string())
    );

    let second = Optimizer::new(CliConfig {
        report_out: Some(dir.path().join("second.json")),
        resume_report: Some(first_report),
        ..config
    })
    .run()
    .unwrap();
    let manifest = second.report.manifest.clone().unwrap();
    assert_eq!(
        manifest.resume_decision.as_deref(),
        Some("resumed 3 compatible completed input(s)")
    );
    assert_eq!(manifest.completed_inputs, 3);
    assert_eq!(names(&second), names(&first));
}

fn avif_with_size(width: u32, height: u32) -> Vec<u8> {
    let boxed = |kind: &[u8; 4], body: &[u8]| {
        let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    };
    let ispe = [
        vec![0; 4],
        width.to_be_bytes().into(),
        height.to_be_bytes().into(),
    ]
    .concat();
    let ipma = [
        &[0, 0, 0, 0][..],
        &1_u32.to_be_bytes(),
        &1_u16.to_be_bytes(),
        &[1, 0x81],
    ]
    .concat();
    let iprp = [
        boxed(b"ipco", &boxed(b"ispe", &ispe)),
        boxed(b"ipma", &ipma),
    ]
    .concat();
    let meta = [
        vec![0; 4],
        boxed(b"pitm", &[0, 0, 0, 0, 0, 1]),
        boxed(b"iprp", &iprp),
    ]
    .concat();
    [
        boxed(b"ftyp", b"avif\0\0\0\0avifmif1"),
        boxed(b"meta", &meta),
    ]
    .concat()
}

#[test]
fn an_avif_header_beyond_the_limits_never_reaches_the_decoder_tool() {
    let dir = tempfile::tempdir().unwrap();
    let optimizer = Optimizer::new(base_config());
    for (width, height, expected) in [
        (
            50_000,
            50_000,
            "image has 2500000000 pixels, exceeding the configured 100000000 pixel limit",
        ),
        (
            100_001,
            1,
            "image width 100001 exceeds the configured 100000 pixel width limit",
        ),
    ] {
        let input = dir.path().join("big.avif");
        fs::write(&input, avif_with_size(width, height)).unwrap();
        let error = optimizer.decode_input(&input).err().unwrap();
        assert_eq!(error.to_string(), expected);
    }
    let limited = Optimizer::new(CliConfig {
        max_decoded_bytes: 1000,
        ..base_config()
    });
    let input = dir.path().join("big.avif");
    fs::write(&input, avif_with_size(100, 100)).unwrap();
    let error = limited.decode_input(&input).err().unwrap();
    assert!(
        error
            .to_string()
            .starts_with("decoded image requires 40000 bytes")
    );
}

#[test]
fn gpu_required_reports_late_metric_failure() {
    if gpu_or_skip(GpuWgpuBackend::new()).is_none() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([20, 30, 40])))
        .save_with_format(&input, image::ImageFormat::Png)
        .unwrap();
    let gpu = crate::config::GpuConfig {
        required: true,
        max_vram_bytes: Some(1),
        ..crate::config::GpuConfig::default()
    };
    let summary = Optimizer::new(CliConfig {
        inputs: vec![input],
        out_dir: Some(dir.path().join("out")),
        formats: CodecSelection::new([Codec::Png]),
        allow_larger: true,
        gpu,
        ..CliConfig::default()
    })
    .run()
    .unwrap();

    assert_eq!(summary.report.failures, 1);
    let image = &summary.report.reports[0];
    assert!(
        image
            .candidates
            .iter()
            .any(|candidate| format!("{:?}", candidate.status).contains("GPU is required")),
        "{:?} {:?}",
        image.errors,
        image
            .candidates
            .iter()
            .map(|c| &c.status)
            .collect::<Vec<_>>()
    );
}

#[test]
fn resume_fingerprint_ignores_report_paths() {
    let base = CliConfig::default();
    let mut with_paths = base.clone();
    with_paths.report_out = Some(PathBuf::from("one.json"));
    with_paths.resume_report = Some(PathBuf::from("old.json"));

    let first = Optimizer::with_progress(base, Arc::new(NoopProgress));
    let second = Optimizer::with_progress(with_paths, Arc::new(NoopProgress));

    assert_eq!(first.config_fingerprint, second.config_fingerprint);
}

/// A candidate whose status is a verdict on quality (accepted, or rejected
/// with a "quality ..." reason) carries the score it was judged by.
fn quality_candidate(codec: Codec, quality: u8, status: CandidateStatus) -> CandidateReport {
    let scored = match &status {
        CandidateStatus::Accepted => true,
        CandidateStatus::Rejected(reason) => reason.starts_with("quality "),
        _ => false,
    };
    CandidateReport {
        codec,
        settings: EncoderSettings::Quality { quality, effort: 3 },
        path: None,
        size_bytes: Some(100),
        encode_ms: Some(1),
        decode_ms: Some(1),
        decode_process_ms: None,
        decode_io_ms: None,
        preprocess_ms: None,
        score_ms: Some(1),
        cache_ms: None,
        external_metric_ms: None,
        gpu_upload_ms: None,
        gpu_transform_ms: None,
        gpu_compute_ms: None,
        gpu_readback_ms: None,
        gpu_timestamp_ns: None,
        total_ms: Some(3),
        quality: scored.then(|| QualityScore {
            overall: 75.0,
            ..QualityScore::lossless()
        }),
        backend: Some("test".to_string()),
        score_backend: Some(ActualScoreBackend::Cpu),
        quality_source: QualitySource::Fresh,
        cache_hit_kind: None,
        process_failure: None,
        gpu_verification: None,
        influence: None,
        bytes: Default::default(),
        status,
        warnings: Vec::new(),
    }
}

/// Images between the planning of their candidates and their end.
#[derive(Default)]
struct ImagesInFlight {
    now: AtomicUsize,
    peak: AtomicUsize,
}

impl ProgressObserver for ImagesInFlight {
    fn candidates_planned(&self, _path: &Path, _total: usize) {
        let now = self.now.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
    }
    fn file_finished(&self, _path: &Path) {
        // Saturating decrement; `fetch_update` is deprecated on newer toolchains
        // and its replacement is missing on older ones.
        let mut now = self.now.load(Ordering::SeqCst);
        while now > 0 {
            match self
                .now
                .compare_exchange(now, now - 1, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => break,
                Err(actual) => now = actual,
            }
        }
    }
}

fn cpu_only() -> crate::config::GpuConfig {
    crate::config::GpuConfig {
        enabled: false,
        ..crate::config::GpuConfig::default()
    }
}

#[test]
fn an_image_larger_than_the_memory_limit_runs_alone_and_completes() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = (0..3)
        .map(|index| noisy_source(dir.path(), &format!("source-{index}.png")))
        .collect::<Vec<_>>();
    let run = |label: &str, memory: u64| {
        let trace = Arc::new(ImagesInFlight::default());
        let summary = Optimizer::with_progress(
            CliConfig {
                inputs: inputs.clone(),
                out_dir: Some(dir.path().join(label)),
                formats: CodecSelection::new([Codec::Jpeg]),
                allow_larger: true,
                min_quality: Some(0.0),
                workers: 4,
                max_memory_bytes: Some(memory),
                gpu: cpu_only(),
                ..CliConfig::default()
            },
            trace.clone(),
        )
        .run()
        .unwrap();
        (summary.report.reports, trace.peak.load(Ordering::SeqCst))
    };

    // Every image is larger than one byte, so each is granted the whole
    // limit, and they take turns.
    let (limited, limited_peak) = run("limited", 1);
    let (open, _) = run("open", u64::MAX);

    assert_eq!(limited_peak, 1);
    // Each image says that it alone needs more than the limit; nothing else does.
    for report in &limited {
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.contains("more than the whole memory limit")),
            "{:?}",
            report.warnings
        );
    }
    assert!(open.iter().all(|report| report.warnings.is_empty()));
    let sizes = |reports: &[ImageReport]| {
        reports
            .iter()
            .map(|report| {
                assert!(report.errors.is_empty(), "{:?}", report.errors);
                report.candidates[report.winner_index.unwrap()].size_bytes
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(sizes(&limited), sizes(&open));
    let waits = |reports: &[ImageReport]| {
        reports
            .iter()
            .map(|report| report.timings.backpressure_wait_ms.unwrap())
            .sum::<u64>()
    };
    assert!(waits(&limited) > 0);
}

#[test]
fn candidates_wait_for_temporary_room_only_while_something_is_running() {
    let optimizer = Optimizer::new(CliConfig {
        max_run_temp_bytes: Some(1_000),
        gpu: cpu_only(),
        ..CliConfig::default()
    });
    let waited = AtomicU64::new(0);
    // The soft limit is three quarters of the run budget.
    optimizer.run_temp_usage.store(900, Ordering::SeqCst);

    // Nothing is running, so nothing will make room: the work goes ahead.
    assert_eq!(optimizer.wait_for_temp_room(None, &waited), Ok(()));

    optimizer.active_candidates.store(1, Ordering::SeqCst);
    let deadline = Instant::now() + std::time::Duration::from_millis(60);
    assert_eq!(
        optimizer.wait_for_temp_room(Some(deadline), &waited),
        Err(WaitEnded::TimedOut)
    );
    assert!(waited.load(Ordering::SeqCst) >= 50);

    std::thread::scope(|scope| {
        let waiting = scope.spawn(|| optimizer.wait_for_temp_room(None, &waited));
        std::thread::sleep(std::time::Duration::from_millis(100));
        optimizer.run_temp_usage.store(100, Ordering::SeqCst);
        optimizer.temp_gate.wake();
        assert_eq!(waiting.join().unwrap(), Ok(()));
    });
}

#[test]
fn the_source_size_comes_from_the_header_not_a_decode() {
    let dir = tempfile::tempdir().unwrap();
    let png = noisy_source(dir.path(), "source.png");
    let optimizer = Optimizer::new(CliConfig {
        gpu: cpu_only(),
        ..CliConfig::default()
    });
    assert_eq!(optimizer.source_pixels(&png, 1), 96 * 96);
    // A file that is no image is sized from its length, at most max_pixels.
    let junk = dir.path().join("junk.png");
    std::fs::write(&junk, [0_u8; 100]).unwrap();
    assert_eq!(optimizer.source_pixels(&junk, 100), 400);
    assert_eq!(
        optimizer.source_pixels(&junk, u64::MAX),
        optimizer.config.max_pixels
    );
}

/// A candidate that scored `score` at `quality`, against the floor of 80.
fn scored(codec: Codec, quality: u8, score: f32) -> CandidateReport {
    let status = if score >= 80.0 {
        CandidateStatus::Accepted
    } else {
        CandidateStatus::Rejected(format!("quality {score:.1} below 80.0"))
    };
    let mut candidate = lossy(codec, quality, 1_000 + 20 * u64::from(quality), status);
    candidate.quality = Some(QualityScore {
        overall: score,
        ..QualityScore::lossless()
    });
    candidate
}

/// Runs the predicted search of one format against a curve, as the search of
/// an image would: the start, then every round it asks for. Returns the
/// rounds (the start counts as one), the candidates evaluated and the lowest
/// passing quality.
fn converge(codec: Codec, start: u8, score: impl Fn(u8) -> f32) -> (usize, usize, Option<u8>) {
    let policy = smallest_policy();
    let mut candidates = vec![scored(codec, start, score(start))];
    let mut track = Track::new(codec);
    let mut rounds = 1;
    loop {
        match track.step(&candidates, 80.0, 92, None, &policy) {
            Step::Probe { qualities, .. } => {
                rounds += 1;
                assert!(rounds < 12, "the search does not close");
                candidates.extend(qualities.into_iter().map(|q| scored(codec, q, score(q))));
            }
            Step::Done => break,
            Step::Fallback(reason) => panic!("handed back: {reason}"),
        }
    }
    let lowest = readings(&candidates, codec)
        .into_iter()
        .find(|reading| reading.outcome == Outcome::Passed)
        .map(|reading| reading.quality);
    (rounds, candidates.len(), lowest)
}

#[test]
fn the_predicted_start_rises_with_the_target_and_stays_within_the_grid() {
    for codec in [Codec::Jpeg, Codec::Webp, Codec::Avif, Codec::Jxl] {
        let starts = [70.0, 80.0, 90.0].map(|target| predicted_quality(codec, target, 92));
        assert!(
            starts[0] < starts[1] && starts[1] < starts[2],
            "{codec:?}: {starts:?}"
        );
        assert_eq!(predicted_quality(codec, 500.0, 92), 92);
        assert_eq!(predicted_quality(codec, -500.0, 92), 1);
    }
}

#[test]
fn a_monotone_curve_closes_in_few_rounds_whatever_the_start() {
    // The score reaches the floor of 80 at q50 and not at q49.
    let curve = |quality: u8| 80.0 + 0.7 * (f32::from(quality) - 50.0);
    // A start on the crossing needs one round, of three qualities, to show q49 failing.
    assert_eq!(converge(Codec::Webp, 50, curve), (2, 4, Some(50)));
    // A start a few steps off needs the window and at most a second look.
    for start in [44, 47, 53, 58] {
        let (rounds, _, lowest) = converge(Codec::Webp, start, curve);
        assert_eq!(lowest, Some(50), "start q{start}");
        assert!(rounds <= 5, "start q{start}: {rounds} rounds");
    }
    // Far off: the range widens or the window moves, and the answer is the same.
    for start in [30, 70] {
        assert_eq!(
            converge(Codec::Webp, start, curve).2,
            Some(50),
            "start q{start}"
        );
    }
}

#[test]
fn a_floor_above_the_grid_is_settled_by_probing_its_top() {
    // The score never gets near 80, whatever the quality.
    let (rounds, _, lowest) = converge(Codec::Avif, 60, |quality| 60.0 + 0.2 * f32::from(quality));
    assert_eq!(lowest, None);
    assert!(rounds <= 4, "{rounds} rounds");
}

#[test]
fn the_error_range_is_sized_by_the_start_error_and_widens_when_the_crossing_is_outside() {
    let policy = smallest_policy();
    let mut track = Track::new(Codec::Jpeg);
    // q60 scored 74: six points short of 80.
    let mut candidates = vec![scored(Codec::Jpeg, 60, 74.0)];
    let Step::Probe { qualities, .. } = track.step(&candidates, 80.0, 92, None, &policy) else {
        panic!("a failing start asks for higher qualities");
    };
    assert!(qualities.iter().all(|quality| *quality > 60));
    let range = track.range().unwrap();
    assert_eq!(range.half_width, 18);
    assert!(range.centre > 60.0);

    // The next point rose so slowly that the crossing is far beyond the range.
    candidates.push(scored(Codec::Jpeg, 70, 74.51));
    let Step::Probe { qualities, .. } = track.step(&candidates, 80.0, 92, None, &policy) else {
        panic!("a first widening still probes");
    };
    assert_eq!(track.range().unwrap().half_width, 36);
    assert_eq!(qualities, vec![90, 91, 92]);
    assert!(matches!(
        track.step(&candidates, 80.0, 92, None, &policy),
        Step::Probe { .. }
    ));
    assert_eq!(track.range().unwrap().half_width, 72);
    // A third widening gives the prediction up.
    let Step::Fallback(reason) = track.step(&candidates, 80.0, 92, None, &policy) else {
        panic!("three widenings are too many");
    };
    assert!(reason.contains("far from the predicted start"), "{reason}");
}

#[test]
fn a_format_that_does_not_fit_the_prediction_is_handed_to_the_staged_search() {
    let policy = smallest_policy();
    let step = |candidates: &[CandidateReport]| {
        Track::new(Codec::Jpeg).step(candidates, 80.0, 92, None, &policy)
    };
    let reason = |step: Step| match step {
        Step::Fallback(reason) => reason,
        other => panic!("{other:?}"),
    };

    // q60 passes although q61 fails: the inversion the staged search guards against.
    let inverted = [scored(Codec::Jpeg, 60, 82.0), scored(Codec::Jpeg, 61, 78.0)];
    assert!(
        reason(step(&inverted))
            .contains("non-monotonic behaviour detected for jpeg: q60 passes while q61 fails")
    );
    // One in a format that cannot beat the best is not worth the staged search.
    let best = best_of(Codec::Avif, 10);
    assert!(matches!(
        Track::new(Codec::Jpeg).step(&inverted, 80.0, 92, Some(&best), &policy),
        Step::Done
    ));
    // The score does not move with quality.
    let flat = [scored(Codec::Jpeg, 60, 74.0), scored(Codec::Jpeg, 70, 74.0)];
    assert!(reason(step(&flat)).contains("does not rise"));
    // The start missed by more than any estimate should.
    assert!(reason(step(&[scored(Codec::Jpeg, 60, 40.0)])).contains("first probe"));
    // A candidate failed to run.
    let mut broken = scored(Codec::Jpeg, 60, 74.0);
    broken.status = CandidateStatus::Failed("encoder crashed".to_string());
    assert!(reason(step(&[broken])).contains("failed before it could be judged"));
    // Nothing was judged at all.
    assert!(reason(step(&[])).contains("no quality was judged"));
}

#[test]
fn a_format_that_cannot_beat_the_best_is_dropped_by_the_size_of_its_failing_quality() {
    let policy = smallest_policy();
    // q60 failed at 2,200 bytes and q80 would be larger still.
    let candidates = [scored(Codec::Jpeg, 60, 74.0)];
    let best = best_of(Codec::Avif, 1_000);
    assert!(matches!(
        Track::new(Codec::Jpeg).step(&candidates, 80.0, 92, Some(&best), &policy),
        Step::Done
    ));
    let best = best_of(Codec::Avif, 5_000);
    assert!(matches!(
        Track::new(Codec::Jpeg).step(&candidates, 80.0, 92, Some(&best), &policy),
        Step::Probe { .. }
    ));
}

#[test]
fn a_quality_too_large_to_be_scored_limits_what_is_probed_below_it() {
    let policy = smallest_policy();
    let oversize = |quality: u8| {
        lossy(
            Codec::Jpeg,
            quality,
            20_000,
            CandidateStatus::Rejected("not smaller than original".to_string()),
        )
    };
    let step = |candidates: &[CandidateReport]| {
        Track::new(Codec::Jpeg).step(candidates, 80.0, 92, None, &policy)
    };
    // q70 is larger than the original, so nothing from q70 up can win: the
    // crossing, estimated far above, is looked for under it.
    let Step::Probe { qualities, .. } = step(&[scored(Codec::Jpeg, 65, 72.0), oversize(70)]) else {
        panic!("qualities between q65 and q70 are untried");
    };
    assert_eq!(qualities, vec![67, 68, 69]);
    // With q69 failing too, nothing is left under it.
    let measured = [
        scored(Codec::Jpeg, 65, 72.0),
        scored(Codec::Jpeg, 69, 74.0),
        oversize(70),
    ];
    assert_eq!(step(&measured), Step::Done);
}

#[test]
fn the_predicted_search_applies_only_where_its_start_was_fitted() {
    if !ToolInventory::detect().ssimulacra2.enabled {
        // Says what is missing, or fails when the tools are required.
        let _ = smart_compress_codecs::tools_or_skip(&["ssimulacra2"]);
        return;
    }
    let optimizer = |adjust: fn(&mut CliConfig)| {
        let mut config = CliConfig {
            search: crate::SearchMode::Predicted,
            gpu: cpu_only(),
            ..CliConfig::default()
        };
        adjust(&mut config);
        Optimizer::new(config)
    };
    let photo = |width: u32, height: u32| ImageProbe {
        path: String::new(),
        format: None,
        width,
        height,
        pixels: u64::from(width) * u64::from(height),
        color_type: "Rgb8".to_string(),
        original_color_type: "Rgb8".to_string(),
        has_alpha: false,
        metadata_size: 0,
        icc_profile_size: None,
        unique_color_ratio: 0.1,
        edge_density: 0.1,
        flat_region_ratio: 0.1,
        alpha_coverage: 0.0,
        orientation_applied: false,
        content_class: ContentClass::Photo,
    };
    assert!(optimizer(|_| {}).searches_predicted(4, &photo(64, 64)));
    for adjust in [
        |config: &mut CliConfig| config.search = crate::SearchMode::Staged,
        |config: &mut CliConfig| config.speed = crate::SpeedPreset::Fast,
        |config: &mut CliConfig| config.verbose = true,
        |config: &mut CliConfig| config.metrics.primary = PrimaryMetric::Builtin,
    ] {
        assert!(!optimizer(adjust).searches_predicted(4, &photo(64, 64)));
    }
    // One candidate at a time, an image ssimulacra2 cannot score, or a graphic.
    let predicted = optimizer(|_| {});
    assert!(!predicted.searches_predicted(1, &photo(64, 64)));
    assert!(!predicted.searches_predicted(4, &photo(4, 64)));
    let screenshot = ImageProbe {
        content_class: ContentClass::Screenshot,
        ..photo(64, 64)
    };
    assert!(!predicted.searches_predicted(4, &screenshot));
}

#[test]
fn the_predicted_search_starts_from_one_quality_and_ends_beside_a_failing_one() {
    let Some(_) = smart_compress_codecs::tools_or_skip(&["ssimulacra2"]) else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    // Large and textured enough to be classified as a photograph.
    let source = dir.path().join("source.jpg");
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % 24) as u32
    };
    RgbImage::from_fn(192, 192, |x, y| {
        Rgb([
            (x + noise()) as u8,
            (y + noise()) as u8,
            ((x + y) / 2 + noise()) as u8,
        ])
    })
    .save(&source)
    .unwrap();
    let run = |label: &str, search: crate::SearchMode| {
        let mut summary = Optimizer::new(CliConfig {
            inputs: vec![source.clone()],
            out_dir: Some(dir.path().join(label)),
            formats: CodecSelection::new([Codec::Jpeg]),
            search,
            allow_larger: true,
            min_quality: Some(60.0),
            workers: 4,
            gpu: cpu_only(),
            ..CliConfig::default()
        })
        .run()
        .unwrap();
        assert_eq!(summary.report.failures, 0);
        summary.report.reports.remove(0)
    };
    // (quality, passed) of every quality that was scored.
    let judged = |report: &ImageReport| {
        report
            .candidates
            .iter()
            .filter_map(|candidate| match candidate.settings {
                EncoderSettings::Quality { quality, .. } if candidate.quality.is_some() => Some((
                    quality,
                    matches!(candidate.status, CandidateStatus::Accepted),
                )),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let lowest_passing = |judged: &[(u8, bool)]| {
        judged
            .iter()
            .filter(|(_, passed)| *passed)
            .map(|(quality, _)| *quality)
            .min()
            .unwrap()
    };
    let staged = run("staged", crate::SearchMode::Staged);
    let predicted = run("predicted", crate::SearchMode::Predicted);

    let tried = judged(&predicted);
    let ends_at = lowest_passing(&tried);
    assert!(tried.contains(&(ends_at - 1, false)), "{tried:?}");
    assert!(ends_at <= lowest_passing(&judged(&staged)));
    assert_eq!(
        predicted.candidates[0].settings,
        EncoderSettings::Quality {
            quality: predicted_quality(Codec::Jpeg, 60.0, 92),
            effort: 0
        }
    );
    let size = |report: &ImageReport| report.candidates[report.winner_index.unwrap()].size_bytes;
    assert!(size(&predicted) <= size(&staged));
}
