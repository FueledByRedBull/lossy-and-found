use std::collections::HashMap;

use smart_compress_codecs::{Codec, EncoderSettings};

use super::{Optimizer, StagedPlan, codec_name};
use crate::{
    candidate::CandidateReport,
    config::AlphaPolicy,
    decision::{AcceptedCandidate, DecisionPolicy},
    image_probe::ImageProbe,
};

impl Optimizer {
    pub(super) fn plan_candidates(&self, probe: &ImageProbe) -> Vec<(Codec, EncoderSettings)> {
        let mut plan = Vec::new();
        let speed = self.config.speed;
        let large_lossy_source = probe.pixels > 50_000_000
            && probe
                .format
                .as_deref()
                .is_some_and(|format| format.eq_ignore_ascii_case("Jpeg"))
            && !probe.has_alpha
            && !matches!(self.config.target, crate::config::TargetPolicy::Archive);
        for codec in self
            .config
            .formats
            .iter()
            .filter(|codec| self.out_codec.is_none_or(|named| named == *codec))
        {
            if probe.has_alpha
                && !codec.supports_alpha()
                && matches!(self.config.alpha, AlphaPolicy::Preserve)
            {
                continue;
            }
            if !self.backend.tools().available_for(codec) {
                continue;
            }
            match codec {
                Codec::Png => {
                    if !large_lossy_source {
                        plan.push((codec, EncoderSettings::Lossless));
                        if self.backend.tools().pngquant.enabled
                            && probe
                                .format
                                .as_deref()
                                .is_some_and(|format| format.eq_ignore_ascii_case("Png"))
                        {
                            for colors in [256, 128, 64, 32] {
                                plan.push((codec, EncoderSettings::Palette { colors }));
                            }
                        }
                    }
                }
                Codec::Jpeg => {
                    for quality in speed.qualities() {
                        plan.push((
                            codec,
                            EncoderSettings::Quality {
                                quality: *quality,
                                effort: speed.probe_effort(codec),
                            },
                        ));
                    }
                }
                Codec::Webp | Codec::Avif | Codec::Jxl => {
                    if !large_lossy_source {
                        plan.push((codec, EncoderSettings::Lossless));
                    }
                    for quality in speed.qualities() {
                        plan.push((
                            codec,
                            EncoderSettings::Quality {
                                quality: *quality,
                                effort: speed.probe_effort(codec),
                            },
                        ));
                    }
                }
            }
        }
        plan
    }

    /// Why no candidate could be planned, per selected format.
    pub(super) fn unplanned_reasons(&self, probe: &ImageProbe) -> String {
        self.config
            .formats
            .iter()
            .filter(|codec| self.out_codec.is_none_or(|named| named == *codec))
            .map(|codec| {
                let missing = self.backend.tools().missing_for(codec);
                let name = codec_name(codec);
                if probe.has_alpha
                    && !codec.supports_alpha()
                    && matches!(self.config.alpha, AlphaPolicy::Preserve)
                {
                    format!("{name} cannot store transparency (try --alpha flatten)")
                } else if !missing.is_empty() {
                    format!(
                        "{name} needs {}, which was not found",
                        missing.join(" and ")
                    )
                } else {
                    format!("{name} is not tried on a very large JPEG")
                }
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// How much larger than the largest file that could still win the lossless WebP
/// probe must be before the other lossless encoders are skipped.
///
/// Another lossless encoder can only get below the ceiling if it beats the
/// probe by more than this factor. Measured on 63 JPEG sources (the six stock
/// photos, 15 re-encodes of three of them at q40 to q100, six corpus JPEGs and
/// 36 corpus graphics saved as JPEG at q60/q90) the probe was at most 1.65x the
/// smallest of the four lossless encoders (1.23x on photographs), and 2.10x on
/// 24 corpus PNG sources. The native PNG encoder and AVIF are cheaper to run
/// but up to 4400x off on graphics, so they cannot gate. On the 22 of those
/// JPEG sources where some lossless encoder does beat the source, the probe was
/// at most 0.95x the source. The gate stops being safe if a source ever has a
/// ratio above 3.0.
const LOSSLESS_GATE_MARGIN: f64 = 3.0;

/// The lossless candidate the others are gated on. WebP lossless tracks the
/// smallest lossless size most closely of the encoders that are cheap enough.
pub(super) fn is_lossless_probe(codec: Codec, settings: &EncoderSettings) -> bool {
    codec == Codec::Webp && *settings == EncoderSettings::Lossless
}

/// Whether the lossless probe may gate the other lossless encoders for this
/// image. Every lossless candidate is a pixel-exact re-encode (`cjxl` is run
/// with `--lossless_jpeg=0`, so a JPEG is never transcoded), which only a
/// lossless source can beat by much: sources that may be lossless (PNG, WebP,
/// AVIF) run all of them, as does a plan with nothing to gate.
pub(super) fn lossless_gate_applies(probe: &ImageProbe, plan: &[(Codec, EncoderSettings)]) -> bool {
    probe
        .format
        .as_deref()
        .is_some_and(|format| format.eq_ignore_ascii_case("Jpeg"))
        && plan
            .iter()
            .any(|(codec, settings)| is_lossless_probe(*codec, settings))
        && plan.iter().any(|(codec, settings)| {
            *settings == EncoderSettings::Lossless && *codec != Codec::Webp
        })
}

/// Why a lossless candidate other than the probe is not worth encoding, given
/// the probe's size: no lossless encoder gets under the largest file that could
/// still win if the probe is [`LOSSLESS_GATE_MARGIN`] times over it.
pub(super) fn lossless_gate(
    candidates: &[CandidateReport],
    codec: Codec,
    settings: &EncoderSettings,
    policy: &DecisionPolicy,
    best: Option<&AcceptedCandidate>,
) -> Option<String> {
    if *settings != EncoderSettings::Lossless || is_lossless_probe(codec, settings) {
        return None;
    }
    let probe = candidates
        .iter()
        .find(|candidate| is_lossless_probe(candidate.codec, &candidate.settings))?
        .size_bytes?;
    let ceiling = policy.size_ceiling(best)?;
    (probe as f64 > LOSSLESS_GATE_MARGIN * ceiling as f64).then(|| {
        format!(
            "cannot win: lossless WebP is {probe} bytes, over {LOSSLESS_GATE_MARGIN}x the {ceiling} bytes a winner can be, which no lossless encoder closes"
        )
    })
}

/// Runs only the middle quality of each lossy format first; qualities of other
/// kinds (lossless, palette) and the rest of each format's grid are held back.
/// When the lossless probe gates the other lossless encoders it runs with the
/// first stage, last, so that the second stage knows its size.
pub(super) fn stage_plan(plan: Vec<(Codec, EncoderSettings)>, gated: bool) -> StagedPlan {
    let mut grids: HashMap<Codec, Vec<u8>> = HashMap::new();
    for (codec, settings) in &plan {
        if let EncoderSettings::Quality { quality, .. } = settings {
            grids.entry(*codec).or_default().push(*quality);
        }
    }
    let mids: HashMap<Codec, u8> = grids
        .into_iter()
        .map(|(codec, qualities)| (codec, qualities[qualities.len() / 2]))
        .collect();
    let (mut first, mut held): (Vec<_>, Vec<_>) = plan.into_iter().partition(|(codec, settings)| {
        matches!(settings, EncoderSettings::Quality { quality, .. } if mids.get(codec) == Some(quality))
    });
    if gated
        && let Some(at) = held
            .iter()
            .position(|(codec, settings)| is_lossless_probe(*codec, settings))
    {
        first.push(held.remove(at));
    }
    StagedPlan { first, held, mids }
}
