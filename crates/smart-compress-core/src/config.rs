use std::{collections::HashSet, path::PathBuf, time::Duration};

use serde::{Deserialize, Deserializer, Serialize};
use smart_compress_codecs::Codec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Preset {
    Strict,
    Balanced,
    Aggressive,
}

impl Preset {
    /// Quality floor on the scale of the metric in use.
    ///
    /// SSIMULACRA2 floors follow its published bands: 90 is visually
    /// lossless, 80 is indistinguishable side by side, 70 is high quality.
    /// The built-in score has no perceptual scale; its floors are the values
    /// that landed on the same candidates for the sample photos.
    pub fn min_quality(self, perceptual: bool) -> f32 {
        match (self, perceptual) {
            (Self::Strict, true) => 90.0,
            (Self::Balanced, true) => 80.0,
            (Self::Aggressive, true) => 70.0,
            (Self::Strict, false) => 88.0,
            (Self::Balanced, false) => 73.0,
            (Self::Aggressive, false) => 68.0,
        }
    }

    pub fn compat_delta(self) -> f32 {
        match self {
            Self::Strict => 0.10,
            Self::Balanced => 0.05,
            Self::Aggressive => 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TargetPolicy {
    /// `modern-web`, which an earlier version offered as a separate target with
    /// the same behaviour, is still read as this.
    #[serde(alias = "modern-web")]
    Web,
    Archive,
    Smallest,
    MaxCompat,
}

impl TargetPolicy {
    /// The name `--target` takes.
    pub fn name(self) -> &'static str {
        match self {
            Self::Web => "web",
            Self::Archive => "archive",
            Self::Smallest => "smallest",
            Self::MaxCompat => "max-compat",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MetadataPolicy {
    Keep,
    Strip,
    Safe,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AlphaPolicy {
    Preserve,
    Flatten { background: String },
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReportFormat {
    Text,
    Json,
    Markdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SpeedPreset {
    Fast,
    Balanced,
    Exhaustive,
}

impl SpeedPreset {
    /// Coarse first-pass qualities. They only have to bracket the quality
    /// floor; the refinement pass then probes around the estimated crossing.
    pub fn qualities(self) -> &'static [u8] {
        match self {
            Self::Fast => &[88, 70],
            Self::Balanced => &[92, 78, 60],
            Self::Exhaustive => &[96, 88, 80, 70, 60, 48],
        }
    }

    /// Effort for the coarse first pass. AVIF is by far the slowest encoder
    /// here, so its coarse pass runs faster; the other formats keep their
    /// final effort, because a cheaper JPEG XL effort scores differently
    /// enough to misplace the floor. `fast` has no refinement pass.
    pub fn probe_effort(self, codec: Codec) -> u8 {
        match (codec, self) {
            (Codec::Avif, Self::Balanced) => 8,
            (Codec::Avif, Self::Exhaustive) => 6,
            _ => self.effort(codec),
        }
    }

    /// Refinement rounds after the coarse pass. Each round probes around
    /// the crossing estimated from everything scored so far.
    pub fn refinement_rounds(self) -> usize {
        match self {
            Self::Fast => 0,
            Self::Balanced => 2,
            Self::Exhaustive => 3,
        }
    }

    /// Qualities probed in a refinement round, as offsets from the estimated
    /// crossing: spread out first, then closing in.
    pub fn refinement_offsets(self, round: usize) -> &'static [i16] {
        match (self, round) {
            (Self::Fast, _) => &[],
            (Self::Balanced, 0) => &[-2, 0, 2],
            (Self::Exhaustive, 0) => &[-4, -2, 0, 2, 4],
            _ => &[-1, 0, 1],
        }
    }

    /// Encoder effort in each tool's own scale: `avifenc --speed` counts
    /// down toward slower, `cwebp -m` and `cjxl -e` count up.
    pub fn effort(self, codec: Codec) -> u8 {
        match (codec, self) {
            (Codec::Avif, Self::Fast) => 8,
            (Codec::Avif, Self::Balanced) => 6,
            (Codec::Avif, Self::Exhaustive) => 4,
            (Codec::Webp, Self::Exhaustive) => 6,
            (Codec::Webp, _) => 4,
            (Codec::Jxl, Self::Fast) => 3,
            (Codec::Jxl, Self::Balanced) => 5,
            (Codec::Jxl, Self::Exhaustive) => 7,
            // The JPEG and PNG paths take no effort setting.
            (Codec::Jpeg | Codec::Png, _) => 0,
        }
    }

    pub fn timeout(self) -> Duration {
        match self {
            Self::Fast => Duration::from_secs(30),
            Self::Balanced => Duration::from_secs(120),
            Self::Exhaustive => Duration::from_secs(300),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecSelection {
    allowed: HashSet<Codec>,
}

impl CodecSelection {
    pub fn new(codecs: impl IntoIterator<Item = Codec>) -> Self {
        Self {
            allowed: codecs.into_iter().collect(),
        }
    }

    pub fn default_web() -> Self {
        Self::new([Codec::Png, Codec::Jpeg, Codec::Webp, Codec::Avif])
    }

    pub fn contains(&self, codec: Codec) -> bool {
        self.allowed.contains(&codec)
    }

    /// The formats as `--formats` takes them, for messages.
    pub fn names(&self) -> String {
        self.iter()
            .map(|codec| format!("{codec:?}").to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn iter(&self) -> impl Iterator<Item = Codec> + '_ {
        [
            Codec::Jpeg,
            Codec::Webp,
            Codec::Avif,
            Codec::Jxl,
            Codec::Png,
        ]
        .into_iter()
        .filter(|codec| self.allowed.contains(codec))
    }

    pub fn apply_target(&mut self, target: TargetPolicy) {
        match target {
            TargetPolicy::Web => {}
            TargetPolicy::Archive => {
                self.allowed.insert(Codec::Jxl);
                self.allowed.remove(&Codec::Jpeg);
            }
            TargetPolicy::Smallest => {
                self.allowed.insert(Codec::Jxl);
            }
            TargetPolicy::MaxCompat => {
                self.allowed
                    .retain(|codec| matches!(codec, Codec::Jpeg | Codec::Png));
            }
        }
    }
}

/// The separate uses of `workers`, each overridable on its own so the
/// benchmark can vary one dimension while the others stay fixed. `None` keeps
/// the value derived from `workers` and the image size.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ParallelismOverrides {
    /// Images processed at the same time.
    pub files: Option<usize>,
    /// Candidates of one image evaluated at the same time.
    pub candidates: Option<usize>,
    /// Threads each external encoder is told to use.
    pub encoder_threads: Option<usize>,
    /// Threads the built-in CPU metric runs on.
    pub metric_threads: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct CliConfig {
    pub inputs: Vec<PathBuf>,
    pub out_dir: Option<PathBuf>,
    pub out: Option<PathBuf>,
    pub recursive: bool,
    pub dry_run: bool,
    /// Replace an output that already exists.
    pub overwrite: bool,
    /// Refuse to write over a file that is an input.
    pub protect_inputs: bool,
    pub preset: Preset,
    pub target: TargetPolicy,
    pub formats: CodecSelection,
    pub report: ReportFormat,
    pub report_out: Option<PathBuf>,
    pub resume_report: Option<PathBuf>,
    /// Explicit floor; `None` takes the preset's floor for the metric in use.
    pub min_quality: Option<f32>,
    pub quality_slack: f32,
    pub metadata: MetadataPolicy,
    pub alpha: AlphaPolicy,
    pub speed: SpeedPreset,
    pub candidate_timeout: Option<Duration>,
    pub max_image_time: Option<Duration>,
    pub max_candidate_attempts: Option<usize>,
    pub max_temp_bytes: Option<u64>,
    pub max_run_temp_bytes: Option<u64>,
    /// Memory the run plans on using; `None` takes two thirds of the physical
    /// memory.
    pub max_memory_bytes: Option<u64>,
    pub workers: usize,
    pub parallelism: ParallelismOverrides,
    pub allow_larger: bool,
    pub max_input_bytes: u64,
    pub max_pixels: u64,
    pub max_width: u32,
    pub max_height: u32,
    pub max_decoded_bytes: u64,
    pub max_metadata_bytes: u64,
    pub max_bits_per_pixel: u16,
    pub gpu: GpuConfig,
    pub cache: CacheConfig,
    pub metrics: MetricsConfig,
    pub verbose: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        let preset = Preset::Balanced;
        let target = TargetPolicy::Web;
        let mut formats = CodecSelection::default_web();
        formats.apply_target(target);
        Self {
            inputs: Vec::new(),
            out_dir: None,
            out: None,
            recursive: false,
            dry_run: false,
            overwrite: false,
            protect_inputs: true,
            preset,
            target,
            formats,
            report: ReportFormat::Text,
            report_out: None,
            resume_report: None,
            min_quality: None,
            quality_slack: 0.0,
            metadata: MetadataPolicy::Safe,
            alpha: AlphaPolicy::Preserve,
            speed: SpeedPreset::Balanced,
            candidate_timeout: None,
            max_image_time: None,
            max_candidate_attempts: None,
            max_temp_bytes: None,
            max_run_temp_bytes: None,
            max_memory_bytes: None,
            workers: std::thread::available_parallelism().map_or(1, usize::from),
            parallelism: ParallelismOverrides::default(),
            allow_larger: false,
            max_input_bytes: crate::image_probe::DEFAULT_MAX_INPUT_BYTES,
            max_pixels: 100_000_000,
            max_width: crate::image_probe::DEFAULT_MAX_IMAGE_WIDTH,
            max_height: crate::image_probe::DEFAULT_MAX_IMAGE_HEIGHT,
            max_decoded_bytes: crate::image_probe::DEFAULT_MAX_DECODED_BYTES,
            max_metadata_bytes: crate::image_probe::DEFAULT_MAX_METADATA_BYTES,
            max_bits_per_pixel: 64,
            gpu: GpuConfig::default(),
            cache: CacheConfig::default(),
            metrics: MetricsConfig::default(),
            verbose: false,
        }
    }
}

impl CliConfig {
    pub fn candidate_timeout(&self) -> Duration {
        self.candidate_timeout
            .unwrap_or_else(|| self.speed.timeout())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CacheConfig {
    pub enabled: bool,
    pub dir: Option<PathBuf>,
    #[serde(default)]
    pub max_bytes: Option<u64>,
    #[serde(default)]
    pub max_age_days: Option<u64>,
}

/// Which score a candidate must reach to be accepted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrimaryMetric {
    /// SSIMULACRA2 when its tool is installed, otherwise the built-in score.
    #[default]
    Auto,
    Ssimulacra2,
    /// SSIMULACRA2 computed inside the program against a reference built once
    /// per image; needs no `ssimulacra2` tool.
    Ssimulacra2Inprocess,
    /// The built-in score: the lowest of the RGB, edge and (with transparency)
    /// alpha and composite agreements. `psnr` is its earlier name.
    #[serde(alias = "psnr")]
    Builtin,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MetricsConfig {
    #[serde(default)]
    pub primary: PrimaryMetric,
    #[serde(default)]
    pub ssimulacra2: bool,
    #[serde(default)]
    pub butteraugli: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GpuConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub required: bool,
    #[serde(default = "default_gpu_backend")]
    pub backend: String,
    #[serde(default)]
    pub adapter_name: Option<String>,
    #[serde(default)]
    pub adapter_vendor: Option<u32>,
    #[serde(default)]
    pub adapter_device: Option<u32>,
    #[serde(default = "default_true")]
    pub prefer_gpu_metrics: bool,
    #[serde(default = "default_gpu_min_pixels")]
    pub min_pixels_for_gpu_metrics: u64,
    #[serde(default)]
    pub prefer_gpu_preprocess: bool,
    #[serde(default, deserialize_with = "deserialize_max_vram_bytes")]
    pub max_vram_bytes: Option<u64>,
    #[serde(default)]
    pub verify_gpu: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            required: false,
            backend: "wgpu".to_string(),
            adapter_name: None,
            adapter_vendor: None,
            adapter_device: None,
            prefer_gpu_metrics: true,
            min_pixels_for_gpu_metrics: default_gpu_min_pixels(),
            prefer_gpu_preprocess: false,
            max_vram_bytes: None,
            verify_gpu: false,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_gpu_backend() -> String {
    "wgpu".to_string()
}

/// The benchmark matrix found GPU scoring faster than CPU scoring in every
/// size class, including images of a few thousand pixels, so no image is
/// routed to the CPU by size.
fn default_gpu_min_pixels() -> u64 {
    0
}

fn deserialize_max_vram_bytes<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum MaxVram {
        Bytes(u64),
        Text(String),
    }

    match Option::<MaxVram>::deserialize(deserializer)? {
        Some(MaxVram::Bytes(bytes)) => Ok(Some(bytes)),
        Some(MaxVram::Text(value)) if value.eq_ignore_ascii_case("auto") => Ok(None),
        Some(MaxVram::Text(value)) => value
            .parse::<u64>()
            .map(Some)
            .map_err(serde::de::Error::custom),
        None => Ok(None),
    }
}

/// The config file. Every section rejects keys it does not know, so a typo is
/// reported instead of silently ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    pub defaults: Option<FileDefaults>,
    pub codecs: Option<FileCodecs>,
    pub performance: Option<FilePerformance>,
    pub gpu: Option<GpuConfig>,
    pub cache: Option<CacheConfig>,
    pub metrics: Option<MetricsConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDefaults {
    pub preset: Option<Preset>,
    pub target: Option<TargetPolicy>,
    pub min_quality: Option<f32>,
    pub quality_slack: Option<f32>,
    pub report: Option<ReportFormat>,
    pub report_out: Option<PathBuf>,
    pub metadata: Option<MetadataPolicy>,
    pub alpha: Option<String>,
    pub background: Option<String>,
    pub speed: Option<SpeedPreset>,
    pub overwrite: Option<bool>,
    /// `keep_original` is the earlier name.
    #[serde(alias = "keep_original")]
    pub protect_inputs: Option<bool>,
    pub out_dir: Option<PathBuf>,
    pub recursive: Option<bool>,
    pub allow_larger: Option<bool>,
    pub max_pixels: Option<u64>,
    pub max_input_bytes: Option<u64>,
    pub max_width: Option<u32>,
    pub max_height: Option<u32>,
    pub max_decoded_bytes: Option<u64>,
    pub max_metadata_bytes: Option<u64>,
    pub max_bits_per_pixel: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileCodecs {
    pub avif: Option<bool>,
    pub webp: Option<bool>,
    pub jpeg: Option<bool>,
    pub png: Option<bool>,
    pub jxl: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilePerformance {
    pub workers: Option<usize>,
    pub candidate_timeout_seconds: Option<u64>,
    pub max_image_time_seconds: Option<u64>,
    pub max_candidate_attempts: Option<usize>,
    pub max_temp_bytes: Option<u64>,
    pub max_run_temp_bytes: Option<u64>,
    pub max_memory_bytes: Option<u64>,
}

/// Reads a config file. An unknown key is an error that names the key and the
/// section it is in.
pub fn parse_file_config(text: &str) -> anyhow::Result<FileConfig> {
    toml::from_str(text).map_err(|error| {
        // A key belongs to the last table header above it. An unknown section
        // is itself a header and belongs to none.
        let section = error.span().and_then(|span| {
            let line_start = text[..span.start].rfind('\n').map_or(0, |at| at + 1);
            if text[line_start..].trim_start().starts_with('[') {
                return None;
            }
            text[..line_start]
                .lines()
                .rev()
                .find(|line| line.trim_start().starts_with('['))
        });
        let message = error.to_string();
        let message = message.trim_end();
        match section {
            Some(section) => anyhow::anyhow!("{message}\nthe key is in section {}", section.trim()),
            None => anyhow::anyhow!("{message}"),
        }
    })
}

pub fn parse_codec_list(value: &str) -> anyhow::Result<CodecSelection> {
    let mut codecs = Vec::new();
    for raw in value.split(',') {
        let codec = match raw.trim().to_ascii_lowercase().as_str() {
            "png" => Codec::Png,
            "jpeg" | "jpg" => Codec::Jpeg,
            "webp" => Codec::Webp,
            "avif" => Codec::Avif,
            "jxl" | "jpeg-xl" => Codec::Jxl,
            other => anyhow::bail!("unknown codec '{other}'"),
        };
        codecs.push(codec);
    }
    Ok(CodecSelection::new(codecs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_config_defaults_apply_to_partial_file_config() {
        let parsed: FileConfig = toml::from_str(
            r#"
            [gpu]
            max_vram_bytes = "auto"
            "#,
        )
        .unwrap();

        let gpu = parsed.gpu.unwrap();
        assert!(gpu.enabled);
        assert_eq!(gpu.backend, "wgpu");
        assert!(gpu.prefer_gpu_metrics);
        assert!(!gpu.prefer_gpu_preprocess);
        assert_eq!(gpu.max_vram_bytes, None);
    }

    #[test]
    fn an_unknown_key_is_rejected_with_its_name_and_section() {
        for (text, key, section) in [
            ("[gpu]\nenabeld = false\n", "enabeld", Some("[gpu]")),
            (
                "[defaults]\npreset = \"strict\"\nqualty_slack = 1\n",
                "qualty_slack",
                Some("[defaults]"),
            ),
            ("[codecs]\njpg = false\n", "jpg", Some("[codecs]")),
            (
                "[performance]\nworker = 3\n",
                "worker",
                Some("[performance]"),
            ),
            (
                "[cache]\nenabled = true\nfolder = \"x\"\n",
                "folder",
                Some("[cache]"),
            ),
            (
                "[metrics]\nprimery = \"auto\"\n",
                "primery",
                Some("[metrics]"),
            ),
            ("[defalts]\npreset = \"strict\"\n", "defalts", None),
            ("verbose = true\n[gpu]\nenabled = true\n", "verbose", None),
        ] {
            let error = parse_file_config(text).unwrap_err().to_string();
            assert!(error.contains(&format!("`{key}`")), "{error}");
            match section {
                Some(section) => assert!(error.contains(&format!("section {section}")), "{error}"),
                None => assert!(!error.contains("in section"), "{error}"),
            }
        }
        assert!(
            parse_file_config("[gpu]\nenabled = false\n[defaults]\npreset = \"strict\"").is_ok()
        );
    }

    #[test]
    fn the_earlier_names_in_a_config_file_still_read() {
        let file = parse_file_config(
            "[defaults]\ntarget = \"modern-web\"\nkeep_original = false\n[metrics]\nprimary = \"psnr\"",
        )
        .unwrap();
        let defaults = file.defaults.unwrap();
        assert_eq!(defaults.target, Some(TargetPolicy::Web));
        assert_eq!(defaults.protect_inputs, Some(false));
        assert_eq!(file.metrics.unwrap().primary, PrimaryMetric::Builtin);
    }

    #[test]
    fn gpu_config_accepts_numeric_max_vram_bytes() {
        let parsed: FileConfig = toml::from_str(
            r#"
            [gpu]
            max_vram_bytes = 4096
            "#,
        )
        .unwrap();

        assert_eq!(parsed.gpu.unwrap().max_vram_bytes, Some(4096));
    }
}
