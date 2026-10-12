use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use image::DynamicImage;
use serde::{Deserialize, Serialize};
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
