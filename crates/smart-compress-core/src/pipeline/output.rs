use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use smart_compress_codecs::{Codec, MetadataPayload, icc_is_plain_srgb};

use super::{DEFAULT_OUTPUT_DIR, Optimizer, folder_of};
use crate::{
    atomic_file::atomic_copy, candidate::CandidateReport, config::MetadataPolicy,
    image_probe::SourceMetadata,
};

impl Optimizer {
    pub(super) fn output_dir(&self, input: &Path) -> PathBuf {
        self.config.out_dir.clone().unwrap_or_else(|| {
            input
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(DEFAULT_OUTPUT_DIR)
        })
    }

    pub(super) fn write_winner(
        &self,
        input: &Path,
        candidate: &CandidateReport,
    ) -> Result<PathBuf> {
        let source = candidate
            .path
            .as_ref()
            .ok_or_else(|| anyhow!("winner does not have a candidate path"))?;
        let destination = self.output_path(input, candidate.codec)?;
        if self.config.protect_inputs && destination.exists() {
            if let (Ok(existing), Ok(source_input)) =
                (fs::canonicalize(&destination), fs::canonicalize(input))
                && existing == source_input
            {
                anyhow::bail!(
                    "output {} would replace the original; choose another output location or pass --protect-inputs=false",
                    destination.display()
                );
            }
            // A file that is already there, in a folder with images to optimize,
            // may be an original of the user's that is not part of this run.
            if fs::canonicalize(folder_of(&destination)).is_ok_and(|folder| {
                self.input_folders
                    .lock()
                    .expect("input folder lock poisoned")
                    .contains(&folder)
            }) {
                anyhow::bail!(
                    "output {} would replace a file that was already there, in a folder that holds input images; --protect-inputs protects it. Choose another output folder or pass --protect-inputs=false",
                    destination.display()
                );
            }
        }
        atomic_copy(source, &destination, !self.config.overwrite)
            .with_context(|| format!("copying winner to {}", destination.display()))?;
        Ok(destination)
    }

    pub(super) fn output_path(&self, input: &Path, codec: Codec) -> Result<PathBuf> {
        if let Some(out) = &self.config.out {
            // The extension names the format, and only that format was tried; a
            // name without a known image extension gets the winner's added.
            if self.out_codec.is_some() {
                return Ok(out.clone());
            }
            let mut named = out.clone().into_os_string();
            named.push(".");
            named.push(codec.extension());
            return Ok(PathBuf::from(named));
        }
        let file_stem = input
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| anyhow!("input has no valid file stem: {}", input.display()))?;
        let ambiguous = self
            .ambiguous_outputs
            .lock()
            .expect("output name lock poisoned")
            .contains(input);
        let filename = if ambiguous {
            let tag = blake3::hash(input.to_string_lossy().as_bytes()).to_hex();
            format!("{file_stem}-{}.{}", &tag[..8], codec.extension())
        } else {
            format!("{file_stem}.{}", codec.extension())
        };
        Ok(self.output_dir(input).join(filename))
    }
}

pub(super) fn metadata_payload(
    metadata: &SourceMetadata,
    policy: MetadataPolicy,
) -> MetadataPayload {
    match policy {
        // A profile that is not plain sRGB says what the pixel values mean, so
        // it is colour data: dropping it would show different colours.
        MetadataPolicy::Strip => MetadataPayload {
            icc_profile: metadata
                .icc_profile
                .clone()
                .filter(|profile| !icc_is_plain_srgb(profile)),
            exif_metadata: None,
        },
        MetadataPolicy::Safe => MetadataPayload {
            icc_profile: metadata.icc_profile.clone(),
            exif_metadata: None,
        },
        MetadataPolicy::Keep => MetadataPayload {
            icc_profile: metadata.icc_profile.clone(),
            exif_metadata: metadata.sanitized_exif_metadata(),
        },
    }
}
