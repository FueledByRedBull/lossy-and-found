use std::{
    ffi::OsString,
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use image::{
    DynamicImage, ImageEncoder,
    codecs::{jpeg::JpegEncoder, png::PngEncoder},
};
use tempfile::TempDir;

use crate::{
    budget::cancellation_requested,
    icc::{icc_color_space, icc_is_plain_srgb},
    process::{ProcessFailure, ProcessFailureKind, run_command, work_unit},
    tools::ToolInventory,
    types::{Codec, EncodeError, EncodeJob, EncodedArtifact, EncoderSettings, MetadataPayload},
};

pub struct EncoderBackend {
    tools: ToolInventory,
}

impl EncoderBackend {
    pub fn new(tools: ToolInventory) -> Self {
        Self { tools }
    }

    pub fn tools(&self) -> &ToolInventory {
        &self.tools
    }

    pub fn encode(&self, job: EncodeJob<'_>) -> Result<EncodedArtifact> {
        if job.has_alpha && !job.codec.supports_alpha() {
            return Err(anyhow!(EncodeError::Failed {
                codec: job.codec,
                message: "codec cannot preserve alpha".to_string(),
            }));
        }

        match job.codec {
            Codec::Png => self.encode_png(job),
            Codec::Jpeg => self.encode_jpeg(job),
            Codec::Webp => self.encode_webp(job),
            Codec::Avif => self.encode_avif(job),
            Codec::Jxl => self.encode_jxl(job),
        }
    }

    fn encode_png(&self, job: EncodeJob<'_>) -> Result<EncodedArtifact> {
        let start = Instant::now();
        if let EncoderSettings::Palette { colors } = job.settings {
            if !job.metadata.is_empty() {
                return Err(anyhow!(EncodeError::Declined(
                    "pngquant palette mode is skipped when metadata must be preserved".to_string()
                )));
            }
            let Some(path) = &self.tools.pngquant.path else {
                return Err(anyhow!(EncodeError::Unavailable(Codec::Png)));
            };
            let run = run_command(
                path,
                &[
                    OsString::from("--force"),
                    OsString::from("--skip-if-larger"),
                    OsString::from("--output"),
                    job.output_path.as_os_str().to_os_string(),
                    OsString::from(colors.to_string()),
                    job.normalized_png.as_os_str().to_os_string(),
                ],
                job.timeout,
                1,
            );
            // pngquant exits with 98 when the palette file would be larger than
            // the input (`--skip-if-larger`) and with 99 when it cannot reach
            // the quality asked for: it made no file, and that is its answer.
            if let Err(error) = run {
                return Err(match error.downcast_ref::<ProcessFailure>() {
                    Some(failure)
                        if failure.kind == ProcessFailureKind::Exit
                            && matches!(failure.exit_code, Some(98 | 99)) =>
                    {
                        anyhow!(EncodeError::Declined(format!(
                            "pngquant made no {colors}-colour file: {failure}"
                        )))
                    }
                    _ => error,
                });
            }
            return artifact(job, "pngquant", start.elapsed(), Vec::new());
        }

        {
            let _unit = work_unit(job.timeout)?;
            write_png(job.output_path, job.original, job.metadata)?;
        }

        let mut warnings = Vec::new();
        if !job.metadata.is_empty() {
            warnings.push("oxipng skipped to preserve embedded metadata".to_string());
        } else if self.tools.oxipng.enabled
            && let Some(path) = &self.tools.oxipng.path
        {
            let result = run_command(
                path,
                &[
                    OsString::from("-o"),
                    OsString::from("2"),
                    OsString::from("--strip"),
                    OsString::from("safe"),
                    OsString::from("--threads"),
                    OsString::from(job.threads.max(1).to_string()),
                    job.output_path.as_os_str().to_os_string(),
                ],
                job.timeout,
                job.threads,
            );
            if let Err(error) = result {
                warnings.push(format!("oxipng skipped: {error}"));
            }
        }

        artifact(job, "native-png+optional-oxipng", start.elapsed(), warnings)
    }

    fn encode_jpeg(&self, job: EncodeJob<'_>) -> Result<EncodedArtifact> {
        let EncoderSettings::Quality { quality, .. } = job.settings else {
            return Err(anyhow!("JPEG requires quality settings"));
        };

        let start = Instant::now();
        // cjpegli carries the ICC profile over from the normalized PNG but
        // cannot attach EXIF, so retained EXIF needs the native encoder.
        let mut warnings = Vec::new();
        if job.metadata.exif_metadata.is_none()
            && let Some(path) = &self.tools.cjpegli.path
        {
            match run_command(
                path,
                &[
                    job.normalized_png.as_os_str().to_os_string(),
                    job.output_path.as_os_str().to_os_string(),
                    OsString::from(format!("--quality={quality}")),
                    OsString::from("--progressive_level=2"),
                ],
                job.timeout,
                1,
            ) {
                Ok(()) => return artifact(job, "cjpegli", start.elapsed(), Vec::new()),
                Err(error) if cancellation_requested() => return Err(error),
                // cjpegli refuses inputs it cannot interpret, such as a
                // damaged colour profile; the built-in encoder still can.
                Err(error) => warnings.push(format!(
                    "cjpegli failed, used the built-in JPEG encoder: {error}"
                )),
            }
        }

        // The JPEG encoder emits tiny writes; unbuffered they dominate encode time.
        let unit = work_unit(job.timeout)?;
        let mut file = BufWriter::new(
            fs::File::create(job.output_path)
                .with_context(|| format!("creating {}", job.output_path.display()))?,
        );
        let mut encoder = JpegEncoder::new_with_quality(&mut file, quality);
        apply_metadata(&mut encoder, job.metadata)?;
        // A grey source stays single-channel so a grey ICC profile still fits
        // it. The buffers are passed concretely: through `DynamicImage` the
        // encoder always sees RGBA and writes three components.
        if job.original.color().has_color() {
            encoder.encode_image(&job.original.to_rgb8())
        } else {
            encoder.encode_image(&job.original.to_luma8())
        }
        .with_context(|| format!("encoding JPEG {}", job.output_path.display()))?;
        file.flush()
            .with_context(|| format!("flushing JPEG {}", job.output_path.display()))?;
        drop(file);
        drop(unit);
        let backend = if job.metadata.is_empty() {
            "native-jpeg"
        } else {
            "native-jpeg+metadata"
        };
        artifact(job, backend, start.elapsed(), warnings)
    }

    fn encode_webp(&self, job: EncodeJob<'_>) -> Result<EncodedArtifact> {
        let Some(path) = &self.tools.cwebp.path else {
            return Err(anyhow!(EncodeError::Unavailable(Codec::Webp)));
        };

        let mut args = Vec::new();
        match job.settings {
            EncoderSettings::Lossless => {
                args.push(OsString::from("-lossless"));
                args.push(OsString::from("-m"));
                args.push(OsString::from("4"));
            }
            EncoderSettings::Quality { quality, effort } => {
                args.push(OsString::from("-q"));
                args.push(OsString::from(quality.to_string()));
                args.push(OsString::from("-m"));
                args.push(OsString::from(effort.to_string()));
            }
            EncoderSettings::Palette { .. } => {
                return Err(anyhow!("WebP does not use palette settings"));
            }
        }
        if job.threads >= 2 {
            // Multithreading does not change the bitstream.
            args.push(OsString::from("-mt"));
        }
        args.push(job.normalized_png.as_os_str().to_os_string());
        args.push(OsString::from("-o"));
        args.push(job.output_path.as_os_str().to_os_string());

        let mut metadata = job.metadata.clone();
        let mut warnings = Vec::new();
        // WebP always stores colour, so a grey profile would no longer match.
        // Without it a viewer assumes sRGB, which is only the same picture when
        // the profile is plain sRGB.
        if let Some(profile) = metadata
            .icc_profile
            .as_deref()
            .filter(|profile| icc_color_space(profile).as_ref() == Some(b"GRAY"))
        {
            anyhow::ensure!(
                icc_is_plain_srgb(profile),
                "WebP stores colour only and cannot carry this grey ICC profile, which is not plain sRGB"
            );
            metadata.icc_profile = None;
            warnings.push("grey ICC profile not attached: WebP stores colour only".to_string());
        }
        let start = Instant::now();
        run_command(path, &args, job.timeout, job.threads.clamp(1, 2))?;
        if !metadata.is_empty() {
            self.attach_webp_metadata(job.output_path, &metadata, job.timeout)?;
        }
        artifact(job, "cwebp", start.elapsed(), warnings)
    }

    fn encode_avif(&self, job: EncodeJob<'_>) -> Result<EncodedArtifact> {
        let Some(path) = &self.tools.avifenc.path else {
            return Err(anyhow!(EncodeError::Unavailable(Codec::Avif)));
        };

        let mut args = Vec::new();
        match job.settings {
            EncoderSettings::Lossless => args.push(OsString::from("--lossless")),
            EncoderSettings::Quality { quality, effort } => {
                args.push(OsString::from("-q"));
                args.push(OsString::from(quality.to_string()));
                args.push(OsString::from("--speed"));
                args.push(OsString::from(effort.to_string()));
            }
            EncoderSettings::Palette { .. } => {
                return Err(anyhow!("AVIF does not use palette settings"));
            }
        }
        // One job produces a different bitstream than two or more, so stay
        // at two or above to keep results independent of the thread budget.
        args.push(OsString::from("-j"));
        args.push(OsString::from(job.threads.max(2).to_string()));
        args.push(job.normalized_png.as_os_str().to_os_string());
        args.push(job.output_path.as_os_str().to_os_string());

        let start = Instant::now();
        run_command(path, &args, job.timeout, job.threads.max(2))?;
        artifact(job, "avifenc", start.elapsed(), Vec::new())
    }

    fn encode_jxl(&self, job: EncodeJob<'_>) -> Result<EncodedArtifact> {
        let Some(path) = &self.tools.cjxl.path else {
            return Err(anyhow!(EncodeError::Unavailable(Codec::Jxl)));
        };

        let mut args = Vec::new();
        match job.settings {
            EncoderSettings::Lossless => {
                // Without an explicit distance cjxl encodes lossily at d=1.
                args.push(OsString::from("-d"));
                args.push(OsString::from("0"));
                args.push(OsString::from("--lossless_jpeg=0"));
            }
            EncoderSettings::Quality { quality, effort } => {
                args.push(OsString::from("-q"));
                args.push(OsString::from(quality.to_string()));
                args.push(OsString::from("-e"));
                args.push(OsString::from(effort.to_string()));
                args.push(OsString::from("--lossless_jpeg=0"));
                args.push(OsString::from("-m"));
                args.push(OsString::from("0"));
            }
            EncoderSettings::Palette { .. } => {
                return Err(anyhow!("JXL does not use palette settings"));
            }
        }
        args.push(OsString::from(format!(
            "--num_threads={}",
            job.threads.max(1)
        )));
        args.push(job.normalized_png.as_os_str().to_os_string());
        args.push(job.output_path.as_os_str().to_os_string());

        let start = Instant::now();
        run_command(path, &args, job.timeout, job.threads)?;
        artifact(job, "cjxl", start.elapsed(), Vec::new())
    }

    fn attach_webp_metadata(
        &self,
        output_path: &Path,
        metadata: &MetadataPayload,
        timeout: Duration,
    ) -> Result<()> {
        let Some(webpmux) = self.tools.webpmux.path.clone() else {
            anyhow::bail!("webpmux not found; cannot preserve WebP metadata safely");
        };

        let temp = tempfile::tempdir()?;
        let mut current = output_path.to_path_buf();
        let mut stage_index = 0_u8;

        if let Some(icc_profile) = &metadata.icc_profile {
            let icc_path = temp.path().join("profile.icc");
            fs::write(&icc_path, icc_profile)
                .with_context(|| format!("writing {}", icc_path.display()))?;
            stage_index += 1;
            let staged = temp.path().join(format!("stage-{stage_index}.webp"));
            run_command(
                &webpmux,
                &[
                    OsString::from("-set"),
                    OsString::from("icc"),
                    icc_path.as_os_str().to_os_string(),
                    current.as_os_str().to_os_string(),
                    OsString::from("-o"),
                    staged.as_os_str().to_os_string(),
                ],
                timeout,
                1,
            )?;
            current = staged;
        }

        if let Some(exif_metadata) = &metadata.exif_metadata {
            let exif_path = temp.path().join("metadata.exif");
            fs::write(&exif_path, exif_metadata)
                .with_context(|| format!("writing {}", exif_path.display()))?;
            stage_index += 1;
            let staged = temp.path().join(format!("stage-{stage_index}.webp"));
            run_command(
                &webpmux,
                &[
                    OsString::from("-set"),
                    OsString::from("exif"),
                    exif_path.as_os_str().to_os_string(),
                    current.as_os_str().to_os_string(),
                    OsString::from("-o"),
                    staged.as_os_str().to_os_string(),
                ],
                timeout,
                1,
            )?;
            current = staged;
        }

        if current != output_path {
            fs::copy(&current, output_path).with_context(|| {
                format!("copying WebP metadata output to {}", output_path.display())
            })?;
        }
        Ok(())
    }
}

pub fn normalized_png(
    image: &DynamicImage,
    metadata: &MetadataPayload,
) -> Result<(TempDir, PathBuf)> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("normalized-input.png");
    write_png(&path, image, metadata)?;
    Ok((dir, path))
}

fn write_png(path: &Path, image: &DynamicImage, metadata: &MetadataPayload) -> Result<()> {
    let file = fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = PngEncoder::new(file);
    apply_metadata(&mut encoder, metadata)?;
    image
        .write_with_encoder(encoder)
        .with_context(|| format!("writing PNG {}", path.display()))
}

fn apply_metadata<E: ImageEncoder>(encoder: &mut E, metadata: &MetadataPayload) -> Result<()> {
    if let Some(icc_profile) = &metadata.icc_profile {
        encoder
            .set_icc_profile(icc_profile.clone())
            .map_err(anyhow::Error::from)?;
    }
    if let Some(exif_metadata) = &metadata.exif_metadata {
        encoder
            .set_exif_metadata(exif_metadata.clone())
            .map_err(anyhow::Error::from)?;
    }
    Ok(())
}

fn artifact(
    job: EncodeJob<'_>,
    backend: &str,
    elapsed: Duration,
    warnings: Vec<String>,
) -> Result<EncodedArtifact> {
    let size_bytes = fs::metadata(job.output_path)
        .with_context(|| format!("reading metadata for {}", job.output_path.display()))?
        .len();
    Ok(EncodedArtifact {
        codec: job.codec,
        settings: job.settings,
        path: job.output_path.to_path_buf(),
        size_bytes,
        encode_ms: elapsed.as_millis() as u64,
        backend: backend.to_string(),
        warnings,
    })
}
