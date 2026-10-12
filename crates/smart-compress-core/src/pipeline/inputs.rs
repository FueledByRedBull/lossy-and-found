use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use ignore::WalkBuilder;

use super::{DEFAULT_OUTPUT_DIR, Optimizer, codec_name};

pub(super) fn input_identity(path: &Path) -> Option<String> {
    let mut file = fs::File::open(path).ok()?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Some(hasher.finalize().to_hex().to_string())
}

impl Optimizer {
    /// `--out` names the one file to write, so exactly one image may be found,
    /// and its format must be among those allowed.
    pub(super) fn check_out_target(&self, inputs: usize) -> Result<()> {
        let Some(out) = &self.config.out else {
            return Ok(());
        };
        anyhow::ensure!(
            inputs == 1,
            "--out names one file, but {inputs} input images were found; use --out-dir for several"
        );
        if let Some(codec) = self.out_codec {
            anyhow::ensure!(
                self.config.formats.contains(codec),
                "--out {} names the {} format, which is not among the selected formats",
                out.display(),
                codec_name(codec)
            );
        }
        Ok(())
    }

    /// The images to process, each once, and what went wrong while looking
    /// for them. Every file counts, whatever its folder is called or an ignore
    /// file says; the output folder of this run does not.
    pub(super) fn discover_inputs(&self) -> Result<(Vec<PathBuf>, Vec<String>)> {
        let mut files = Vec::new();
        let mut warnings = Vec::new();
        let mut seen = HashSet::new();
        let output_dir = self
            .config
            .out_dir
            .as_deref()
            .and_then(|dir| fs::canonicalize(dir).ok());
        let mut add = |path: &Path| {
            if seen.insert(fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())) {
                files.push(path.to_path_buf());
            }
        };
        // Outputs of this run, and of earlier ones, are not found by a walk:
        // those under the output folder, or under a default one below the
        // folder walked. A file named on the command line is always an input.
        let is_output = |path: &Path, root: &Path| {
            if self.config.out_dir.is_some() {
                output_dir.as_ref().is_some_and(|dir| {
                    fs::canonicalize(path).is_ok_and(|path| path.starts_with(dir))
                })
            } else {
                path.strip_prefix(root)
                    .ok()
                    .and_then(Path::parent)
                    .is_some_and(|inside| {
                        inside
                            .components()
                            .any(|part| part.as_os_str().eq_ignore_ascii_case(DEFAULT_OUTPUT_DIR))
                    })
            }
        };
        for input in &self.config.inputs {
            if input.is_file() {
                add(input);
            } else if input.is_dir() {
                if !self.config.recursive {
                    anyhow::bail!(
                        "{} is a directory; pass --recursive to process it",
                        input.display()
                    );
                }
                let walker = WalkBuilder::new(input)
                    .follow_links(false)
                    .standard_filters(false)
                    .build();
                for entry in walker {
                    match entry {
                        Ok(entry)
                            if entry.file_type().is_some_and(|kind| kind.is_file())
                                && is_supported_image(entry.path())
                                && !is_output(entry.path(), input) =>
                        {
                            add(entry.path());
                        }
                        Ok(_) => {}
                        Err(error) => warnings.push(format!(
                            "could not read part of {}: {error}",
                            input.display()
                        )),
                    }
                }
            } else {
                anyhow::bail!("input does not exist: {}", input.display());
            }
        }

        if files.is_empty() {
            anyhow::bail!("no input images found");
        }
        Ok((files, warnings))
    }

    /// Inputs whose output directory and file stem coincide would publish to
    /// the same file whenever their winners share a codec.
    pub(super) fn colliding_inputs(&self, inputs: &[PathBuf]) -> HashSet<PathBuf> {
        let mut by_target: HashMap<(PathBuf, String), Vec<&PathBuf>> = HashMap::new();
        for input in inputs {
            let stem = input
                .file_stem()
                .map(|stem| stem.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            by_target
                .entry((self.output_dir(input), stem))
                .or_default()
                .push(input);
        }
        by_target
            .into_values()
            .filter(|group| group.len() > 1)
            .flatten()
            .cloned()
            .collect()
    }
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "webp" | "avif"
            )
        })
        .unwrap_or(false)
}
