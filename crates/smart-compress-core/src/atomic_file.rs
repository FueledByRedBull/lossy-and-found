use std::{
    fs,
    io::{self, Write},
    path::Path,
};

use anyhow::{Context, Result};
use tempfile::NamedTempFile;

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("creating directory {}", parent.display()))?;
    let mut temporary = NamedTempFile::new_in(parent)
        .with_context(|| format!("creating temporary file in {}", parent.display()))?;
    temporary
        .write_all(bytes)
        .with_context(|| format!("writing temporary file for {}", path.display()))?;
    temporary
        .as_file()
        .sync_all()
        .with_context(|| format!("syncing temporary file for {}", path.display()))?;
    persist(temporary, path, false)
}

pub fn atomic_copy(source: &Path, destination: &Path, no_overwrite: bool) -> Result<u64> {
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("creating directory {}", parent.display()))?;
    let mut input =
        fs::File::open(source).with_context(|| format!("opening source {}", source.display()))?;
    let mut temporary = NamedTempFile::new_in(parent)
        .with_context(|| format!("creating temporary file in {}", parent.display()))?;
    let copied = io::copy(&mut input, &mut temporary)
        .with_context(|| format!("copying {} to temporary file", source.display()))?;
    temporary
        .as_file()
        .sync_all()
        .with_context(|| format!("syncing temporary file for {}", destination.display()))?;
    persist(temporary, destination, no_overwrite)?;
    Ok(copied)
}

fn persist(temporary: NamedTempFile, destination: &Path, no_overwrite: bool) -> Result<()> {
    let result = if no_overwrite {
        temporary.persist_noclobber(destination)
    } else {
        temporary.persist(destination)
    };
    result
        .map(|_| ())
        .map_err(|error| error.error)
        .with_context(|| format!("atomically publishing {}", destination.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_replaces_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.json");
        atomic_write(&path, b"first").unwrap();
        atomic_write(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
    }

    #[test]
    fn atomic_copy_publishes_complete_content() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.bin");
        let destination = dir.path().join("destination.bin");
        fs::write(&source, b"candidate").unwrap();

        let copied = atomic_copy(&source, &destination, true).unwrap();

        assert_eq!(copied, 9);
        assert_eq!(fs::read(destination).unwrap(), b"candidate");
    }
}
