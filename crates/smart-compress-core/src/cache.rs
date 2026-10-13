use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{Duration, SystemTime},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use smart_compress_codecs::{Codec, EncodedArtifact, EncoderSettings, ToolInventory};
use smart_compress_metrics::QualityScore;

use crate::{
    atomic_file::{atomic_copy, atomic_write},
    candidate::ActualScoreBackend,
    config::CacheConfig,
};

/// Written when a cache directory is created. Pruning and storing refuse a
/// non-empty directory without it, so a wrong `--cache-dir` is left alone.
const MARKER_NAME: &str = ".smart-compress-cache-marker";
const MARKER_TEXT: &str = "smart-compress candidate cache\n";
const HASH_EXTENSION: &str = "b3";

#[derive(Debug, Clone)]
pub struct CandidateCache {
    dir: PathBuf,
    tool_signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedCandidateMetadata {
    pub size_bytes: u64,
    /// BLAKE3 of the candidate these scores belong to.
    pub artifact_blake3: String,
    pub decode_ms: Option<u64>,
    pub decode_process_ms: Option<u64>,
    pub decode_io_ms: Option<u64>,
    pub preprocess_ms: Option<u64>,
    pub score_ms: Option<u64>,
    pub quality: QualityScore,
    pub encoded_backend: Option<String>,
    pub score_backend: Option<ActualScoreBackend>,
}

impl CandidateCache {
    pub fn from_config(config: &CacheConfig, tools: &ToolInventory) -> Option<Self> {
        config.enabled.then(|| {
            let dir = config.dir.clone().unwrap_or_else(default_cache_dir);
            Self {
                dir,
                tool_signature: serde_json::to_vec(tools).unwrap_or_default(),
            }
        })
    }

    pub fn prune(&self, max_bytes: Option<u64>, max_age_days: Option<u64>) -> Result<usize> {
        self.check_prunable()?;
        let now = SystemTime::now();
        let max_age = max_age_days.map(|days| Duration::from_secs(days.saturating_mul(86_400)));
        let mut files = Vec::new();
        collect_files(&self.dir, &mut files)?;
        let mut removed = 0;
        for path in files.iter().filter(|path| {
            max_age.is_some_and(|age| {
                fs::metadata(path)
                    .and_then(|metadata| metadata.modified())
                    .ok()
                    .and_then(|modified| now.duration_since(modified).ok())
                    .is_some_and(|elapsed| elapsed > age)
            })
        }) {
            if fs::remove_file(path).is_ok() {
                removed += 1;
            }
        }

        let Some(max_bytes) = max_bytes else {
            return Ok(removed);
        };
        let mut remaining = Vec::new();
        collect_files(&self.dir, &mut remaining)?;
        let mut total = remaining
            .iter()
            .filter_map(|path| fs::metadata(path).ok().map(|metadata| metadata.len()))
            .sum::<u64>();
        remaining.sort_by_key(|path| {
            fs::metadata(path)
                .and_then(|metadata| metadata.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH)
        });
        for path in remaining {
            if total <= max_bytes {
                break;
            }
            let size = fs::metadata(&path).map_or(0, |metadata| metadata.len());
            if fs::remove_file(path).is_ok() {
                total = total.saturating_sub(size);
                removed += 1;
            }
        }
        Ok(removed)
    }

    pub fn source_hash(&self, path: &Path) -> Result<String> {
        let mut file = fs::File::open(path)
            .with_context(|| format!("opening cache source {}", path.display()))?;
        let mut hasher = blake3::Hasher::new();
        let mut buffer = [0_u8; 128 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .with_context(|| format!("hashing cache source {}", path.display()))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(hasher.finalize().to_hex().to_string())
    }

    pub fn load(
        &self,
        source_hash: &str,
        codec: Codec,
        settings: &EncoderSettings,
        has_alpha: bool,
        output_path: &Path,
    ) -> Result<Option<EncodedArtifact>> {
        let cache_path = self.cache_path(source_hash, codec, settings, has_alpha);
        if !cache_path.is_file() {
            return Ok(None);
        }
        // Entries from before artifacts were hashed have no hash file: misses.
        let Ok(expected_hash) = fs::read_to_string(cache_path.with_extension(HASH_EXTENSION))
        else {
            return Ok(None);
        };
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating candidate directory {}", parent.display()))?;
        }
        atomic_copy(&cache_path, output_path, false).with_context(|| {
            format!(
                "copying cached candidate {} to {}",
                cache_path.display(),
                output_path.display()
            )
        })?;
        // The copy is hashed, not the cache file, so what was checked is what is used.
        if file_hash(output_path)? != expected_hash.trim() {
            let _ = fs::remove_file(output_path);
            anyhow::bail!(
                "cached candidate {} does not match its hash; encoding it again",
                cache_path.file_name().unwrap_or_default().to_string_lossy()
            );
        }
        let size_bytes = fs::metadata(output_path)
            .with_context(|| format!("reading metadata for {}", output_path.display()))?
            .len();
        if size_bytes == 0 {
            anyhow::bail!("cached candidate {} is empty", cache_path.display());
        }
        Ok(Some(EncodedArtifact {
            codec,
            settings: settings.clone(),
            path: output_path.to_path_buf(),
            size_bytes,
            encode_ms: 0,
            backend: "cache".to_string(),
            warnings: vec!["cache hit".to_string()],
        }))
    }

    pub fn store(
        &self,
        source_hash: &str,
        artifact: &EncodedArtifact,
        has_alpha: bool,
    ) -> Result<()> {
        self.ensure_marker()?;
        let cache_path =
            self.cache_path(source_hash, artifact.codec, &artifact.settings, has_alpha);
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating cache directory {}", parent.display()))?;
        }
        let copied = atomic_copy(&artifact.path, &cache_path, false).with_context(|| {
            format!(
                "copying candidate {} to cache {}",
                artifact.path.display(),
                cache_path.display()
            )
        })?;
        anyhow::ensure!(
            copied == artifact.size_bytes,
            "cache candidate size changed while publishing: copied {copied}, expected {}",
            artifact.size_bytes
        );
        let hash = file_hash(&cache_path)?;
        anyhow::ensure!(
            file_hash(&artifact.path)? == hash,
            "cache candidate hash changed while publishing"
        );
        atomic_write(&cache_path.with_extension(HASH_EXTENSION), hash.as_bytes())
            .context("writing cache candidate hash")
    }

    pub fn load_metadata(
        &self,
        source_hash: &str,
        codec: Codec,
        settings: &EncoderSettings,
        has_alpha: bool,
        scoring_signature: &str,
        artifact: &Path,
    ) -> Result<Option<CachedCandidateMetadata>> {
        let metadata_path =
            self.metadata_path(source_hash, codec, settings, has_alpha, scoring_signature);
        if !metadata_path.is_file() {
            return Ok(None);
        }
        let text = fs::read_to_string(&metadata_path)
            .with_context(|| format!("reading cache metadata {}", metadata_path.display()))?;
        let metadata: CachedCandidateMetadata = serde_json::from_str(&text)
            .with_context(|| format!("parsing cache metadata {}", metadata_path.display()))?;
        let size_bytes = fs::metadata(artifact)
            .with_context(|| format!("reading metadata for {}", artifact.display()))?
            .len();
        if metadata.size_bytes != size_bytes || metadata.artifact_blake3 != file_hash(artifact)? {
            return Ok(None);
        }
        Ok(Some(metadata))
    }

    pub fn store_metadata(
        &self,
        source_hash: &str,
        codec: Codec,
        settings: &EncoderSettings,
        has_alpha: bool,
        scoring_signature: &str,
        metadata: &CachedCandidateMetadata,
    ) -> Result<()> {
        self.ensure_marker()?;
        let metadata_path =
            self.metadata_path(source_hash, codec, settings, has_alpha, scoring_signature);
        if let Some(parent) = metadata_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("creating cache metadata directory {}", parent.display())
            })?;
        }
        let text = serde_json::to_string_pretty(metadata).context("serializing cache metadata")?;
        atomic_write(&metadata_path, text.as_bytes())
            .with_context(|| format!("writing cache metadata {}", metadata_path.display()))
    }

    /// A directory is a cache when it holds the marker or nothing at all.
    fn check_prunable(&self) -> Result<()> {
        if self.has_marker() || !self.dir.exists() || is_empty_dir(&self.dir)? {
            return Ok(());
        }
        anyhow::bail!(
            "{} is not a smart-compress cache (it has no {MARKER_NAME} file); refusing to delete from it",
            self.dir.display()
        )
    }

    fn has_marker(&self) -> bool {
        fs::read_to_string(self.dir.join(MARKER_NAME)).is_ok_and(|text| text == MARKER_TEXT)
    }

    /// Marks a new or empty directory as a cache; any other is refused.
    fn ensure_marker(&self) -> Result<()> {
        // Writing the marker puts a temporary file in the directory first, so a
        // second worker checking meanwhile would see a non-empty directory with
        // no marker and refuse it. One process marks one directory at a time.
        // The lock is process-wide: two processes starting on the same new
        // directory can still collide, and the loser only skips that one store.
        static MARKING: Mutex<()> = Mutex::new(());
        let _marking = MARKING
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.has_marker() {
            return Ok(());
        }
        fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating cache directory {}", self.dir.display()))?;
        // Another worker may have marked it and added entries meanwhile.
        if !is_empty_dir(&self.dir)? && !self.has_marker() {
            anyhow::bail!(
                "{} is not empty and is not a smart-compress cache; use an empty folder",
                self.dir.display()
            );
        }
        atomic_write(&self.dir.join(MARKER_NAME), MARKER_TEXT.as_bytes())
            .context("marking the cache directory")
    }

    fn cache_path(
        &self,
        source_hash: &str,
        codec: Codec,
        settings: &EncoderSettings,
        has_alpha: bool,
    ) -> PathBuf {
        let key = self.cache_key(source_hash, codec, settings, has_alpha);
        self.dir
            .join(&key[..2])
            .join(format!("{key}.{}", codec.extension()))
    }

    fn metadata_path(
        &self,
        source_hash: &str,
        codec: Codec,
        settings: &EncoderSettings,
        has_alpha: bool,
        scoring_signature: &str,
    ) -> PathBuf {
        let candidate_key = self.cache_key(source_hash, codec, settings, has_alpha);
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"smart-compress-quality-cache-v1");
        hasher.update(candidate_key.as_bytes());
        hasher.update(scoring_signature.as_bytes());
        let key = hasher.finalize().to_hex().to_string();
        self.dir.join(&key[..2]).join(format!("{key}.json"))
    }

    fn cache_key(
        &self,
        source_hash: &str,
        codec: Codec,
        settings: &EncoderSettings,
        has_alpha: bool,
    ) -> String {
        let mut hasher = blake3::Hasher::new();
        // v2: WebP no longer caches a grey candidate whose profile it dropped.
        // v3: every entry has a BLAKE3 hash that is checked when it is loaded.
        hasher.update(b"smart-compress-candidate-cache-v3");
        hasher.update(source_hash.as_bytes());
        hasher.update(format!("{codec:?}").as_bytes());
        hasher.update(settings.label().as_bytes());
        hasher.update(if has_alpha { b"alpha=1" } else { b"alpha=0" });
        hasher.update(&self.tool_signature);
        hasher.finalize().to_hex().to_string()
    }
}

/// A per-user folder, never the working directory: whoever controls that
/// folder would control what the cache hands back.
pub fn default_cache_dir() -> PathBuf {
    let absolute = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
    };
    #[cfg(windows)]
    let dir = absolute("LOCALAPPDATA").map(|base| base.join("smart-compress").join("cache"));
    #[cfg(not(windows))]
    let dir = absolute("XDG_CACHE_HOME")
        .or_else(|| absolute("HOME").map(|home| home.join(".cache")))
        .map(|base| base.join("smart-compress"));
    dir.unwrap_or_else(|| std::env::temp_dir().join("smart-compress-cache"))
}

fn is_empty_dir(dir: &Path) -> Result<bool> {
    Ok(fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .next()
        .is_none())
}

fn is_hex(name: &str, length: usize) -> bool {
    name.len() == length
        && name
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// `<64 hex>.<extension>` of a codec, of the scores, or of the hash file.
fn is_entry_name(name: &str) -> bool {
    name.split_once('.').is_some_and(|(stem, extension)| {
        is_hex(stem, 64)
            && (matches!(extension, "json" | HASH_EXTENSION)
                || [
                    Codec::Png,
                    Codec::Jpeg,
                    Codec::Webp,
                    Codec::Avif,
                    Codec::Jxl,
                ]
                .iter()
                .any(|codec| codec.extension() == extension))
    })
}

/// The files of the cache layout (`<2 hex>/<entry>`) and nothing else.
fn collect_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    for shard in fs::read_dir(root).with_context(|| format!("reading cache {}", root.display()))? {
        let shard = shard.with_context(|| format!("reading cache entry {}", root.display()))?;
        let is_shard = shard.file_type().is_ok_and(|kind| kind.is_dir())
            && is_hex(&shard.file_name().to_string_lossy(), 2);
        if !is_shard {
            continue;
        }
        for entry in fs::read_dir(shard.path())
            .with_context(|| format!("reading cache {}", shard.path().display()))?
        {
            let entry =
                entry.with_context(|| format!("reading cache {}", shard.path().display()))?;
            if entry.file_type().is_ok_and(|kind| kind.is_file())
                && is_entry_name(&entry.file_name().to_string_lossy())
            {
                files.push(entry.path());
            }
        }
    }
    Ok(())
}

pub(crate) fn file_hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("opening cache artifact {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hashing cache artifact {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cache(dir: &Path) -> CandidateCache {
        CandidateCache {
            dir: dir.to_path_buf(),
            tool_signature: Vec::new(),
        }
    }

    fn entry(dir: &Path, name: char, extension: &str) -> PathBuf {
        let shard = dir.join("ab");
        fs::create_dir_all(&shard).unwrap();
        shard.join(format!("{}.{extension}", name.to_string().repeat(64)))
    }

    #[test]
    fn cache_prune_enforces_byte_budget() {
        let dir = tempfile::tempdir().unwrap();
        let cache = test_cache(dir.path());
        cache.ensure_marker().unwrap();
        fs::write(entry(dir.path(), '1', "webp"), vec![1_u8; 8]).unwrap();
        fs::write(entry(dir.path(), '2', "webp"), vec![2_u8; 8]).unwrap();

        let removed = cache.prune(Some(8), None).unwrap();

        assert_eq!(removed, 1);
        assert_eq!(fs::read_dir(dir.path().join("ab")).unwrap().count(), 1);
    }

    #[test]
    fn prune_refuses_a_directory_that_is_not_a_cache() {
        let dir = tempfile::tempdir().unwrap();
        let photo = dir.path().join("holiday.jpg");
        fs::write(&photo, b"precious").unwrap();

        let error = test_cache(dir.path()).prune(Some(0), Some(0)).unwrap_err();

        assert!(error.to_string().contains("not a smart-compress cache"));
        assert!(photo.exists());
        // Storing refuses it too, and does not mark it.
        assert!(test_cache(dir.path()).ensure_marker().is_err());
        assert!(!dir.path().join(MARKER_NAME).exists());
    }

    #[test]
    fn workers_marking_a_new_directory_at_once_all_succeed() {
        for _ in 0..50 {
            let dir = tempfile::tempdir().unwrap();
            let cache_dir = dir.path().join("cache");
            std::thread::scope(|scope| {
                let workers: Vec<_> = (0..8)
                    .map(|_| scope.spawn(|| test_cache(&cache_dir).ensure_marker()))
                    .collect();
                for worker in workers {
                    worker.join().unwrap().unwrap();
                }
            });
        }
    }

    #[test]
    fn prune_only_deletes_files_named_like_cache_entries() {
        let dir = tempfile::tempdir().unwrap();
        let cache = test_cache(dir.path());
        cache.ensure_marker().unwrap();
        let ours = entry(dir.path(), '3', "avif");
        let foreign = [
            dir.path().join("ab").join("notes.txt"),
            dir.path().join("readme.json"),
            dir.path()
                .join("zz")
                .join(format!("{}.avif", "4".repeat(64))),
        ];
        fs::write(&ours, b"x").unwrap();
        fs::create_dir_all(dir.path().join("zz")).unwrap();
        for path in &foreign {
            fs::write(path, b"x").unwrap();
        }

        assert_eq!(cache.prune(Some(0), Some(0)).unwrap(), 1);

        assert!(!ours.exists());
        assert!(foreign.iter().all(|path| path.exists()));
        assert!(dir.path().join(MARKER_NAME).exists());
    }

    fn stored_artifact(dir: &Path) -> (CandidateCache, EncodedArtifact, PathBuf) {
        let cache = test_cache(&dir.join("cache"));
        let path = dir.join("candidate.webp");
        fs::write(&path, b"encoded bytes").unwrap();
        let artifact = EncodedArtifact {
            codec: Codec::Webp,
            settings: EncoderSettings::Lossless,
            path: path.clone(),
            size_bytes: 13,
            encode_ms: 0,
            backend: "test".to_string(),
            warnings: Vec::new(),
        };
        cache.store("source", &artifact, false).unwrap();
        (cache, artifact, dir.join("loaded.webp"))
    }

    #[test]
    fn a_cached_artifact_is_checked_against_its_hash() {
        let dir = tempfile::tempdir().unwrap();
        let (cache, artifact, loaded) = stored_artifact(dir.path());
        let load = || cache.load("source", Codec::Webp, &artifact.settings, false, &loaded);
        assert_eq!(load().unwrap().unwrap().size_bytes, 13);

        // Same name, different bytes: refused and not left behind.
        fs::write(
            cache.cache_path("source", Codec::Webp, &artifact.settings, false),
            b"forged bytes!",
        )
        .unwrap();
        assert!(load().unwrap_err().to_string().contains("hash"));
        assert!(!loaded.exists());

        // An entry without a hash file, as older versions wrote, is a miss.
        fs::remove_file(
            cache
                .cache_path("source", Codec::Webp, &artifact.settings, false)
                .with_extension(HASH_EXTENSION),
        )
        .unwrap();
        assert!(load().unwrap().is_none());
    }

    #[test]
    fn cached_scores_belong_to_one_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let (cache, artifact, _) = stored_artifact(dir.path());
        let metadata = CachedCandidateMetadata {
            size_bytes: 13,
            artifact_blake3: file_hash(&artifact.path).unwrap(),
            decode_ms: None,
            decode_process_ms: None,
            decode_io_ms: None,
            preprocess_ms: None,
            score_ms: None,
            quality: QualityScore::lossless(),
            encoded_backend: None,
            score_backend: None,
        };
        let settings = &artifact.settings;
        cache
            .store_metadata("source", Codec::Webp, settings, false, "sig", &metadata)
            .unwrap();
        let load = |path: &Path| {
            cache
                .load_metadata("source", Codec::Webp, settings, false, "sig", path)
                .unwrap()
        };
        assert!(load(&artifact.path).is_some());

        // The same size with other bytes does not inherit the score.
        let other = dir.path().join("other.webp");
        fs::write(&other, b"other bytes!!").unwrap();
        assert!(load(&other).is_none());
    }

    #[test]
    fn the_default_cache_is_outside_the_working_directory() {
        let dir = default_cache_dir();
        assert!(dir.is_absolute());
        assert!(!dir.starts_with(std::env::current_dir().unwrap()));
    }
}
