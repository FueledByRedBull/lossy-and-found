use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use anyhow::{Result, anyhow};
use smart_compress_codecs::{Budget, Codec, EncoderSettings};

use crate::progress::ProgressObserver;

pub(super) fn directory_bytes(path: &Path) -> u64 {
    let mut total = 0_u64;
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if let Ok(metadata) = entry.metadata() {
            if metadata.is_dir() {
                total = total.saturating_add(directory_bytes(&path));
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    total
}

pub(super) struct TempUsageGuard {
    usage: Arc<AtomicU64>,
    /// Woken when usage drops, for candidates waiting for room.
    gate: Arc<Budget>,
    current: u64,
}

impl TempUsageGuard {
    pub(super) fn new(usage: Arc<AtomicU64>, gate: Arc<Budget>) -> Self {
        Self {
            usage,
            gate,
            current: 0,
        }
    }

    pub(super) fn set(&mut self, value: u64, limit: Option<u64>) -> Result<()> {
        if value > self.current {
            let delta = value - self.current;
            let total = self
                .usage
                .fetch_add(delta, Ordering::SeqCst)
                .saturating_add(delta);
            if let Some(limit) = limit
                && total > limit
            {
                self.usage.fetch_sub(delta, Ordering::SeqCst);
                return Err(anyhow!(
                    "run temporary storage budget of {limit} bytes exceeded ({total} bytes)"
                ));
            }
        } else {
            self.usage.fetch_sub(self.current - value, Ordering::SeqCst);
            self.gate.wake();
        }
        self.current = value;
        Ok(())
    }
}

impl Drop for TempUsageGuard {
    fn drop(&mut self) {
        self.usage.fetch_sub(self.current, Ordering::SeqCst);
        self.gate.wake();
    }
}

pub(super) struct ActiveCandidate<'a>(pub(super) &'a AtomicUsize, pub(super) &'a Budget);

impl Drop for ActiveCandidate<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
        self.1.wake();
    }
}

pub(super) struct FileProgressGuard {
    progress: Arc<dyn ProgressObserver>,
    path: PathBuf,
}

impl FileProgressGuard {
    pub(super) fn new(progress: Arc<dyn ProgressObserver>, path: &Path) -> Self {
        progress.file_started(path);
        Self {
            progress,
            path: path.to_path_buf(),
        }
    }
}

impl Drop for FileProgressGuard {
    fn drop(&mut self) {
        self.progress.file_finished(&self.path);
    }
}

pub(super) struct CandidateProgressGuard {
    progress: Arc<dyn ProgressObserver>,
    path: PathBuf,
    index: usize,
    total: usize,
}

impl CandidateProgressGuard {
    pub(super) fn new(
        progress: Arc<dyn ProgressObserver>,
        path: &Path,
        index: usize,
        total: usize,
        codec: Codec,
        settings: &EncoderSettings,
    ) -> Self {
        progress.candidate_started(path, index, total, codec, settings);
        Self {
            progress,
            path: path.to_path_buf(),
            index,
            total,
        }
    }
}

impl Drop for CandidateProgressGuard {
    fn drop(&mut self) {
        self.progress
            .candidate_finished(&self.path, self.index, self.total);
    }
}
