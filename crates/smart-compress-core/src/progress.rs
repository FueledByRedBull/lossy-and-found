use std::path::Path;

use smart_compress_codecs::{Codec, EncoderSettings};

pub trait ProgressObserver: Send + Sync {
    fn batch_started(&self, _total_files: usize) {}
    fn batch_finished(&self) {}
    fn file_started(&self, _path: &Path) {}
    fn file_finished(&self, _path: &Path) {}
    fn candidates_planned(&self, _path: &Path, _total_candidates: usize) {}
    fn candidate_started(
        &self,
        _path: &Path,
        _index: usize,
        _total: usize,
        _codec: Codec,
        _settings: &EncoderSettings,
    ) {
    }
    fn candidate_finished(&self, _path: &Path, _index: usize, _total: usize) {}
    /// The search for `path` is over. The candidates planned but skipped,
    /// assumed, or never started because the search stopped will not finish, so
    /// the file's total is the number that did.
    fn candidates_settled(&self, _path: &Path) {}
}

#[derive(Debug, Default)]
pub struct NoopProgress;

impl ProgressObserver for NoopProgress {}
