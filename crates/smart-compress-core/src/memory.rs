//! What an image costs in memory, and how much memory a run plans on using.

/// Memory an image holds from the moment it is decoded until its report is
/// finished, per source pixel: the decoded source, the RGBA copies the built-in
/// metric works from, the buffers of the probe and of the normalized PNG, and
/// the staging for the GPU.
///
/// Fitted to peak working sets summed over the process tree, measured with
/// `tools/measure-process-tree.rs` on the photos in `samples/photos` and a
/// 2048 px noise image, one image at a time with one and then seven candidates
/// at once. Peak memory is `FIXED + (FILE + n * CANDIDATE) * pixels`:
///
/// | megapixels | 1 candidate | 7 candidates | per added candidate |
/// |-----------:|------------:|-------------:|--------------------:|
/// |        3.8 |      772 MB |      1272 MB |               23 B/px |
/// |        4.2 |      830 MB |      1874 MB |               44 B/px |
/// |        6.0 |     1099 MB |      2651 MB |               45 B/px |
/// |       12.0 |     2010 MB |      5095 MB |               45 B/px |
///
/// The line through the one-candidate peaks has a slope of 159 B/px and about
/// 190 MB left over for the process itself, so the image alone holds 159 - 45 =
/// 114 B/px. Images over 16 MP are scored on a reduction, and the one measured
/// (16.7 MP) peaked at 1640 MB and 4370 MB, below the line; the line is kept
/// because it is the conservative one.
pub const FILE_BYTES_PER_PIXEL: u64 = 114;

/// Memory each candidate in flight holds per source pixel: the encoder
/// process, the decoded candidate, and the RGBA copies made to score it. The
/// slope of the table above; the 3.8 MP row, where few candidates peak at the
/// same time, is the one that falls under it.
pub const CANDIDATE_BYTES_PER_PIXEL: u64 = 45;

/// Memory one `ssimulacra2` process holds while it scores a candidate, per
/// pixel of the image it reads. It is a separate process, so it is not part of
/// the per-candidate line above, and it holds far more than the encoders do:
/// 12 MP photos peaked at 1.4 to 1.7 GB each (up to 150 B/px), and six at once,
/// which a run on six workers reaches when candidates finish encoding together,
/// made 9 GB of a 9.5 GB peak.
pub const EXTERNAL_METRIC_BYTES_PER_PIXEL: u64 = 150;

/// Memory assumed when the machine will not say how much it has.
const UNKNOWN_MEMORY_LIMIT: u64 = 8 << 30;

/// Physical memory of the machine in bytes.
#[cfg(windows)]
pub fn physical_memory() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_physical: u64,
        rest: [u64; 6],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx {
        length: size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_physical: 0,
        rest: [0; 6],
    };
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status.total_physical)
}

#[cfg(not(windows))]
pub fn physical_memory() -> Option<u64> {
    None
}

/// Memory a run may plan on using: two thirds of the physical memory, shared
/// by the images, the candidates and the external metric (see [`shares`]). The
/// total rather than what is free at the start, so the plan, and with it the
/// search, does not depend on what else the machine happens to be running.
pub fn default_memory_limit(physical: Option<u64>) -> u64 {
    physical.map_or(UNKNOWN_MEMORY_LIMIT, |total| total / 3 * 2)
}

/// How a memory limit is shared by the three kinds of work that hold memory.
/// The shares add up to the limit, so the budgets together never plan on more
/// than the limit allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shares {
    /// Images between their decode and their report.
    pub files: u64,
    /// Candidates being encoded, decoded and scored by the built-in metric.
    pub candidates: u64,
    /// `ssimulacra2` processes.
    pub external_metric: u64,
}

/// Three eighths for the images, three eighths for the candidates and a
/// quarter for the external metric. With the default limit of two thirds of the
/// physical memory that is a quarter, a quarter and a sixth of it.
///
/// The external metric needs at least 5.4 GB for three 12 MP photos at once
/// (3 * 12 MP * 150 B/px); a sixth of the 31 GiB of the machine these were
/// measured on is 5.5 GB, which still holds three. A quarter (8.3 GB) for the
/// images holds six 12 MP images (1.4 GB each) and a quarter for the candidates
/// holds fifteen of them (0.54 GB each), more than the workers of a run on most
/// machines. A split in thirds of a limit of half the physical memory held only
/// four 12 MP images at once instead of six, and a batch of six 12 MP photos on
/// six workers took 164 s instead of 139 s; these budgets are those of the
/// faster plan, whose peak working set was 5.6 GB, and they still add up to the
/// limit.
pub fn shares(limit: u64) -> Shares {
    let eighth = limit / 8;
    Shares {
        files: 3 * eighth,
        candidates: 3 * eighth,
        external_metric: limit - 6 * eighth,
    }
}

/// Memory an image is estimated to need on its own: itself, one candidate and,
/// with `ssimulacra2`, one external metric process of `metric_pixels` pixels.
pub fn single_image_bytes(pixels: u64, metric_pixels: Option<u64>) -> u64 {
    file_bytes(pixels)
        .saturating_add(candidate_bytes(pixels))
        .saturating_add(metric_pixels.map_or(0, external_metric_bytes))
}

/// Held by an image for as long as it is being processed.
pub fn file_bytes(pixels: u64) -> u64 {
    FILE_BYTES_PER_PIXEL.saturating_mul(pixels)
}

/// Held by each candidate of an image while it runs.
pub fn candidate_bytes(pixels: u64) -> u64 {
    CANDIDATE_BYTES_PER_PIXEL.saturating_mul(pixels)
}

/// Held by each external metric process while it runs.
pub fn external_metric_bytes(pixels: u64) -> u64 {
    EXTERNAL_METRIC_BYTES_PER_PIXEL.saturating_mul(pixels)
}

/// How many candidates of an image of `pixels` pixels fit in `limit` bytes,
/// between one and `workers`.
pub fn candidate_slots(limit: u64, pixels: u64, workers: usize) -> usize {
    usize::try_from(limit / candidate_bytes(pixels).max(1))
        .unwrap_or(usize::MAX)
        .clamp(1, workers.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MP: u64 = 1_000_000;

    /// Candidates per image, and the images that fit at once, for the share
    /// of the limit each kind of work gets.
    fn plan(ram: u64, pixels: u64, workers: usize) -> (usize, u64) {
        let shares = shares(default_memory_limit(Some(ram)));
        let slots = candidate_slots(shares.candidates, pixels, workers);
        (slots, (shares.files / file_bytes(pixels)).max(1))
    }

    #[test]
    fn the_shares_add_up_to_the_limit() {
        for limit in [0, 1, 2, 3, 8 * GIB, 16 * GIB + 1, u64::MAX] {
            let shares = shares(limit);
            assert_eq!(
                shares.files + shares.candidates + shares.external_metric,
                limit
            );
        }
        // Three 12 MP photos at once fit the external metric share of a 31 GB machine.
        let limit = default_memory_limit(Some(31 * GIB));
        assert!(shares(limit).external_metric >= 3 * external_metric_bytes(12 * MP));
    }

    #[test]
    fn a_lone_image_is_judged_against_the_whole_limit() {
        assert_eq!(single_image_bytes(MP, None), 159 * MP);
        assert_eq!(single_image_bytes(MP, Some(MP)), 309 * MP);
        assert!(single_image_bytes(200 * MP, Some(16 * MP)) > default_memory_limit(Some(31 * GIB)));
    }

    #[test]
    fn the_limit_is_two_thirds_of_the_physical_memory() {
        assert_eq!(default_memory_limit(Some(30 * GIB)), 20 * GIB);
        assert_eq!(default_memory_limit(Some(3)), 2);
        assert_eq!(default_memory_limit(None), 8 * GIB);
    }

    #[test]
    fn the_default_budgets_are_a_quarter_a_quarter_and_a_sixth_of_the_memory() {
        let physical = 48 * GIB;
        let shares = shares(default_memory_limit(Some(physical)));
        assert_eq!(shares.files, physical / 4);
        assert_eq!(shares.candidates, physical / 4);
        assert_eq!(shares.external_metric, physical / 6);
    }

    #[test]
    fn candidates_and_images_in_flight_follow_the_image_size_and_the_memory() {
        // (RAM, megapixels, 16 workers) -> (candidates per image, images at once)
        for (ram, megapixels, expected) in [
            (16, 2, (16, 18)),
            (16, 12, (7, 3)),
            (16, 50, (1, 1)),
            (16, 200, (1, 1)),
            (32, 2, (16, 37)),
            (32, 12, (15, 6)),
            (32, 50, (3, 1)),
            (32, 200, (1, 1)),
        ] {
            assert_eq!(
                plan(ram * GIB, megapixels * MP, 16),
                (expected.0, expected.1),
                "{ram} GiB, {megapixels} MP"
            );
        }
    }

    #[test]
    fn an_explicit_worker_count_caps_the_candidates_not_the_memory() {
        assert_eq!(plan(32 * GIB, 2 * MP, 4).0, 4);
        assert_eq!(plan(32 * GIB, 2 * MP, 1).0, 1);
    }

    #[test]
    fn an_image_too_large_for_the_limit_still_gets_one_slot() {
        assert_eq!(candidate_slots(GIB, 200 * MP, 16), 1);
        assert_eq!(candidate_slots(0, 1, 16), 1);
        assert_eq!(candidate_slots(u64::MAX, 0, 16), 16);
    }
}
