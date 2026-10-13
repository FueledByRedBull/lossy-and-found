# Benchmarking

`smart-compress-benchmark` measures the real optimizer pipeline, including source decoding, normalization, candidate encoding and decoding, scoring, cache activity, search and output publication. This executable is the source of end-to-end numbers.

All measured results below were taken on one Windows 11 machine with 16 logical cores, 31 GB of RAM and one discrete GPU, in October 2026, so they show relative changes and orders of magnitude, not what another machine will do. Reproduction commands are in [Reproducing the published numbers](#reproducing-the-published-numbers). Run the harness from the workspace root in release mode, with the encoders installed by `tools/fetch-tools.ps1`:

```powershell
cargo run --release --bin smart-compress-benchmark -- tests/fixtures --formats png,jpeg,webp --runs 3 --cache-dir target/benchmark-cache --out target/benchmark-results.json
```

The checked-in `tests/fixtures` corpus is the reproducible baseline corpus. It contains photo-like, screenshot/text, flat-icon, gradient, noisy, transparency, WebP, GIF, TIFF, and BMP cases. The benchmark report stores each input's byte length and BLAKE3 hash, tool inventory, operating system, architecture, selected mode, cache state, stage totals, per-image reports, and wall time.

Modes are recorded separately:

- `cpu`: GPU disabled.
- `gpu-metrics`: GPU metric scoring with CPU preprocessing.
- `gpu-prepared`: GPU metric scoring with the experimental prepared-original path (`--prepared-gpu`).
- `gpu-verify`: GPU scoring plus the CPU reference pass (`--verify-gpu`).

Use separate runs for cold and warm cache. With `--cache-dir`, the harness creates a mode-specific cache and labels run zero `cold` and later runs `warm`. Use `--runs 3` or more before comparing medians. Use `--formats png` (or one codec at a time) to isolate codec families; omit it for the normal mixed search. `--speed fast`, `balanced`, and `exhaustive` measure the search presets.

Acceptance gates for promotion decisions are deliberately explicit rather than inferred from isolated GPU microbenchmarks:

- compare at least three repeated runs per mode and report median and p95 wall time;
- require at least a 10% median end-to-end wall-time improvement for a default-routing change;
- reject a promotion with a material p95 regression, a winner-decision mismatch, a quality-floor crossing, or a resource-budget violation;
- retain the report and input hashes with the experiment.

The harness does not turn noisy external codec timings into hard unit-test failures.

## Budget classes

These are measurement gates for the machine described above, not claims that every codec/tool combination already meets them:

| Class | Pixels | Single-image wall-time gate | Peak decoded RAM | Temporary disk |
|---|---:|---:|---:|---:|
| Tiny | up to 262k | 1 s | 256 MiB | 64 MiB |
| Normal | up to 4 MP | 10 s | 1 GiB | 512 MiB |
| 12 MP | up to 12 MP | 30 s | 2 GiB | 1 GiB |
| 50 MP | up to 50 MP | 90 s | 4 GiB | 4 GiB |
| 200 MP | approximately 200 MP | 900 s | 8 GiB | 16 GiB |

The command line can enforce the per-image wall-time, candidate-attempt, temporary-byte, decoded-byte, and pixel limits directly. CPU RAM and VRAM are reported as measured or estimated fields; a benchmark is not considered a pass when those budgets are exceeded even if wall time improves.

For batch smoke gates, use at least 30 tiny images and 10 normal images: the provisional targets are 10 tiny images/s and 0.5 normal images/s on that machine, measured with the same codec/search configuration and a warm process/tool cache. No retained baseline report backs these targets yet.

## Synthetic corpus

`cargo run --release --bin smart-compress-benchmark -- --generate-corpus target/corpus` writes a deterministic synthetic corpus and a `corpus.json` manifest. Classes: photo-like, screenshot-like, UI text, flat icon, gradient, noise, soft-alpha transparency, alpha fringe, 1 px checkerboard, one-pixel lines, small text, a JPEG with EXIF orientation 6, PNGs with embedded sRGB and Display P3 ICC profiles, and a CMYK JPEG. Every image is procedurally generated; none is a real photograph or screenshot, so use real photos as well (`samples/photos/` has ten) before drawing conclusions about photographic content. The default size classes are `tiny` (16 to 64 px), `512` and `2048`. `--corpus-sizes tiny,512,2048,12mp,50mp,199mp` adds the large classes (photo and screenshot only; the 199.8 MP PNG needs about 1 GB of RAM to generate). The manifest records each file's path, content class, size class, dimensions, format, byte length, BLAKE3 hash and provenance (generator, generator version, seed). Equal options always produce byte-identical files; `--corpus-seed` changes the content. Benchmark it with `cargo run --release --bin smart-compress-benchmark -- target/corpus --runs 3 --out target/benchmark-results.json`; directory inputs are searched recursively.

## Harness options

Run `smart-compress-benchmark --help` for the full list. Inputs are files or folders (searched recursively); the output file is `benchmark-results.json` unless `--out` says otherwise.

- `--runs N` repeats the whole run (default 1).
- `--modes cpu,gpu-metrics,gpu-prepared,gpu-verify` selects the execution modes (repetitions alternate between them); it overrides `--prepared-gpu` and `--verify-gpu`.
- `--metric auto|ssimulacra2|builtin` is the deciding score.
- `--preset`, `--speed`, `--target` and `--formats` select the search (defaults `balanced`, `balanced`, `web`, and the default format list).
- `--cache-dir DIR` enables a mode-specific cache; run zero is labelled `cold`, later runs `warm`.
- `--gpu-min-pixels`, `--workers` and `--max-pixels` pass through to the optimizer.
- `--shape files=N,candidates=N,encoder-threads=N,metric-threads=N` overrides the separate uses of `--workers`, to vary one dimension alone (used for the parallelism table below).
- `--full-reports` embeds every image report instead of one summary row per image.
- `--generate-corpus DIR`, `--corpus-sizes` and `--corpus-seed` write the synthetic corpus and exit.

The benchmark output includes the paths of the inputs, so read it before sharing it. The output's `summary` holds median and p95 wall time per mode, median stage times per size class, and two decision checks across every run of every mode: `winner_mismatches` and `quality_floor_crossings`. Both must be empty for a backend to be considered equivalent. The GPU computes only the built-in score, so backend comparisons are run with `--metric builtin`.

## GPU baseline

The runs were pinned to 5 cores (`start /affinity 1F`, `--workers 5`), so the absolute times are for that budget, not for the whole machine. Inputs are the synthetic corpus from `--generate-corpus` (sizes tiny, 512, 2048, 12mp, 50mp) and a 199.8 MP phone photo that is not in the repository (`--corpus-sizes 199mp` generates a synthetic image of the same size class).

| Run | Inputs | Mode | Median wall | Winner mismatches | Floor crossings |
|---|---|---|---:|---:|---:|
| built-in score, GPU forced at every size, 3 runs | 47 images up to 12 MP | cpu | 243 s | 0 | 0 |
| | | gpu-metrics | 224 s | | |
| built-in score, JPEG only, 3 runs | 2 images of 49.9 MP | cpu | 9.67 s | 0 | 0 |
| | | gpu-metrics | 9.55 s | | |
| | | gpu-prepared | 9.07 s | | |
| built-in score, JPEG only, 1 run | 199.8 MP phone photo (not in the repository) | gpu-metrics | 53.0 s | 0 | 0 |
| | | gpu-prepared | 53.5 s | | |
| SSIMULACRA2 (default), 1 run | 47 images up to 12 MP | cpu | 452 s | 0 | 0 |
| | | gpu-metrics | 366 s | | |

The CPU run on the 199.8 MP photo (88 s) went first on a cold file cache with a single repetition and is not comparable.

Median time spent scoring one image with the built-in score (sum over its candidates):

| Size class | CPU | GPU |
|---|---:|---:|
| tiny (up to 64 px) | 140 ms | 18 ms |
| normal (512 and 2048 px) | 4.5 s | 0.34 s |
| 12 MP | 12.6 s | 1.05 s |
| 49.9 MP (reduced level) | 3.4 s | 2.2 s, 0.9 s prepared |

Where the time goes, built-in score, images up to 12 MP: encoding 73% (CPU mode) to 84% (GPU mode) of per-image time, decoding candidates 13%, scoring 10% on the CPU and under 1% on the GPU. With SSIMULACRA2 the external metric is 37% of per-image time.

What these numbers decided:

- GPU scoring stays the default and there is no size threshold (`--gpu-min-pixels` defaults to 0): the GPU was faster in every size class, including images of a few hundred pixels.
- Prepared GPU preprocessing stays in the code but is off by default. It was 5% faster than plain GPU scoring at 49.9 MP and 1% slower at 199.8 MP, below the 10% promotion gate, with identical decisions.
- Further GPU work (buffer reuse, asynchronous completion, GPU-side reduction) is not pursued: with the GPU, scoring is already under 1% of the time. Encoding and, with SSIMULACRA2, the external metric are the costs that matter.

Peak resident memory of the harness process was 0.5 GB for the small corpus, 2.2 GB at 49.9 MP and 4.2 GB at 199.8 MP; encoder child processes are not included. Temporary files peaked at 589 MB per image in the small corpus.

## Search comparison: two-pass search and PAM decoding

Before and after the two-pass search and PAM decoding, default metric, target `smallest`, balanced, pinned to 7 cores. Photos are the six larger stock images in `samples/photos/` (2000 px and more on the long side).

| Image | Before: time, candidates, winner | After: time, candidates, winner |
|---|---|---|
| picsum-1015, 12 MP | 65 s, 39, JXL q83 1,415,647 B | 67 s, 29, JXL q82 1,377,253 B |
| picsum-1025, 12 MP | 70 s, 39, JXL q83 1,578,736 B | 61 s, 27, JXL q83 1,578,736 B |
| picsum-1043, 16.7 MP | 93 s, 39, AVIF q60 2,098,214 B | 89 s, 35, AVIF q61 2,098,214 B |
| picsum-1069, 3.8 MP | 17 s, 24, none (original kept) | 18 s, 27, JXL q78 207,995 B |
| picsum-237, 6 MP | 36 s, 39, AVIF q62 248,895 B | 25 s, 22, AVIF q62 248,895 B |
| picsum-433, 7.7 MP | 38 s, 39, JXL q76 429,535 B | 32 s, 24, JXL q76 429,535 B |

Across the six photos: 195 to 137 candidates, 302 s to 274 s. On four corpus images (photo and screenshot at 2048 px, two transparent images at 512 px) three winners are byte-identical and one 1 KB transparent AVIF is 113 bytes larger. Time spent decoding one JPEG XL candidate fell from 0.5 to 1.9 s to 0.2 to 0.4 s on the photos.

## Search and scheduling: staged grid and shared work budget

This compares the build before the change with the build after it. All runs `--formats jpeg,webp,avif,jxl,png --target smallest --workers 7`, pinned to 7 of 16 logical cores, cold caches, wrapped by `tools/measure-process-tree.rs` (wall time, CPU seconds and peak working set over the whole process tree).

Decisions over 60 image/configuration pairs: 59 identical, 1 smaller, 0 worse. The one difference is `transparency-2048.png`, where the old build failed to decode an AVIF probe and settled on q45 (10,294 B); the new build finds q27 (6,587 B, score 80.3). Seven of the identical cases are "no acceptable candidate" in both builds: all six photos at the strict preset, and one photo at `--speed fast`.

| Configuration | Wall before | Wall after | Change | Encoded | Scored | Encode s |
|---|---:|---:|---:|---:|---:|---:|
| Six photos, one batch (median of 3) | 211 s | 142 s | -33% | 164 to 130 | 84 to 84 | 444 to 251 |
| 512 px corpus, 15 images (median of 3) | 74 s | 58 s | -21% | 353 to 284 | 224 to 170 | 138 to 95 |
| 2048 px corpus, 15 images | 215 s | 191 s | -11% | 352 to 292 | 225 to 167 | 451 to 368 |
| Photos, strict | 228 s | 135 s | -41% | 149 to 122 | 75 to 75 | 434 to 211 |
| Photos, aggressive | 263 s | 143 s | -45% | 135 to 92 | 75 to 73 | 538 to 246 |
| Photos, `--speed fast` | 101 s | 54 s | -46% | 72 to 46 | 31 to 30 | 229 to 79 |

Identical runs differed by 2 to 8% in wall time (one before-run by 17%); the last three rows are single runs.

Single-image latency with default settings (SSIMULACRA2), pinned to 7 cores, median of three, against the budget classes above:

| Photo | Pixels | Before | After | Gate | Result |
|---|---:|---:|---:|---:|---|
| picsum-1069 | 3.8 MP | 15.0 s | 11.8 s | 10 s | miss |
| picsum-237 | 6.0 MP | 17.3 s | 14.0 s | 30 s | pass |
| picsum-433 | 7.7 MP | 25.5 s | 18.7 s | 30 s | pass |
| picsum-1015 | 12.0 MP | 48.2 s | 32.5 s | 30 s | miss |
| picsum-1025 | 12.0 MP | 44.7 s | 31.7 s | 30 s | miss |
| picsum-1043 | 16.7 MP | 65.6 s | 41.8 s | 90 s | pass |

The tiny class (a 32 x 16 image) cannot meet its 1 s gate: about 1.7 s is start-up (tool detection and GPU initialisation) and the rest is process start-up for roughly ninety encoder, decoder and metric invocations, each costing 100 to 700 ms here regardless of image size.

Encoding is the largest share of summed candidate time before and after (69% to 55% on the photo batch); the external metric is second (24% to 33%).

Peak working set over the process tree rose on the larger photos in this change: 5.9 GB to 8.6 GB and 5.8 GB to 9.5 GB for the two 12 MP photos alone, 6.4 GB to 9.6 GB for the photo batch. Candidates waiting for work units keep their decoded buffers. It fell or held elsewhere (2048 px corpus 4.2 GB to 3.3 GB). Peak temporary files per image fell by about a third (605 MB to 390 MB on photos). The next section fixes the rise.

### Where the time went

- Staged coarse grid: scoring time on the corpus fell 44% at 512 px and 33% at 2048 px. On photos it removed 16 encodes and no scoring.
- Lossless gate for JPEG sources: lossless encodes on the six photos fell from 24 to 6 and their time from 175 s to 31 s; batch wall time 239 s to 152 s against the build just before it. Lossless WebP was at most 1.65 times the smallest lossless encoder's output on the JPEG sources measured; the gate fires at 3 times the largest size that could still win. This is a measured margin, not a proof.
- AVIF decode to an uncompressed PNG: about 1000 ms to 680 to 875 ms per decode on the photos.
- Shared per-image pixel conversions: built-in scoring time roughly halved; no visible wall-time change, since encoders and SSIMULACRA2 dominate.

### Parallelism, measured one dimension at a time

Measured with the override knobs behind `--shape`, pinned to 7 logical cores.

| Dimension | Result |
|---|---|
| Files in parallel, six photos | 348 s at 1, 187 s at 2, 115 s at 4, 123 s at 7: flat from 4, the largest photo is the critical path |
| Files in parallel, 512 px corpus | 91 s at 1, 56 s at 2, 52 s at 4 and 7: flat from 2, limited by process start-up (about 21 processes per second) |
| Candidates in flight, 7.7 MP | 40 s at 1, 30 s at 2, 24 s at 4, 19 s at 7 |
| Encoder threads, 7.7 MP | 40 s at 1, 34 s at 4, 32 s at 7 for 21% more CPU: candidates side by side scale better than threads inside one encoder |
| CPU metric threads, 7.7 MP | 90 ms per candidate at 1, 26 ms at 7; about 7% of a candidate, so wall time barely moves |
| GPU contention, 7.7 MP | 0.6 to 0.7 ms of GPU compute per candidate by timestamp query at 1 to 7 candidates in flight, inside about 10 ms of upload and readback; no queueing |

Average and peak cores in use over the process tree, unpinned, photo batch:

| Workers | Before: average / peak | After: average / peak | Wall before / after |
|---:|---|---|---|
| 4 | 4.0 / 7.1 | 3.1 / 4.1 | 99 s / 124 s |
| 7 | 6.2 / 13.0 | 5.4 / 7.1 | 73 s / 80 s |
| 16 | 9.2 / 15.6 | 9.4 / 14.9 | 60 s / 59 s |

Memory per job, fitted to peak working sets: about 190 MB plus (114 + 45 x candidates in flight) bytes per pixel. See [Workers, memory and backpressure](usage.md#workers-memory-and-backpressure) for how the limit is shared.

Forced backpressure on the six photos (7 workers; 89 s without a limit): a memory limit of 4,000,000,000 bytes took 158 s and 2,000,000,000 bytes took 242 s, winners identical; `--max-run-temp-bytes 600000000` took 106 s, winners identical; `500000000` tripped the hard budget on one image, because simultaneous candidates can overshoot the soft limit at three quarters of the budget.

### Decode questions closed

| Question | Finding | Decision |
|---|---|---|
| Native libjxl/libavif decode | JXL and AVIF decode is 5 to 8% of summed candidate time; an in-process decoder would save about 5%. Tool launch costs about 100 ms. jpegxl-rs is GPL-3.0; the `image` crate's AVIF decode needs dav1d through vcpkg on Windows | Deferred; jxl-oxide (pure Rust, MIT/Apache) is the candidate if it is revisited |
| Reduced-resolution decode | `djxl --downsampling` is a no-op for these files; avifdec and djpegli cannot scale. Reduced SSIMULACRA2 scores were 3.7 to 15 points high at 2x and 7 to 22 at 4x on photos, and the sign flips on synthetic content | Rejected |
| Header probe before decode | AVIF boxes and the JPEG XL size header parse in-process in microseconds; `jxlinfo` costs a 12 ms launch | Implemented in-process, 64 KiB read cap |

## Memory: sharing the limit

The staged search raised the peak working set on larger photos. Sampling every process in the tree about 20 times a second, at the peak, on `picsum-1015` (12 MP), 6 workers, pinned to 6 cores:

| Build | Peak | `ssimulacra2` at the peak | Optimizer process |
|---|---:|---|---:|
| Before the staged search | 5.2 to 6.0 GB | 3 to 4 processes, 4.7 to 5.1 GB | 0.34 GB |
| With the staged search, no metric budget | 7.6 to 9.5 GB | 5 to 6 processes, 7.1 to 9.0 GB | 0.46 to 0.53 GB |

One `ssimulacra2` process holds 1.4 to 1.7 GB on a 12 MP image, about 150 bytes per pixel. It counted as one unit of work, and the staged search brings candidates to scoring together. Halving the candidate pool did not move the peak. Each external metric process now reserves 150 bytes per pixel from its own share of the memory limit. The share was measured as a fraction of a limit of half the physical memory:

| Share of that limit | At once, 12 MP | Peak | Wall |
|---|---:|---|---:|
| 1/4 | 2 | 3.9 GB | 62 s |
| 1/3 (chosen) | 3 | 5.6 GB | 47 s |
| 1/2 | 4 | 7.0 GB | 45 s |

The limit is now two thirds of the physical memory with the metric share a quarter of it, which is the same sixth of physical memory as one third of half. Before and after the budget, alternating, cold caches (means of three):

| Case | Peak before | Peak after | Wall before | Wall after |
|---|---:|---:|---:|---:|
| 12 MP photo | 7.9 GB | 5.5 GB | 54.7 s | 53.8 s |
| Six-photo batch | 9.2 GB | 5.6 GB | 223 s | 238 s |

The batch runs of one build ranged from 193 s to 242 s, so the 7% is inside the noise. 164 photo candidates and 361 corpus candidates are identical in size, status and score, with the same winners. Measured on a 31 GB machine only; the share scales with installed memory.

## Six sample photos

Balanced preset, SSIMULACRA2 deciding (floor 80), `--target smallest`, default formats, `--workers 6`, pinned to six cores, with libjxl 0.11.2, libavif 1.4.2, libwebp 1.6.0 and `oxipng` 10.1.1. Sizes in bytes. "The six photos" in this document are the ones of 2000 px and more on the long side; the four smaller low-detail photos in the folder exist for the content classifier and are not part of these measurements.

| Photo | Original | Winner | Size | Score | Smaller by |
|---|---:|---|---:|---:|---:|
| picsum-1015, 12 MP | 1,690,868 | JPEG XL q82 | 1,377,253 | 80.6 | 18.5% |
| picsum-1025, 12 MP | 1,904,469 | JPEG XL q83 | 1,578,736 | 81.1 | 17.1% |
| picsum-1043, 16.7 MP | 2,676,881 | AVIF q61 | 2,098,214 | 80.9 | 21.6% |
| picsum-1069, 3.8 MP | 241,164 | JPEG XL q78 | 207,995 | 80.2 | 13.8% |
| picsum-237, 6 MP | 446,804 | AVIF q62 | 248,895 | 80.2 | 44.3% |
| picsum-433, 7.7 MP | 633,331 | JPEG XL q76 | 429,535 | 80.6 | 32.2% |
| Total | 7,593,517 | | 5,940,628 | | 21.8% |

The batch took 121 s of wall time. It is not comparable with the 142 s at seven workers in the scheduling table above, which used a different build and core count. The photos are already lossy JPEGs, which limits the gain: at the strict preset (floor 90) no candidate is smaller than the source for any of them.

The four smaller, low-detail photos, same settings, run in one batch with the six:

| Photo | Original | Winner | Size | Score | Smaller by |
|---|---:|---|---:|---:|---:|
| picsum-31, 1.7 MP, greyscale | 78,994 | AVIF q61 | 44,820 | 80.6 | 43.3% |
| picsum-44, 1.7 MP, fog | 98,932 | AVIF q63 | 66,907 | 80.4 | 32.4% |
| picsum-63, 1.7 MP, studio | 41,441 | AVIF q55 | 18,001 | 80.1 | 56.6% |
| picsum-72, 1.7 MP, greyscale | 223,291 | JPEG XL q65 | 111,655 | 80.1 | 50.0% |
| All ten | 8,036,175 | | 6,182,011 | | 23.1% |

## Experimental modes: predicted search and in-process scoring

`--search predicted` and `--metric ssimulacra2-inprocess` are off by default. Balanced preset, `--workers 6`, pinned to six cores, cache off. Batch and single-photo times are medians of three after a discarded cold run, except the last column and the 30-photo row, which are single runs. "Staged" is the default search with the external `ssimulacra2` tool.

| Run | Staged | Predicted | Predicted, in-process scoring |
|---|---:|---:|---:|
| picsum-1015 (12 MP), `smallest` | 37.8 s | 24.3 s | 22.2 s |
| picsum-1015 (12 MP), `web` | 29.3 s | 19.8 s | 18.6 s |
| picsum-1025 (12 MP), `smallest` | 35.2 s | 29.3 s | 27.7 s |
| picsum-1025 (12 MP), `web` | 27.6 s | 25.9 s | 24.9 s |
| Ten sample photos, `smallest` | 131.8 s | 117.3 s | 105.9 s |
| Ten sample photos, `web` | 101.9 s | 86.5 s | 80.0 s |
| 30 other stock photos, `smallest` | 209.3 s | 195.5 s | - |

Predicted search, balanced: rounds per photo fell from about 5 to 2.5 to 3. Over 186 image comparisons no output was larger than the staged one and a few were smaller (the largest, picsum-1043: AVIF q61 2,098,214 to q59 2,047,293 bytes). It is not the default because it is slower on the other presets: on the ten photos by 44 to 60% at strict, where the floor is often out of reach and the staged search settles that with one probe, and by 21 to 47% at aggressive, where the starting estimate is furthest off. One synthetic image came out 1.4% larger at strict. The starting qualities were fitted on the ten sample photos, all re-encoded stock JPEGs; other sources are untested, as are the cache, resume, budgets, images over 16 MP and photos with transparency in this mode.

In-process scoring against the external tool: scoring one candidate took 643 ms against 3,033 ms at 12 MP and 180 ms against 968 ms at 3.8 MP (six threads; the external figure includes writing its input file), and preparing the original once per image took 421 ms at 12 MP. Whole runs were 4 to 23% faster and the peak working set of a single 12 MP photo fell from 5.6 GB to 3.5 GB. On 545 candidates from real runs the two scores differed by 0.013 points on average and 0.315 at most, no accept or reject verdict differed and every winner was the same file. On 855 synthetic degradations the largest difference was 1.2 points (ICC-profile and checkerboard images at 2048 px), which is unexplained and is why `auto` does not use it.

## Other findings

- **Scoring metric.** The built-in score, twice the PSNR over the colour channels combined with the same measure on an edge map, is not perceptual. Against SSIMULACRA2 on two phone photos (not in the repository) it ranked codecs inconsistently: the same SSIMULACRA2 score, 19 points apart on the built-in scale. Its old floor of 90 forced near-lossless output (88% of the original size). SSIMULACRA2 is now the deciding score when its program is present, with floors of 90, 80 and 70; one of those photos lands at 33% of its original size at floor 80.
- **A 199.8 MP image.** A 12240 x 16320 phone JPEG (32.3 MB; not in the repository), target `smallest`, balanced preset, completed in about 9 minutes with a valid report and about 1 GB of temporary files, inside the 900 s gate. With SSIMULACRA2 scored on full-resolution samples the winner is JPEG q83 at 7.02 MB (22% of the original), score 80.9. An earlier figure for this image (JPEG XL q44, "quality 90.5") came from the built-in score on a 2048 px reduction; on full-resolution samples that file scores 64, and a run that still scored the reduction accepted a WebP at quality 9. That is why large images are scored on full-resolution samples.
- **Bugs found while measuring.** JPEG XL "lossless" candidates were being encoded at distance 1 (lossy); the exhaustive speed mapped to faster, weaker WebP and JPEG XL settings; truncated JPEGs were accepted. All three are fixed.
- **Timing of GPU work.** With timestamp queries, the metric kernel takes 0.3 to 0.7 ms per candidate against about 10 ms of host-side upload and readback, with no queueing up to seven candidates at once. This is why further GPU work is not pursued (see the [roadmap](../ROADMAP.md)).

## Measuring the whole process tree

`tools/measure-process-tree.rs` runs a command inside a Windows job object and reports wall time, user and kernel CPU seconds, average and peak cores, and peak working set and commit summed over every process in the tree, which a harness that only looks at its own process cannot see. Build it with `rustc -O --edition 2024 tools/measure-process-tree.rs -o target/measure/measure.exe` and run `measure.exe [--summary F] -- <command> <args>`; process-tree figures such as peak working set are taken this way (about 20 samples a second).

## Reproducing the published numbers

Install the encoders (`tools/fetch-tools.ps1`) and build first (`cargo build --release --bins`). Pinning keeps runs comparable: from `cmd`, `start /B /WAIT /AFFINITY 3F <exe> <args>` restricts a command and its children to six logical cores (from Git Bash, pass that whole string to `cmd //C`). Match `--workers` to the number of cores you pin to.

```powershell
# All sample photos (the table above lists the six larger ones): output files and a JSON report
target\release\smart-compress.exe samples\photos --recursive --target smallest --workers 6 --no-progress --out-dir target\photos-out --report json --report-out target\photos-report.json

# Synthetic corpus, then the end-to-end harness over it
target\release\smart-compress-benchmark.exe --generate-corpus target\corpus
target\release\smart-compress-benchmark.exe target\corpus --runs 3 --out target\benchmark-results.json
```

The first command reproduced the sizes in the table above exactly with the tool versions listed there; other tool versions can change them, and times depend on the machine. The corpus is byte-identical for equal options and seed, and `corpus.json` records the hash of every file. The phone-photo rows cannot be reproduced from the repository; `--generate-corpus target\corpus --corpus-sizes 199mp` produces a synthetic image of the same size class.
