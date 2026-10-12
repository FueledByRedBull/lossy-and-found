# Roadmap

What is still open in `smart-compress`, and what has been decided against. The
measurements behind these items are in [docs/benchmarking.md](docs/benchmarking.md);
behaviour that already exists is described in [docs/usage.md](docs/usage.md).
Nothing here is a commitment or a schedule.

## Design rules

The optimizer minimises output bytes subject to a quality floor, an alpha and
dimension policy, a colour and metadata policy, a compatibility preference, and
budgets for time, memory, temporary disk and CPU. These rules change only for a
reason that is measured:

- Never overwrite an original by default.
- Never generate a lossy candidate from another lossy candidate; every candidate
  is encoded from the decoded original pixels.
- Never silently strip required metadata, flatten transparency, ignore
  animation or change colour interpretation. Refuse with a reason instead.
- Use mature external encoders where they beat or are easier to maintain than a
  native integration. Do not write a new image codec.
- Promote a path to the default only when an end-to-end benchmark shows at least
  a 10% median gain with no material p95 regression and identical decisions.

## Open work

### Speed

- Single-image latency misses its gates at a 7-core cap for 3.8 MP (11.8 s
  against 10 s) and 12 MP (about 32 s against 30 s). Encoding and the external
  SSIMULACRA2 process are the remaining costs.
- Hardware AV1 encoding (for example AMF or NVENC through ffmpeg) has not been
  evaluated, so its speed and size against the software encoder are unmeasured.
  No hardware encoder is used.

### Quality of results

- An opt-in mode that scrubs the colour hidden under fully transparent pixels.
  Today that colour is ignored by the built-in score but still reaches the
  encoders unchanged, so it can cost bytes.
- AVIF searches for the quality crossing at a cheaper encoder effort and then
  re-encodes near it at the final effort. An edge case in that final-effort
  re-encode is known and still open.
- Images above 16 MP are scored on a 3 x 3 mosaic of full-resolution samples; a
  small busy region between the samples can be missed.

### Colour

- Not covered: EXIF colour-space flags in untagged JPEGs, HDR gain maps and
  rendering intent. Profile carriage is pinned by tests against the tool versions
  the fetch script installs and is not checked at run time.
- No colour conversion and no wide-gamut or HDR output. Sources that cannot be
  carried unchanged are refused with a reason.

### Code

- Split `pipeline.rs` and `scoring.rs` in `smart-compress-core` into modules;
  each is several thousand lines.
- Add a keyed check to the score cache. Cached files are verified by hash, but
  someone who can write to the cache folder and recompute hashes could alter a
  stored score.
- Peak memory on 12 MP photos is about 5.5 GB over the process tree, above the
  per-class budgets in the benchmarking document. The share given to SSIMULACRA2
  processes was measured on one 31 GB machine only.

### Platforms and distribution

- Windows is the only tested platform. Process-tree termination, the Ctrl+C
  handler, physical-memory detection, the control-panel window and the native
  file dialogs are Windows-only, and the fetch script downloads Windows builds.
  Supporting other platforms means providing each of these, and CI for them. See
  [Platform notes](docs/usage.md#platform-notes).
- Published release binaries. Today the program is built from source; see
  [docs/release.md](docs/release.md) for how a package is staged.

## Not pursued

- GPU tuning (buffer reuse, batched submissions, asynchronous completion, GPU-side
  reductions). The GPU stage is under 1% of a run: timestamp queries put GPU
  compute at 0.6 to 0.7 ms per candidate, inside about 10 ms of upload and
  readback, against 1 to 5 s per candidate for encoding and SSIMULACRA2. Revisit
  only if the built-in score becomes the deciding score on large batches of small
  images. The experimental prepared-GPU preprocessing path stays in the code, off
  by default: it was 5% faster at 49.9 MP and 1% slower at 199.8 MP, below the
  10% gate.
- Reduced-resolution decoding to speed up scoring: reduced scores read 4 to 22
  points high on photos and cannot gate acceptance.
- In-process JPEG XL and AVIF decoding: it would save about 5% of summed work.
  `jxl-oxide` (pure Rust, MIT/Apache) is the candidate if that share grows.

## Deferred ideas

Considered, not started:

- Bayesian or bandit allocation of candidates (only if the measured pruning no
  longer meets the latency targets).
- Saliency-weighted or protected-region scoring; face and text detection;
  perceptual difference heat maps.
- Multi-output bundles (AVIF with WebP and JPEG fallbacks); crop and resize
  suggestions.
- Watch-folder mode; a CI image-budget checker; a plugin system for codecs and
  metrics.
- Native or FFI replacements for the encoder programs.
