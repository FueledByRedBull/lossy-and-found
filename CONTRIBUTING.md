# Contributing

Bug reports, fixes and measured improvements are welcome. For anything larger
than a small fix, open an issue first so the approach can be agreed before you
spend time on it.

## Build

You need Windows 11 and Rust 1.94 or newer (edition 2024).

```powershell
powershell -File tools\fetch-tools.ps1
cargo build --release --bins
```

The first command downloads the encoder and metric programs the optimizer runs
into `tools/` (see the [README](README.md#requirements)). They are not stored in
the repository.

## Checks

CI (`.github/workflows/ci.yml`, on `windows-latest`) runs these, and a change
should pass them before review:

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --lib --tests -- --skip gpu_prepared_scores_match_known_large_image
cargo test -p smart-compress --bin smart-compress --bin smart-compress-benchmark
```

Tests that need `cwebp`, `webpmux`, `avifenc`/`avifdec`, `cjxl`/`djxl` or
`ssimulacra2` print a "skipped" line when the tool is not found, and GPU tests
skip with one line when no adapter exists. CI installs the tools and sets
`SMART_COMPRESS_REQUIRE_TOOLS`, which turns a tool skip into a failure; you can
set it too. `SMART_COMPRESS_REQUIRE_GPU` does the same for GPU tests. The
`gpu_prepared_scores_match_known_large_image` test is ignored and needs an image
of about 200 MP (`SMART_COMPRESS_LARGE_IMAGE`); contributors do not need to run
it. Details are in [docs/maintainers.md](docs/maintainers.md).

If you change the report layout, regenerate the pinned renderings with
`UPDATE_SNAPSHOTS=1 cargo test -p smart-compress-core --test report_snapshots`
and include the diff.

## Changes that affect results or speed

Search, scoring, scheduling and default-routing changes need numbers. Compare at
least three runs with `smart-compress-benchmark`
([docs/benchmarking.md](docs/benchmarking.md)), show that winners and quality
floors did not get worse, and describe the machine. A speed change is worth
taking when it gives a median gain of about 10% without a material p95
regression.

## Style

Follow the code around you and let `rustfmt` and `clippy` decide formatting. Keep
a pull request to one logical change. Update [docs/usage.md](docs/usage.md) when
behaviour or an option changes.

## Test images

Do not add personal photographs. New sample images need a licence that allows
redistribution, and their source recorded in `samples/photos/SOURCES.md`.

## Licence

The project is available under MIT or Apache-2.0, at your option. By submitting a
contribution you agree that it is licensed the same way.
