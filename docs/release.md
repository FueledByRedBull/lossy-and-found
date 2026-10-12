# Windows release packaging

No release binaries are published yet; this describes how to stage one.
`tools/package-release.ps1` stages a folder that can be zipped and handed to
someone without a Rust toolchain. It is a staging script only: it does not run
checks, create an archive or compute hashes.

## Steps

1. Start from a clean checkout of the commit to release, and run
   `powershell -File tools\fetch-tools.ps1` to install the encoders.
2. Run the checks that CI runs:
   `cargo fmt --all -- --check`,
   `cargo clippy --workspace --all-targets -- -D warnings` and
   `cargo test --workspace --lib --tests -- --skip gpu_prepared_scores_match_known_large_image`.
3. Run `powershell -File tools\package-release.ps1` from the workspace root. An
   optional `-Output <folder>` changes the staging folder (default
   `target/release-package`). Use a folder that does not exist yet or is empty:
   copying over an earlier staging folder nests the `docs` folder inside itself.
4. Smoke-test the staged folder from a different directory:
   `smart-compress.exe <image> --dry-run`, then a real run with `--out-dir`, then
   double-click `smart-compress-app.exe`.
5. Zip the folder and record the SHA-256 of the archive.

## What the script stages

It runs `cargo build --release --bins`, stops if that fails, and copies into the
staging folder:

- `smart-compress.exe`, `smart-compress-app.exe` (the windowed launcher, which
  must stay next to `smart-compress.exe`) and `smart-compress-benchmark.exe`
- `README.md`, `ROADMAP.md`, `THIRD_PARTY_NOTICES.md`, `LICENSE-MIT`,
  `LICENSE-APACHE`
- the `docs` folder
- those of the tool folders `tools/libjxl`, `tools/libavif`, `tools/libwebp` and
  `tools/pngquant` that exist, so the package contains only what
  `tools/fetch-tools.ps1` installed

## Tool discovery in a package

Discovery checks `PATH` first, then the `tools` folder next to the executable (see
[usage.md](usage.md#external-tools)), so the staged `tools` folder is found
without configuration, and a package may omit it if the programs are installed
separately. The report records every unavailable tool and its install hint.
`oxipng` is not part of `tools` and is used only if it is on `PATH`.

The staged tool folders contain programs under their own licences, with the
licence files the fetch script installs next to them; `THIRD_PARTY_NOTICES.md`
travels with the package. pngquant is GPL-3.0-or-later, so a package that
includes it must also offer its source (the notices file names the version and
where to find it). Leave `tools/pngquant` out of the package if you do not want
that obligation; PNG output still works without it.

## Before publishing

Do not publish a staging folder that contains an empty report, a cache-only run
or generated candidate artifacts. Keep the benchmark JSON and source hashes for
the release separately as evidence of what was measured.
