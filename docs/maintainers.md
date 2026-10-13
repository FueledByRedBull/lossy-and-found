# Maintainer notes

## What is authoritative

The repository is the source of truth. The checked-in CI workflow and
`Cargo.lock` are the reproducibility boundary. `target/`, output folders and the downloaded programs under
`tools/` are generated and are not tracked. Personal photographs are never
committed; `samples/photos/` holds stock photos with their sources (see
`samples/photos/SOURCES.md`).

## Workflows

- `ci` (`.github/workflows/ci.yml`) runs on every push and pull request on
  `windows-latest` with the stable Rust toolchain. It first runs
  `tools/fetch-tools.ps1` (the downloads are cached, keyed on the hash of the
  script), then:
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace --lib --tests -- --skip gpu_prepared_scores_match_known_large_image`,
  and `cargo test -p smart-compress --bin smart-compress --bin smart-compress-benchmark`.
  The job sets `SMART_COMPRESS_REQUIRE_TOOLS=1`, so a test that cannot find its
  tool fails instead of skipping. Hosted runners have no GPU, so
  `SMART_COMPRESS_REQUIRE_GPU` is not set and GPU tests skip.
- `gpu-large-validation` (`.github/workflows/gpu-large.yml`) is started by hand
  and needs a self-hosted runner labelled `windows` and `gpu`. It fetches the
  tools, runs the ignored large-image test with `SMART_COMPRESS_REQUIRE_GPU` set,
  then a `--modes cpu,gpu-prepared` benchmark run, and uploads the benchmark JSON. A
  GitHub-hosted runner has no GPU and no large image, so it would validate
  nothing.

## Tests that depend on the machine

- **External tools.** Tests that need `cwebp`, `webpmux`, `avifenc`/`avifdec`,
  `cjxl`/`djxl` or `ssimulacra2` check the tool inventory and print a "skipped"
  line when a tool is missing. A green run on a machine without the tools
  therefore proves less than one with them. Setting
  `SMART_COMPRESS_REQUIRE_TOOLS` (to any value) turns each of those skips into a
  failure; CI sets it. No test asks for `oxipng` or `pngquant`: the
  script never installs the first and installs `pngquant` on a best-effort
  basis.
- **GPU.** GPU tests skip with a one-line message on machines without an
  adapter, so a green CI run proves nothing about the GPU path. Setting
  `SMART_COMPRESS_REQUIRE_GPU` (to any value) turns each of those skips into a
  failure; run the suite with it on any machine that is meant to have a GPU.
- **Large image.** The ignored test `gpu_prepared_scores_match_known_large_image`
  fails, rather than skips, when its input is missing. Its only input is the file
  named by `SMART_COMPRESS_LARGE_IMAGE`: a JPEG of about 200 MP (the test accepts up to 250 MP), which is not
  published. Run it with
  `cargo test -p smart-compress-core --test fixture_corpus gpu_prepared_scores_match_known_large_image -- --ignored --nocapture`.
- **Windows.** Some unit tests start `powershell.exe` and are compiled only on
  Windows.

## Snapshots

Report renderings are pinned by `tests/snapshots/`. After an intended change to
the text, Markdown or JSON report, regenerate them with
`UPDATE_SNAPSHOTS=1 cargo test -p smart-compress-core --test report_snapshots`
and review the diff. When the report layout changes in a way readers of old
reports would notice, bump `REPORT_SCHEMA_VERSION` in
`crates/smart-compress-core/src/report.rs` (currently 7) and the scoring version
when scores change meaning.

## Third-party tools

`tools/fetch-tools.ps1` downloads the official Windows releases of libjxl 0.11.2
(`cjxl`, `djxl`, `cjpegli`, `ssimulacra2`, `jxlinfo`),
libavif 1.4.2 (`avifenc`, `avifdec`), libwebp 1.6.0 (`cwebp`, `webpmux`) and
pngquant 2.17.0, verifies each against a pinned SHA-256, and installs only those
programs and the licence files into the layout the program searches. Run it with
`-Force` to reinstall and `-CacheDir <folder>` to keep the downloads (CI does).
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) lists the packages and their
licences. `oxipng` is not installed and is looked up on `PATH`.

To upgrade a tool, change the URL and hash in the script and the notices file
together. The report records each tool's name and version line, and the candidate
cache key includes the tool inventory, so replacing a tool invalidates cached
candidates. After an upgrade, rerun the colour-fidelity tests
(`cargo test -p smart-compress-core --test colour_fidelity`), because profile
carriage is pinned against the installed versions.

## Release validation

Validate a release from a clean checkout:

1. The CI commands above.
2. The CPU and fixture suite, on a machine with the tools installed and
   `SMART_COMPRESS_REQUIRE_TOOLS` set.
3. A retained JSON report from `smart-compress-benchmark` over `tests/fixtures`
   and `samples/photos`, kept with the input hashes it records.
4. For changes that touch GPU scoring or large-image handling, the
   `gpu-large-validation` workflow, with its report, tool versions, input hash,
   selected adapter and completion status.

Packaging is described in [release.md](release.md).

## Measuring changes

Follow [benchmarking.md](benchmarking.md): compare at least three runs, report
median and p95 wall time, and require a 10% median gain before a path becomes a
default. Pin runs to a fixed set of cores and set `--workers` to match, or the
numbers are not comparable. `tools/measure-process-tree.rs` measures the cost of
the whole process tree, including the encoder processes.
