# lossy-and-found

[![ci](https://github.com/FueledByRedBull/lossy-and-found/actions/workflows/ci.yml/badge.svg)](https://github.com/FueledByRedBull/lossy-and-found/actions/workflows/ci.yml)

**`smart-compress` shrinks JPEG, PNG, WebP and AVIF images to the smallest file
that still passes a quality check you choose.** It encodes each image with
several codecs at several settings, measures every result against the original
with a perceptual metric (SSIMULACRA2), and keeps the smallest one above the
quality floor. Your original is never overwritten.

(The repository is `lossy-and-found`; the program it builds is `smart-compress`.)

```text
> smart-compress samples\photos\picsum-433-3200x2400.jpg
Original: 633331 bytes, 3200x2400, Photo, decoded Rgb8, source Rgb8
[...]
Winner: Avif q67/e6 -> 455682 bytes
Output: samples\photos\optimized\picsum-433-3200x2400.avif
[...]
Summary: 1 improved, 0 kept as they were, 0 skipped, 0 failed
```

You state a quality level; it finds the codec and the quality number. The price
is time: it makes tens of encodes per image, so a 12 MP photo takes about half a
minute on six cores. If nothing smaller passes the check, it writes nothing and
says so.

![The control panel before any image is added](docs/panel.png)

## Contents

- [What it saves](#what-it-saves)
- [Get started](#get-started)
- [Using it](#using-it)
- [The desktop app](#the-desktop-app)
- [How it decides](#how-it-decides)
- [Known limits](#known-limits)
- [Documentation](#documentation)
- [Status](#status)
- [Licence](#licence)

## What it saves

The ten stock photos in [`samples/photos/`](samples/photos/SOURCES.md), balanced
preset (no visible difference side by side), `--target smallest`:

| Photo | Original | Result | Smaller by |
|---|---:|---|---:|
| `picsum-1015`, 12 MP | 1,690,868 | 1,377,253, JPEG XL | 18.5% |
| `picsum-1043`, 16.7 MP | 2,676,881 | 2,098,214, AVIF | 21.6% |
| `picsum-237`, 6 MP | 446,804 | 248,895, AVIF | 44.3% |
| `picsum-63`, 1.7 MP | 41,441 | 18,001, AVIF | 56.6% |
| All ten | 8,036,175 | 6,182,011 | 23.1% |

Sizes are in bytes. These photos were already lossy JPEGs, which limits the gain.
At the strict preset (visually lossless) the six larger ones cannot be made
smaller at all, and the program leaves them as they are. The full table, the
method and the timings are in [docs/benchmarking.md](docs/benchmarking.md).

## Get started

You need **Windows 11 (x64)** and **Rust 1.94 or newer**. No release binaries are
published yet, so you build it:

```powershell
git clone https://github.com/FueledByRedBull/lossy-and-found
cd lossy-and-found
powershell -File tools\fetch-tools.ps1
cargo build --release --bins
target\release\smart-compress.exe samples\photos\picsum-433-3200x2400.jpg
```

The result lands in an `optimized` folder next to the image.

`tools\fetch-tools.ps1` downloads the encoders the program runs as separate
processes: the official releases of libjxl 0.11.2 (JPEG XL, `cjpegli` and the
`ssimulacra2` score), libavif 1.4.2, libwebp 1.6.0 and pngquant 2.17.0. Each
download is checked against a pinned SHA-256, and only the programs that are
used are kept, with their licence files. They are not stored in this repository;
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists them. `oxipng` is optional
and is used if it is on `PATH`.

Without the tools the program still runs: only the PNG and JPEG encoders are
built in, the formats whose tools are missing are skipped, and the report lists
what is missing. See [External tools](docs/usage.md#external-tools).

## Using it

```powershell
# One image
smart-compress photo.jpg

# A folder, with a chosen output folder and a JSON report
smart-compress .\photos --recursive --out-dir out --report json --report-out report.json

# See what would be processed, without encoding anything
smart-compress photo.jpg --dry-run
```

(`smart-compress` here stands for `target\release\smart-compress.exe`.)

Two options do most of the work:

| `--preset` | SSIMULACRA2 floor | Meaning |
|---|---:|---|
| `strict` | 90 | Visually lossless |
| `balanced` (default) | 80 | No visible difference side by side at full size |
| `aggressive` | 70 | High quality |

| `--target` | Formats tried | How the winner is picked |
|---|---|---|
| `web` (default) | JPEG, PNG, WebP, AVIF | Smallest, but the more widely supported format wins when it is within a few percent |
| `smallest` | The above and JPEG XL | Purely on size |
| `max-compat` | JPEG, PNG | Smallest |

Things worth knowing:

- **Nothing smaller passed.** The image is reported as `kept`, nothing is
  written, and that is not an error. The exit status is 1 only when an image
  failed.
- **Second runs.** Images whose output already exists are skipped unless you
  pass `--overwrite`.
- **Reports** come as text, JSON or Markdown and list every candidate with its
  size, score, timing and the reason it was rejected or skipped.

<details>
<summary>The candidate table of the run at the top of this page</summary>

`samples/photos/picsum-433-3200x2400.jpg`, default settings, `--workers 6`,
17 seconds. The 633,331-byte JPEG became a 455,682-byte AVIF (28% smaller) that
scores 80.8 against a floor of 80.

| Codec | Settings | Size (bytes) | SSIMULACRA2 | Status |
|---|---|---:|---:|---|
| Jpeg | q78/e0 | 661230 | - | rejected: not smaller than original |
| Webp | q78/e4 | 396126 | 70.1 | rejected: quality below 80.0 |
| Avif | q78/e8 | 590085 | 83.6 | accepted |
| Webp | lossless | 3416746 | - | rejected: not smaller than original |
| Jpeg | q60/e0 | 470523 | 76.8 | rejected: quality below 80.0 |
| Webp | q92/e4 | 809914 | - | rejected: not smaller than original |
| Avif | q60/e8 | 400901 | 77.0 | rejected: quality below 80.0 |
| Webp | q60/e4 | 294758 | 61.3 | rejected: quality below 80.0 |
| Jpeg | q92/e0 | - | - | skipped: assumed to pass |
| Avif | lossless | - | - | skipped: cannot win |
| Avif | q92/e8 | - | - | skipped: assumed to pass |
| Png | lossless | - | - | skipped: cannot win |
| Jpeg | q67/e0 | 534122 | 78.4 | rejected: quality below 80.0 |
| Jpeg | q69/e0 | 557926 | 80.1 | accepted |
| Jpeg | q71/e0 | 582296 | 81.3 | accepted |
| Avif | q66/e6 | 431882 | 79.9 | rejected: quality below 80.0 |
| Avif | q68/e6 | 471336 | 81.3 | accepted |
| Avif | q70/e6 | 481797 | 81.7 | accepted |
| Avif | q67/e6 | 455682 | 80.8 | accepted (winner) |

Qualities above or below an already-tested point are skipped as "assumed to pass"
instead of being encoded, and candidates that cannot beat the best so far are
skipped as "cannot win".

</details>

Every option, the defaults, the report layout and the configuration file are in
[docs/usage.md](docs/usage.md).

## The desktop app

Double-click `gui.cmd` (it builds first if the program is missing) or
`target\release\smart-compress-app.exe`. Drop images onto the window or use the
file dialogs, change any setting, watch progress, cancel, and compare each result
with its original. It runs the same optimizer as the command line and shows the
equivalent command.

The window is a page compiled into the executable and served on `127.0.0.1`
only, behind a per-session token, shown by Microsoft Edge in app mode (a browser
tab is the fallback). `smart-compress-app.exe` must stay next to
`smart-compress.exe`. `smart-compress --gui` opens the same page in your default
browser instead.

## How it decides

1. **Read.** It checks the content against the extension, applies the EXIF
   orientation to the pixels, and labels the image (photo, screenshot,
   transparent asset, simple graphic) for the report.
2. **Encode.** Candidates are made from the decoded original pixels, never from
   another lossy candidate. The quality where each codec crosses the floor is
   found by bracketing and interpolation, and candidates that cannot win are
   skipped.
3. **Score.** Every candidate is decoded and scored against the original with
   SSIMULACRA2. Transparent images must also pass alpha checks.
4. **Keep.** The smallest candidate that is smaller than the original and at or
   above the floor is written atomically.

Memory, threads, temporary disk and time are budgeted: `--workers` is a real cap
on threads, and work waits instead of exceeding the memory plan.

Two experimental options make it faster and are off by default: `--search
predicted` starts each format near its expected quality (a 12 MP photo in 20 to
29 seconds instead of 28 to 38; it helps at the balanced preset only), and `--metric
ssimulacra2-inprocess` scores inside the program without the external tool.
Their measurements and limits are in
[docs/benchmarking.md](docs/benchmarking.md#experimental-modes-predicted-search-and-in-process-scoring).

## Known limits

- **Not fast.** With a seven-core cap a single photo takes about 12 s at 3.8 MP,
  32 s at 12 MP and 42 s at 16.7 MP.
- **Windows first.** Process-tree termination, the Ctrl+C handler,
  physical-memory detection, the app window and the file dialogs are
  Windows-only, and nothing else has been tested. See
  [Platform notes](docs/usage.md#platform-notes).
- **No colour conversion.** An output carries the source's colour profile
  unchanged. A source whose colour cannot be carried (a non-sRGB PNG without a
  profile, a CMYK JPEG with its own profile) is refused with the reason in the
  report. No HDR.
- **No animation, no resizing.** GIF, animated WebP, APNG and animated AVIF are
  rejected, as are SVG, BMP, TIFF and JPEG XL inputs. Dimensions never change.
- **No hardware encoders.**
- **The scoring has limits.** Images above 16 MP are scored on a mosaic of
  full-resolution samples, so a small busy region between samples can be missed.
  Without the `ssimulacra2` tool a built-in PSNR-derived score decides, which is
  not perceptual.

## Documentation

- [docs/usage.md](docs/usage.md): every option, quality scoring, colour and
  metadata rules, reports, the configuration file, troubleshooting.
- [docs/benchmarking.md](docs/benchmarking.md): method, results, how to
  reproduce them.
- [docs/maintainers.md](docs/maintainers.md) and
  [docs/release.md](docs/release.md): CI, machine-dependent tests, packaging.
- [ROADMAP.md](ROADMAP.md): open work and what was decided against.

## Status

Version 0.1.0, before a first release. It is maintained by one person and tested
on Windows 11 only; flags, defaults and the report layout can still change
(reports carry a schema version, currently 6). Bug reports and measured
improvements are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). To report a
vulnerability, see [SECURITY.md](SECURITY.md).

## Licence

`smart-compress` is available under either the [MIT](LICENSE-MIT) or the
[Apache-2.0](LICENSE-APACHE) licence, at your option.

The encoder programs it runs are third-party software under their own licences,
listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); they are downloaded,
not included here. The photographs in `samples/photos/` are Unsplash photos
under the Unsplash License, listed in
[samples/photos/SOURCES.md](samples/photos/SOURCES.md), and are not covered by
the project licence.

It builds on the work of others: [libjxl](https://github.com/libjxl/libjxl)
(JPEG XL, jpegli and the SSIMULACRA 2 metric),
[libavif](https://github.com/AOMediaCodec/libavif) with aom and dav1d,
[libwebp](https://developers.google.com/speed/webp), pngquant,
[oxipng](https://github.com/oxipng/oxipng), and the Rust crates in `Cargo.lock`.
