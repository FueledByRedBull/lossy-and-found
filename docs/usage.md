# Usage

This is the reference for `smart-compress`, the program built from this
repository. For an overview, the requirements and a first run, start with the
[README](../README.md).

`smart-compress` reads an image, makes several smaller versions of it with
different formats and quality settings (the candidates), scores each candidate
against the original, and writes the smallest one that stays above a quality
floor. Nothing is written when no candidate qualifies, and the original file is
never modified.

Contents: [Quick start](#quick-start) | [Command line](#command-line) |
[Options](#options) | [Targets and presets](#targets-and-presets) |
[Quality scoring](#quality-scoring) |
[Colour, metadata and alpha](#colour-metadata-and-alpha) | [Inputs](#inputs) |
[How the search works](#how-the-search-works) |
[Workers, memory and backpressure](#workers-memory-and-backpressure) |
[Reports, resume and cache](#reports-resume-and-cache) | [GPU](#gpu) |
[External tools](#external-tools) | [Control panel](#control-panel) |
[Configuration file](#configuration-file) | [Platform notes](#platform-notes) |
[Environment variables](#environment-variables) |
[Troubleshooting](#troubleshooting)

## Quick start

These commands run from the repository root after
[building](../README.md#install) and
[installing the external tools](../README.md#install). `optimized` is the
output folder; it is ignored by git.

```powershell
# See what would be processed in a folder, without encoding anything
target\release\smart-compress.exe samples\photos --recursive --dry-run

# Optimize one image; the result goes to the "optimized" folder
target\release\smart-compress.exe samples\photos\picsum-433-3200x2400.jpg --out-dir optimized

# Write a JSON report to a file instead of printing a text report
target\release\smart-compress.exe samples\photos\picsum-237-2000x3000.jpg --out-dir optimized --report json --report-out optimized\report.json

# Check that a JSON report is readable and consistent
target\release\smart-compress.exe --validate-report optimized\report.json
```

Without `--out-dir`, each result goes to an `optimized` folder next to its input.

## Command line

```powershell
smart-compress [OPTIONS] [INPUT]...
```

An input is an image file or, with `--recursive`, a folder. Without
`--recursive` a folder is an error. `smart-compress --help` lists every option in
the groups used below, and `--version` prints the version.

A folder is walked completely: hidden folders and files, and files that a
`.gitignore` or `.ignore` file would exclude, are included. Symbolic links are
not followed. Only files with the extensions `png`, `jpg`, `jpeg`, `webp` and
`avif` are collected, a file reached twice is processed once, and a part of the
tree that cannot be read is reported as a warning instead of stopping the run.
A walk skips the run's own output: everything below the `--out-dir` folder, or,
without `--out-dir`, files inside any folder named `optimized`. A file named on
the command line is always an input.

Each winner is written as `<input stem>.<extension of the winning format>` in the
`--out-dir` folder, or in an `optimized` folder next to the input when `--out-dir`
is not given. When two inputs of one run would publish to the same name, each
gets a short hash suffix. `--out FILE` writes the winner of a single input to
exactly that path. It needs exactly one input. If the name ends in a supported
extension (`png`, `jpg`, `jpeg`, `webp`, `avif` or `jxl`), only that format is
tried; any other name gets the winner's extension appended, with a warning.

An output that already exists is not replaced: the image is skipped before any
search, so running the same command twice is cheap. `--overwrite` replaces
existing outputs. While `--protect-inputs` is on (the default), an output never
replaces an input image, nor any existing file in a folder that holds input
images, even with `--overwrite`.

### Outcomes and exit status

Every image ends in one outcome, shown in the report:

| Outcome | Meaning |
|---|---|
| `optimized` | A smaller output passed the quality floor and was written. |
| `kept` | No candidate was both smaller than the original and above the floor. Nothing was written, the original is untouched, and the report says so in a warning. |
| `skipped` | The output already exists and `--overwrite` is off. |
| `dry-run` | `--dry-run`: the image was read and classified, nothing was encoded or written. |
| `failed` | The image could not be processed (unsupported or corrupt file, a limit exceeded, a colour that cannot be carried, a write error, or candidates that failed before they could be judged). The report's `errors` say why. |

The exit status is 0 unless at least one image `failed`, in which case it is 1.
Usage errors (an unknown option, a missing input) exit with 2. The text and
Markdown reports end with a line such as
`Summary: 3 improved, 1 kept as they were, 0 skipped, 0 failed`. A panic while
processing one image becomes a `failed` entry for that image; the rest of the
batch continues. A search stopped by `--max-image-time-seconds` or
`--max-candidate-attempts` prints `Search stopped early` with the reason and adds
a warning, because a smaller passing candidate may have gone untried.

### More examples

```powershell
# A stricter floor and a wider search
target\release\smart-compress.exe samples\photos\picsum-237-2000x3000.jpg --out-dir optimized --min-quality 85 --speed exhaustive

# Allow JPEG XL and take the smallest file regardless of support
target\release\smart-compress.exe samples\photos\picsum-237-2000x3000.jpg --out-dir optimized --target smallest

# Only these formats
target\release\smart-compress.exe samples\photos --recursive --out-dir optimized --formats avif,webp,jpeg

# Score on the CPU only, with no progress bars
target\release\smart-compress.exe samples\photos\picsum-237-2000x3000.jpg --out-dir optimized --cpu-only --no-progress

# Reuse encoded candidates and scores between runs
target\release\smart-compress.exe samples\photos\picsum-237-2000x3000.jpg --out-dir optimized --cache

# Write a report file while a batch runs, and carry on from it after an interruption
target\release\smart-compress.exe samples\photos --recursive --out-dir optimized --report json --report-out optimized\report.json
target\release\smart-compress.exe samples\photos --recursive --out-dir optimized --report json --report-out optimized\report.json --resume-report optimized\report.json
```

## Options

The tables follow the groups of `smart-compress --help`. Defaults are those of
the code (`crates/smart-compress-core/src/config.rs`). An option given on the
command line overrides the same setting in a
[configuration file](#configuration-file).

### Input and output

| Option | Default | Meaning |
|---|---|---|
| `--out FILE` | none | Write the result to exactly this file. One input only; the extension picks the format. |
| `--out-dir DIR` | `optimized` next to each input | Folder for the results. |
| `--recursive` | off | Process the images in the folders given as inputs, subfolders included. |
| `--dry-run` | off | Read and classify the images and report, without encoding or writing anything. |
| `--overwrite[=BOOL]` | off | Replace output files that already exist. `--overwrite=false` overrides `overwrite = true` in a configuration file. |
| `--protect-inputs[=BOOL]` | on | With `--overwrite`, refuse to replace an input image or any file in a folder that holds input images. `--protect-inputs=false` allows it. |
| `--config FILE` | none | Read settings from a TOML file; see [Configuration file](#configuration-file). |
| `INPUT...` | none | Files, or folders with `--recursive`. |

### Quality

| Option | Default | Meaning |
|---|---|---|
| `--preset strict\|balanced\|aggressive` | `balanced` | How demanding the quality floor is; see [Targets and presets](#targets-and-presets). |
| `--min-quality SCORE` | from the preset | Lowest score, 0 to 100, a candidate may have, on the scale of the metric in use. Replaces the preset's floor. A `--preset` given on the command line also replaces a `min_quality` from a configuration file. |
| `--quality-slack POINTS` | `0` | Points a score may fall short of the floor and still pass. |
| `--metric auto\|ssimulacra2\|ssimulacra2-inprocess\|builtin` | `auto` | The score that decides; see [Quality scoring](#quality-scoring). `ssimulacra2-inprocess` computes the SSIMULACRA2 score inside the program, needs no `ssimulacra2` tool, and is not the choice of `auto`. |
| `--speed fast\|balanced\|exhaustive` | `balanced` | How many settings are tried per format; see [How the search works](#how-the-search-works). |
| `--search staged\|predicted` | `staged` | Experimental. `predicted` starts each lossy format at a quality estimated from the quality floor, then searches the range that the start's measured error allows, in fewer rounds than the coarse grid of `staged`. It applies to photographs, with `--speed balanced` or `exhaustive`, SSIMULACRA2 scoring and candidates evaluated in parallel; any other image, and a format whose measurements do not fit the estimate, is searched as `staged`. |
| `--allow-larger[=BOOL]` | false | Accept an output larger than the original. |

### Formats

| Option | Default | Meaning |
|---|---|---|
| `--target web\|archive\|smallest\|max-compat` | `web` | What the output is for, which decides which formats may be used. |
| `--formats LIST` | `jpeg,webp,avif,png` | Comma-separated formats to try: `jpeg` (or `jpg`), `webp`, `avif`, `jxl`, `png`. The target is applied to this list afterwards. |

To leave AVIF out, name the others: `--formats jpeg,webp,png`. To add JPEG XL to
the `web` target, name it: `--formats jpeg,webp,avif,png,jxl`. The list replaces the
default and any `[codecs]` section of a configuration file, so name every format
you want.

A set of formats that the target empties (for example `--formats webp --target
max-compat`) is rejected with a message that names the conflict.

### Metadata and alpha

| Option | Default | Meaning |
|---|---|---|
| `--metadata keep\|safe\|strip` | `safe` | What to keep of the image's metadata; see [Colour, metadata and alpha](#colour-metadata-and-alpha). |
| `--alpha preserve\|flatten\|auto` | `preserve` | What to do with transparency. |
| `--background #rrggbb` | `#ffffff` | The colour flattened transparency is drawn over, with `--alpha flatten`. |

### Limits and budgets

| Option | Default | Meaning |
|---|---|---|
| `--workers N` | logical processors | Images and candidates worked on at once, clamped to 1 through 256. A cap on the threads of work for the whole run; see [Workers, memory and backpressure](#workers-memory-and-backpressure). |
| `--candidate-timeout-seconds N` | 30 (`fast`), 120 (`balanced`), 300 (`exhaustive`) | Longest an encoder, decoder or metric process may run. |
| `--max-image-time-seconds N` | none | Stop searching an image after this long and keep the best result found. |
| `--max-candidate-attempts N` | none | Most candidates tried for one image. Skipped candidates do not count. |
| `--max-temp-bytes N` | none | Most temporary-file space one image may use; with it, the candidates of an image run one at a time. |
| `--max-run-temp-bytes N` | none | Soft limit on the temporary files of the whole run: work waits for room above three quarters of it. |
| `--max-memory-bytes N` | two thirds of physical memory | Memory the run plans on using, shared between images, candidates and `ssimulacra2` processes. |
| `--max-pixels N` | 100,000,000 | Largest image accepted, in pixels (width times height). |
| `--max-input-bytes N` | 2 GiB | Largest input file accepted. Applies to the inputs you name, not to the program's own temporary files. |
| `--max-width N`, `--max-height N` | 100,000 | Largest dimensions accepted. |
| `--max-decoded-bytes N` | 2 GiB | Most memory the decoded image may take. |
| `--max-metadata-bytes N` | 16 MiB | Most metadata (EXIF, XMP, colour profile) an image may carry. |
| `--max-bits-per-pixel N` | 64 | Most bits per pixel a source may have. |

An image over a limit is refused before its pixels are allocated. A 199.8
megapixel photo, for example, needs `--max-pixels 200000000`.

### Cache

| Option | Default | Meaning |
|---|---|---|
| `--cache` | off | Reuse the encoded files and scores of earlier runs, and keep this run's. |
| `--cache-dir DIR` | `%LOCALAPPDATA%\smart-compress\cache` | Folder of the cache; turns the cache on. |
| `--cache-max-bytes N` | none | Keep the cache under this size by deleting the oldest entries when a run starts; turns the cache on. |
| `--cache-max-age-days N` | none | Delete entries older than this when a run starts; turns the cache on. |
| `--cache-prune` | off | Apply the two limits above to the cache now and exit, without processing any image. |

### GPU

| Option | Default | Meaning |
|---|---|---|
| `--cpu-only` | off | Score on the CPU only. Conflicts with `--gpu-required`. |
| `--gpu-required` | off | Fail instead of falling back to the CPU when the GPU cannot be used. |
| `--gpu-adapter-name TEXT` | best hardware GPU | Use the GPU whose name contains this text, ignoring case. Naming a software adapter selects it. |
| `--gpu-adapter-vendor ID`, `--gpu-adapter-device ID` | none | Use the GPU with this PCI vendor or device id (decimal). |
| `--verify-gpu` | off | Score each candidate on the CPU as well and report where the two differ. Slow; for checking the GPU. |

### Reports

| Option | Default | Meaning |
|---|---|---|
| `--report text\|json\|markdown` | `text` | How the report is written. |
| `--report-out FILE` | terminal | Write the report to a file. It is rewritten every few seconds while the run goes on, so an interrupted run leaves what it finished. |
| `--resume-report FILE` | none | Skip the images already finished in this JSON report from an earlier run with the same settings. Needs `--report-out`. |
| `--validate-report FILE` | none | Check that a JSON report is readable and consistent, then exit. |
| `--verbose` | off | Try every planned candidate: turns off the shortcuts that skip candidates that cannot win. Slower. |
| `--no-progress` | off | Do not show the progress bars (they appear only when stderr is a terminal). |

### Control panel

| Option | Default | Meaning |
|---|---|---|
| `--gui` | off | Serve the control panel and open it in the default browser. |
| `--gui-window` | off | Show the control panel in its own window and exit when it closes. |
| `--gui-no-open` | off | Serve the control panel without opening anything. |
| `--gui-port PORT` | `7878` | Port to try first on this computer. |

## Targets and presets

A preset sets the quality floor and how much extra size is accepted for a more
widely supported format. The floor is a score on the scale of the metric in use
(the second column applies when SSIMULACRA2 decides; the third when the built-in
score does, see [Quality scoring](#quality-scoring)).

| Preset | Floor, SSIMULACRA2 | Floor, built-in score | Compatibility allowance |
|---|---:|---:|---:|
| `strict` | 90 | 88 | 10% |
| `balanced` | 80 | 73 | 5% |
| `aggressive` | 70 | 68 | 0% |

Among the candidates that pass the floor and are smaller than the original, the
smallest is found first. Every passing candidate within the allowance of that
size is then a contender, and the most compatible format wins (JPEG and PNG, then
WebP, then AVIF, then JPEG XL); ties go to the smaller file. The allowance is
measured against the smallest candidate only, so it cannot chain into a much
larger winner. The `smallest` target sets the allowance to zero.

Targets decide which formats may be tried:

| Target | Formats |
|---|---|
| `web` | The `--formats` list (default JPEG, WebP, AVIF, PNG). |
| `archive` | The list plus JPEG XL, with JPEG removed. |
| `smallest` | The list plus JPEG XL, and no compatibility allowance. |
| `max-compat` | Only JPEG and PNG, the formats every program opens. |

A format is skipped, with the reason in the report, when it cannot represent the
image (for example JPEG for an image with transparency under `--alpha preserve`)
or when its tools are not installed.

## Quality scoring

A candidate is accepted when its score plus `--quality-slack` reaches the floor.
`--metric` picks the score:

- `auto` (default): SSIMULACRA2 when the `ssimulacra2` tool is found, otherwise
  the built-in score.
- `ssimulacra2`: SSIMULACRA2, failing the image if the tool is missing. It was
  designed to track human opinion scores. Its bands are 90 visually lossless, 80
  no visible difference side by side at full size, 70 high quality, 50 medium.
- `ssimulacra2-inprocess` (experimental): the same metric computed inside the
  program by the `fast-ssim2` crate, with the original prepared once per image.
  It needs no tool, scores a candidate two to five times faster and uses less
  memory. On 545 candidates from real runs its scores were within 0.32 points
  of the tool's (0.01 on average) and every winner was the same file, but on
  synthetic test images differences of up to 1.2 points were measured and are
  not yet explained, so `auto` does not choose it.
- `builtin`: the built-in score, the lowest of an RGB agreement (twice the PSNR
  over the colour channels, capped at 100), an edge agreement of the same kind
  and, for images with transparency, an alpha agreement and a composite
  agreement. It is faster and can run on the GPU, but PSNR ranks codecs
  differently from human viewers (on two phone photos, a JPEG and an AVIF with
  the same SSIMULACRA2 score were 19 points apart on it; see
  [benchmarking](benchmarking.md#other-findings)), so use it only when the tool is
  unavailable. The report fields `rgb` and `edge` hold its two main components.

Preset floors follow the metric (table above). The built-in floors were set to
where the same candidates passed on a small number of photos; treat them as
approximate. `--min-quality` replaces the floor on whichever scale is active.
Reports record `quality_metric` (`ssimulacra2` or `builtin`) and `min_quality`
per image.

An image with a side under 8 px cannot be scored by `ssimulacra2`. It is scored
with the built-in score and that score's floor, and the report says so in a
warning.

Images above 16 megapixels are scored by SSIMULACRA2 on a 3 x 3 mosaic of
full-resolution 1024 px samples taken from a regular grid, because a reduced copy
hides damage that is visible at full size. A small busy region between the
samples can still be missed. The built-in score uses a 2048 px reduction above 16
megapixels; every backend reduces with the same area-average filter, so CPU and
GPU scores agree to within 0.01.

Transparent images must pass two more checks whatever the main score says. The
alpha channel as a whole is compared, and a local check looks only at the 16 px
tiles where the original's alpha varies and takes the worst one: the original
colour is drawn with the original alpha and with the candidate's alpha over
white, black and both phases of a checkerboard, and the difference is scored. A
thin halo or fringe changes too few pixels to move a whole-image score, but it
fails this check. The result is reported per candidate as `alpha_edge`. The
built-in score ignores the colour stored under fully transparent pixels (scoring
version `score-v10`); that colour still reaches the encoders unchanged.

A candidate whose encoding, decoding or scoring failed (a tool error, a timeout,
an unreadable output) has status `Failed` and gives no verdict on quality. If no
candidate can be accepted and some failed, the image fails with an error that
says how many, instead of being reported as `kept`.

## Colour, metadata and alpha

Nothing is colour-converted. The metric compares channel values, and a file that
carries the same numbers under a different (or no) profile would score well
while looking different, so a candidate is only trusted when it keeps the
source's colour interpretation. Each output therefore carries the source's
profile, or the image or candidate is refused with the reason in the report. The
metric compares decoded channel values as sRGB-relative values (the
`srgb-relative-v1` boundary named in reports). Full ICC conversion is not
implemented, so colour-critical workflows should keep their source profiles and
read the report.

An embedded profile is carried over only when it can describe the pixels being
written. A profile whose header names a different colour model than the decoded
image (a grey profile on a colour image, a colour or CMYK profile on a grey or
RGB image), or data that is not an ICC profile at all, is dropped and the image's
`warnings` list it. A PNG whose profile cannot be read, or is larger than
`--max-metadata-bytes`, is refused. Metadata is size-checked before it is
inflated, and an AVIF source is checked against the size limits before its
decoder runs.

Grey images stay single-channel in PNG, JPEG, AVIF and JPEG XL output so a grey
profile still fits. WebP always stores colour: a grey profile that is plain sRGB
is not attached there and the candidate carries a warning, and any other grey
profile makes the WebP candidate fail with that reason, since the greys would
change. Four-component (CMYK) JPEGs are detected from their frame header,
reported with an original colour type of `Cmyk8`, converted to RGB by the decoder
without a colour-managed transform, and flagged with a warning. One that carries a
CMYK ICC profile is refused instead: viewers show it through the profile, so the
unmanaged conversion would change its colours.

`--metadata` chooses what else is kept:

- `keep` preserves the ICC profile and sanitized EXIF bytes.
- `safe` (default) preserves the ICC profile and drops EXIF and everything else.
- `strip` removes everything except a colour profile that is not plain sRGB.

A profile that is not plain sRGB (Display P3, Adobe RGB, a gamma 1.8 or linear
curve, a grey curve other than sRGB) says what the pixel values mean, so it is
colour data rather than metadata: `strip` keeps it and the image's `warnings` say
so. A profile counts as plain sRGB only when it is a matrix/TRC profile with sRGB
primaries (within 0.4% of XYZ) and curves within 0.01 of the sRGB curve, or a
grey profile with such a curve, so gamma 2.2 stand-ins count; profiles with
lookup tables, damaged profiles and anything else are kept. Under `safe` and
`strip`, external encoders always read a normalized copy of the decoded pixels,
never the source file, so they cannot copy EXIF or XMP across. Orientation is
applied to the pixels and removed from retained EXIF. XMP, GPS, thumbnails,
comments and HDR transfer functions are not claimed to be preserved.

Colour that a source declares without an ICC profile is not carried by any
output either. A PNG with a `cICP` chunk other than sRGB (Display P3, BT.2020, PQ
and HLG HDR), or with `gAMA` or `cHRM` chunks that are not sRGB, is refused with
an error; so is an AVIF whose colour is a non-sRGB CICP triple, which `avifdec`
reports to the pipeline as a `cICP` chunk. BT.709 primaries with the BT.709 curve
(code points 1/1) are the exception: viewers show them as sRGB, so the image is
processed with a warning. An embedded ICC profile takes precedence and is
carried. The external metric tools are always given two PNGs without profiles, so
they convert neither.

Not covered: EXIF colour-space flags in untagged JPEGs, HDR gain maps, and
rendering intent. Profile carriage is pinned by tests against the tool versions
that `tools/fetch-tools.ps1` installs; it is not re-checked at run time.

Alpha: `--alpha preserve` keeps transparency, even if no pixel is transparent,
and never offers JPEG for an image with an alpha channel. `--alpha flatten
--background #rrggbb` composites onto the colour first and drops the channel.
`--alpha auto` drops the alpha channel only when every pixel is fully opaque (so
JPEG can be offered) and otherwise behaves like `preserve`.

## Inputs

PNG, JPEG, WebP and AVIF are the supported input formats. The content is checked
against the extension. Corrupt files, unsupported formats (SVG, GIF, BMP, TIFF,
JPEG XL), animated WebP, APNG and AVIF, truncated JPEGs and files whose content
does not match their extension are refused with an error before any
optimization; the program never silently processes only one frame of an
animation. The dimensions of the image are never changed.

The content classifier labels each image `photo`, `screenshot`,
`transparent-asset`, `simple-graphic` or `unknown` in the report (shown as
`Photo`, `Screenshot` and so on in the text report).
The label comes from colour, edge and flat-area statistics of a 512 px overview
and only names the image in the report; it does not change the search. A
sharply textured greyscale photograph can be labelled `screenshot`.

## How the search works

Planned candidates for an image are evaluated concurrently, in the stages below,
bounded by `--workers` and by the memory the image needs; decisions still run in
plan order. Setting either temporary-disk budget (`--max-temp-bytes`,
`--max-run-temp-bytes`) makes the candidates of one image run one at a time
(other images still run in parallel). Images too large for more than one
candidate at a time in the memory limit are also evaluated one at a time; that
path walks each format from its lowest quality upward and stops the format once a
quality passes, is dominated, or is already too large, since higher qualities
only produce larger files. With the cache enabled, candidates that are not
smaller than the original are not stored.

The search runs in passes. The coarse pass encodes a few qualities per format
(`fast`: 88, 70; `balanced`: 92, 78, 60; `exhaustive`: 96, 88, 80, 70, 60, 48).
AVIF, the slowest encoder, runs this pass at a cheaper effort (`avifenc --speed`
8 for `balanced`, 6 for `exhaustive`); the other formats use their final effort.
That brackets the quality floor for each format: the upper end is the lowest
quality that passed or was already larger than the original, the lower end the
highest one that failed.

When candidates run concurrently the coarse pass is staged per format, so that
only the qualities that can place the bracket are encoded. The first stage runs
the middle quality of every lossy format (78 for `balanced`, 70 for `fast` and
`exhaustive`). If it passed, the higher qualities are assumed to pass too and only
the lower ones run; if it failed, only the higher ones run and the lower ones are
assumed to fail. Lossless and palette candidates run with the second stage. The
bracket ends, and so the refinement, are the same as when the whole grid runs,
provided score and size fall with quality. A format whose failing middle quality
is already no smaller than the best file so far is not continued (its higher
qualities cannot win), and a lossless or palette candidate that is no smaller
than the best is skipped before it is decoded. When the upper bracket end was too
large to be scored, the next quality below the middle one runs as well, as the
crossing is then estimated from two failures. Each quality left out is in the
report as skipped, with the reason `assumed to pass`, `assumed to fail` or
`cannot win`, and does not count against `--max-candidate-attempts`. `--verbose`
evaluates the whole grid in one stage instead.

Lossless candidates are re-encodes of the decoded pixels (JPEG XL is run with
`--lossless_jpeg=0`, so a JPEG is never transcoded), and for a photographic JPEG
source they are several times the size of the source. When the source is a JPEG
and more than one lossless-capable format is selected, lossless WebP is encoded
with the first stage and acts as a probe: if its file is more than three times
the largest file that could still win (the original size, unless `--allow-larger`
is set, or the best accepted candidate so far plus the compatibility allowance,
whichever is smaller), the lossless AVIF, JPEG XL and PNG candidates are skipped
with a reason starting `cannot win: lossless WebP`. Three times is the headroom
over the 1.65 that the probe was at most of the smallest lossless encoder on 63
JPEG sources (measured during development; the list of those sources is not kept
in the repository), so the rule rests on measurements, not on a proof. The probe
itself is always encoded. Sources that may be lossless (PNG, WebP, AVIF) run every
lossless candidate, as does `--verbose`, a run with `--allow-larger` where
nothing has passed the quality floor yet, and a format list without WebP or
without a second lossless format.

The staging relies on score and file size falling with quality within a format.
When the evaluated qualities of a format show otherwise (a lower quality passing
while a higher one fails, or a lower quality producing a larger file at the same
effort), the image report lists a `non-monotonic behaviour detected` warning. The
qualities the staging had skipped for that format are then evaluated, and the
refinement also probes the qualities next to the lower point of the inversion,
within the usual attempt and time budgets.

Refinement rounds follow (two for `balanced`, three for `exhaustive`, none for
`fast`). Each round estimates where every format crosses the floor by
interpolating score against quality, ranks the formats by their estimated size at
that point, and probes around the crossing at the final effort (`avifenc --speed`
6 or 4, `cwebp -m` 4 or 6, `cjxl -e` 5 or 7), only for formats within 25% of the
smallest estimate. The first round spreads its probes two quality steps apart;
later rounds close in one step at a time. Every candidate that can be accepted is
still encoded, decoded and scored, so the floor is enforced exactly; what the
ranking can cost is a slightly larger winner when a format outside the margin
would have won.

PNG candidates are the built-in encoder, optionally re-packed by `oxipng`, plus
`pngquant` palette candidates, which are scored like any lossy candidate. Both
tools are skipped when metadata has to be preserved. JPEG candidates use `cjpegli`
when it is installed and no EXIF has to be written; the ICC profile is carried
over. If `cjpegli` rejects an input (for example one with a damaged profile) the
built-in encoder is used and the candidate records the reason as a warning. JPEG
XL "lossless" candidates are encoded with distance 0.

JPEG XL candidates are decoded to an uncompressed PAM file and read directly;
AVIF candidates are decoded to an uncompressed PNG, the only lossless format
`avifdec` writes, which is deleted as soon as its pixels are loaded. Before a
decoder runs, the candidate's header is read in-process from the first 64 KiB of
the file: the AVIF boxes, the JPEG XL size header and image metadata, and the
image header for JPEG, PNG and WebP. A candidate whose size, frame count or alpha
channel cannot match the image that was encoded is rejected with a
`candidate header ...` reason and no decoder process is started. A header that
cannot be read, or whose boxes lie beyond those 64 KiB, is left to the decoder.

## Workers, memory and backpressure

`--workers N` (default: the number of logical processors) is a budget of N
threads of work for the whole run, and nothing runs more threads than that at any
moment. An external encoder, decoder or metric takes as many units as the threads
it is told to use; in-process encoding, decoding and the built-in CPU metric take
the units they use; a GPU submission takes one. Work that does not fit waits its
turn, oldest request first, and a request larger than the whole budget runs
alone. Images are processed by up to N file workers, which hold nothing while
they wait, and their candidates run on N threads of their own. An encoder gets at
most one thread per 3 megapixels of the image, and no more than its share of N,
so small images are not slowed by threads that only wait on each other.

Memory is planned the same way. The run plans on two thirds of the physical
memory, or `--max-memory-bytes` (`max_memory_bytes` under `[performance]` in the
configuration file), and splits it three eighths for images, three eighths for
candidates and a quarter for the external `ssimulacra2` processes. Before an
image is decoded it reserves about 114 bytes per pixel, held until its report is
done; each candidate reserves about 45 bytes per pixel while it runs; each
`ssimulacra2` process reserves about 150 bytes per pixel. For a 12 megapixel
photo that is about 1.4 GB for the image, 0.54 GB per candidate in flight and
1.8 GB per `ssimulacra2` process. The figures are fitted to the peak working set
of the whole process tree and documented in
`crates/smart-compress-core/src/memory.rs`.

When a reservation does not fit in its share, the work waits for earlier work to
finish instead of starting; one larger than its whole share runs alone, and an
image estimated above the entire limit still runs, with a warning. How many
candidates of an image run at once is the number that fits in the candidates'
share, up to `--workers`; when only one fits, the image takes the one-at-a-time
path. Where the physical memory cannot be read (see
[Platform notes](#platform-notes)) the limit is 8 GiB.

With `--max-run-temp-bytes`, new candidates also wait while the temporary files of
the run use more than three quarters of that budget and other candidates are
still running, since only they free room by finishing; when nothing else is
running the candidate goes ahead, and the hard budgets still end the search if
they are exceeded. The time an image's work spent waiting for memory or
temporary-disk room is reported as `backpressure_wait_ms` in the image's timings
(summed over its waits, so several candidates waiting together add up). The
per-image time limit and cancellation also end a wait.

## Reports, resume and cache

`--report text` (default) prints a table of every candidate and a summary;
`--report markdown` writes the same information as Markdown; `--report json`
writes the machine-readable form described next.

### JSON report, schema version 7

The top-level object has these members:

| Member | Content |
|---|---|
| `schema_version` | `7`. `--validate-report` and `--resume-report` refuse a report with no version or from a newer schema. A report of an older schema is read, and members this version no longer writes are ignored; `--resume-report` skips it, though, because the configuration fingerprint changed. |
| `metric_color_space` | The colour space the built-in metric compares in (`srgb-relative-v1`). |
| `tools` | The external tools found at the start of the run, by name, each with `name`, `version`, `enabled` and `install_hint` (only when missing). Paths are not recorded. |
| `gpu` | The GPU detection: `status` (`disabled`, `enabled`, `unavailable` or `required-unavailable`), the requested adapter, `selected_adapter` and `available_adapters` (name, vendor, device, type, backend, limits, whether timestamp queries work) and `warnings`. |
| `reports` | One entry per image, in input order. |
| `failures` | The number of images with outcome `failed`. |
| `warnings` | Warnings about the run as a whole, such as a folder that could not be read. |
| `stage_totals` | Milliseconds summed over all images: decode, normalize, search, encode, candidate decode, score, cache, external metric, output write, total, and `gpu_timestamp_ns` (GPU-side time from timestamp queries, when the adapter supports them). |
| `manifest` | The run: `run_id`, `status` (`partial` or `complete`), start and completion times, `expected_inputs`, `completed_inputs`, `config_fingerprint` and `resume_decision`. |

Each entry of `reports` describes one image:

| Member | Content |
|---|---|
| `input`, `input_identity`, `output` | The paths as given on the command line, and a hash of the input. `output` is `null` when nothing was written. |
| `original_size`, `saved_bytes` | Bytes of the original, and bytes saved by the winner. |
| `outcome` | `optimized`, `kept`, `skipped`, `dry-run` or `failed`. |
| `probe` | What was read from the image: format, dimensions, colour types, alpha, `metadata_size` (ICC plus EXIF bytes), `icc_profile_size`, content class and the measures behind it. |
| `quality_metric`, `min_quality`, `scoring_version` | The score that gated acceptance (`ssimulacra2` or `builtin`), the floor used, and the version of the scoring code. |
| `compute_backend`, `gpu_work`, `gpu_preprocess_backend` | Where the built-in score ran, and the estimated GPU buffer sizes. |
| `timings` | Milliseconds per stage: open, decode, orientation, probe, normalize, cache lookup, external metric preparation, search, output write, total, and `backpressure_wait_ms`. All are unsigned 64-bit. |
| `candidates` | Every candidate, described below. |
| `winner_index` | Index into `candidates` of the winner, or `null`. |
| `errors`, `warnings` | Why the image failed, and things done differently from what was asked that did not fail it. |
| `search_stop_reason`, `dry_run` | Set when a time or attempt budget ended the search; whether this was a dry run. |
| `temporary_bytes_current`, `temporary_bytes_peak`, `temporary_budget_bytes` | Temporary-disk use. |

Each candidate records:

| Member | Content |
|---|---|
| `codec`, `settings` | The format and its settings: `{"Quality": {"quality": 78, "effort": 4}}`, `"Lossless"` or `{"Palette": {"colors": 64}}`. |
| `size_bytes`, `backend` | The encoded size and the encoder that made it. |
| `status` | `"Accepted"`, or an object with a reason: `{"Rejected": "..."}` (scored below the floor or too large to be worth scoring), `{"Skipped": "..."}` (not evaluated) or `{"Failed": "..."}` (no verdict). |
| `influence` | `first-passing`, `improved-winner`, `rejected`, `skipped` or `failed`. |
| `quality` | The scores: `rgb`, `edge`, `alpha`, `composite_min`, `alpha_edge`, `ssimulacra2` and `overall` (the one compared with the floor). Members that do not apply are `null`. |
| `score_backend`, `quality_source`, `cache_hit_kind` | Where the score was computed (`gpu`, `cpu`, `cpu-fallback` with the reason, or `ssimulacra2-inprocess`), whether it is `fresh` or from the `score-cache`, and which cache entry hit (`encoded-artifact` or `score-metadata`). |
| `encode_ms`, `decode_ms`, `score_ms`, `external_metric_ms`, `total_ms` and the GPU stage timings | Milliseconds per stage; `gpu_timestamp_ns` is GPU-side time from timestamp queries when supported. |
| `bytes` | Counters for uploads, temporary output and decoded buffers. |
| `process_failure` | For a failed external process: the program's file name, kind, exit code, message and elapsed time. |
| `gpu_verification` | With `--verify-gpu`: the tolerance, CPU and GPU `overall`, and `component_deltas`, a map from component name to difference. |
| `warnings` | Notes about this candidate. |

Input and output paths appear as they were given, so read a report before sharing
it.

### Resume

`--report-out` writes the report atomically through a temporary file in the same
folder, and again after each finished image, with the run's status `partial`
until the end. A run interrupted by Ctrl+C or a crash therefore leaves a valid
partial report, and a report file that is empty or truncated is never treated as
a finished run. `--resume-report` reuses the finished images of such a report when
the settings, including the dry-run and output settings, are the same and the
outputs it claims still exist; otherwise it says in `manifest.resume_decision`
why it started over. With `--report-out`, images are processed in parallel and the
report stays in input order.

### Cache

Caching is off until `--cache`, `--cache-dir` or one of the limits is given. It
stores each encoded candidate under a name derived from the encoder's source
pixels, the format and settings, the alpha state and the installed tools' names
and versions, so replacing a tool invalidates its entries. Scores are stored
separately, keyed also by the scoring mode, together with the BLAKE3 hash of the
candidate they belong to. A cached candidate is copied out and verified against
its stored BLAKE3 hash before use, and is decoded and scored again unless a
matching score entry exists.

The default cache folder is per user: `%LOCALAPPDATA%\smart-compress\cache` on
Windows, `$XDG_CACHE_HOME/smart-compress` or `~/.cache/smart-compress` elsewhere,
and never the working directory. A new cache folder gets a marker file.
`--cache-prune` and the start-up limits only delete files laid out like cache
entries, and refuse a non-empty folder that has no marker, so a mistaken
`--cache-dir` is left alone.

A person who can write to the cache folder can also recompute the stored hashes
and so alter a stored score; keep the cache folder, like the `tools` folder, in a
location only you can write to.

### Files, processes and permissions

External programs are started with argument lists, never shell strings, with
timeouts, bounded draining of their output, the global work budget described
above and, on Windows, termination of the whole process tree. Candidates are
written to the operating system's temporary folder, cache entries under the cache
folder with hash-derived names, and final outputs to the output path. Outputs are
published through a temporary file in the same folder and an atomic rename, so
Windows gives them the destination folder's permissions rather than broader ones.
Keep the output and cache folders on trusted local drives and do not point them at
shared writable folders.

## GPU

GPU scoring is on by default and uses `wgpu`. It computes the built-in score only;
SSIMULACRA2 always runs on the CPU in its own process, so with SSIMULACRA2
deciding, the GPU affects only a small part of the run (see
[benchmarking](benchmarking.md)). The adapter is chosen once per run: the best
hardware GPU, or the one selected by `--gpu-adapter-name`, `--gpu-adapter-vendor`
or `--gpu-adapter-device`. A software adapter (such as Windows' Basic Render
Driver) is used only when named explicitly; otherwise scoring runs on the CPU and
the report says the GPU is unavailable. The report names the adapter the compute
device was created from.

If no adapter can be created, the run continues on the CPU and the report says
why; `--gpu-required` turns any such fallback, at any stage, into a failure.
`--cpu-only` disables the GPU, and `--verify-gpu` scores each candidate on both
and records the per-component differences.

GPU preprocessing of the original (`prefer_gpu_preprocess`) is experimental and
off by default. It applies only to images above 16 megapixels and is reachable
only from the configuration file and from `smart-compress-benchmark --modes
cpu,gpu-prepared`.

## External tools

Native PNG and JPEG candidates are always available. Everything else needs an
external program, installed by `tools/fetch-tools.ps1` (see the
[README](../README.md#install)). At start-up each tool is looked up by name,
in this order:

1. on `PATH`, using only absolute entries (an empty or relative entry would mean
   the working directory);
2. in folders next to the executable: `tools`, `tools/bin`, `tools/libjxl/bin`,
   `tools/libavif`, `tools/libwebp/libwebp-1.6.0-windows-x64/bin` and
   `tools/pngquant/pngquant`;
3. in the same folders of this repository's workspace, when the executable runs
   from the workspace's `target` folder.

The working directory and parent folders are never searched. A format whose tools
are missing is not planned, and the report lists the tools under "Missing
optional tools" with an install hint.

| Tool | Used for | If missing |
|---|---|---|
| `cwebp` | WebP candidates | WebP is not tried |
| `webpmux` | Writing metadata into WebP candidates | Metadata cannot be kept in WebP |
| `avifenc` and `avifdec` | AVIF candidates and decoding them | AVIF is not tried; both are needed |
| `cjxl` and `djxl` | JPEG XL candidates and decoding them | JPEG XL is not tried; both are needed |
| `cjpegli` | JPEG candidates | The built-in JPEG encoder is used |
| `oxipng` | Re-packing PNG candidates | PNG candidates are not re-packed |
| `pngquant` | Palette PNG candidates | No palette candidates |
| `ssimulacra2` | The deciding score | The built-in score decides, with lower floors |

## Control panel

`smart-compress-app.exe` (double-click), or `gui.cmd` in the repository root, opens
the panel in its own window and exits when the window closes. It starts
`smart-compress --gui --gui-window`, so `smart-compress.exe` must sit next to it.
The window is Microsoft Edge in app mode, started from its installed path, with
the default browser as the fallback. `smart-compress --gui` serves the same page
to the default browser and stops with Ctrl+C. Arguments given to
`smart-compress-app.exe` are passed on, for example `--gui-port 9000`.

The panel listens on `127.0.0.1` only, on port 7878 or the next free one (up to
20 further), or from `--gui-port`. Every request needs the session's random token
(the link the program prints carries it), a `Host` header that matches the
address, and, for requests that send JSON, the `application/json` content type;
at most 16 connections are served at once. The page is compiled into the
executable and drives the same argument parser and optimizer as the command line,
in the same process: every setting maps to a flag, and the "Equivalent command"
section shows the exact invocation. The server reads image files only for
previews.

Images are added by dropping them on the window, through native file dialogs, or
by pasting paths; folders always include their subfolders. Network (UNC) and
device paths are refused: use a drive letter. A dropped file is copied to a
temporary folder that belongs to that panel instance and is removed when it
exits (browsers hide the original path), so its result goes to
`Downloads\smart-compress` in the user profile unless an output folder is chosen.
One run executes at a time. Cancel stops running encoders and writes nothing for
unfinished images.

## Configuration file

`--config FILE` reads a TOML file. Built-in defaults apply first, then the file,
then options given on the command line. A key the program does not know is an
error that names the key and its section. The `[gpu]`, `[cache]` and `[metrics]`
sections replace the whole section when present, so name every key you want
changed from its default; `[cache]` needs `enabled`. A `[codecs]` section
replaces the format list (`--formats` and the target still apply after it).

This file lists every section and key with its default value. Keys that start
with `#` are optional and have no default.

```toml
[defaults]
preset = "balanced"          # strict | balanced | aggressive
target = "web"               # web | archive | smallest | max-compat
# min_quality = 80.0         # replaces the preset's floor
quality_slack = 0.0
report = "text"              # text | json | markdown
# report_out = "report.json"
metadata = "safe"            # keep | safe | strip
alpha = "preserve"           # preserve | flatten | auto
background = "#ffffff"       # used only when alpha = "flatten"
speed = "balanced"           # fast | balanced | exhaustive
search = "staged"            # staged | predicted (experimental)
overwrite = false            # true replaces existing outputs
protect_inputs = true
# out_dir = "optimized"
recursive = false
allow_larger = false
max_pixels = 100000000
max_input_bytes = 2147483648
max_width = 100000
max_height = 100000
max_decoded_bytes = 2147483648
max_metadata_bytes = 16777216
max_bits_per_pixel = 64

[codecs]                     # each defaults to true, except jxl
jpeg = true
webp = true
avif = true
png = true
jxl = false

[performance]
# workers = 4                # default: the number of logical processors
# candidate_timeout_seconds = 120
# max_image_time_seconds = 600
# max_candidate_attempts = 60
# max_temp_bytes = 1073741824
# max_run_temp_bytes = 4294967296
# max_memory_bytes = 8589934592

[gpu]
enabled = true
required = false
# adapter_name = "radeon"
# adapter_vendor = 4098
# adapter_device = 29822
prefer_gpu_preprocess = false  # experimental
max_vram_bytes = "auto"        # "auto" or a number of bytes
verify_gpu = false

[cache]
enabled = false
# dir = "cache"              # default: the per-user cache folder
# max_bytes = 1073741824
# max_age_days = 30

[metrics]
primary = "auto"             # auto | ssimulacra2 | builtin
```

## Platform notes

Windows 10 and 11 on x64 is the platform the program is developed and tested on,
and the only one CI builds on. The code compiles conditionally on Windows in a
few places, and has not been built or run elsewhere. A reader porting it should
know what changes:

- Tool lookup uses `PATH` and `PATHEXT` on Windows and plain names elsewhere. The
  folders it searches under `tools` are the layouts of the Windows packages.
- Killing a timed-out or cancelled encoder's whole process tree, and the Ctrl+C
  handler that cancels a run cleanly, are Windows-only.
- Physical memory is read only on Windows. Elsewhere the memory limit is a fixed
  8 GiB unless `--max-memory-bytes` is given.
- The control panel's window mode, native file dialogs, "open folder" and the
  `Downloads` default for dropped files are Windows-only. Elsewhere the server
  starts and prints its address but does not open a browser; paste paths instead
  of using dialogs.
- `smart-compress-app.exe`, `gui.cmd` and the scripts under `tools` are Windows
  programs and scripts.
- Some tests start `powershell.exe` and are compiled only on Windows.

## Environment variables

| Variable | Effect |
|---|---|
| `PATH`, `PATHEXT` | Used to find the [external tools](#external-tools). |
| `LOCALAPPDATA` | Where the default cache folder is on Windows. |
| `XDG_CACHE_HOME`, `HOME` | Where the default cache folder is elsewhere. |
| `TEMP`, `TMP` | The temporary folder for candidates and dropped files (Windows). |
| `USERPROFILE` | Its `Downloads` folder is where the control panel puts the results of dropped files. |
| `ProgramFiles`, `ProgramFiles(x86)` | Where the control panel looks for Microsoft Edge. |
| `SMART_COMPRESS_REQUIRE_TOOLS` | Test suite only: turns each test skipped for a missing tool into a failure. Any value enables it. |
| `SMART_COMPRESS_REQUIRE_GPU` | Test suite only: turns each GPU test skip into a failure. Any value enables it. |
| `SMART_COMPRESS_LARGE_IMAGE` | Test suite only: the path of the large image used by the ignored 199.8 megapixel GPU test. |
| `UPDATE_SNAPSHOTS` | Test suite only: rewrites the pinned report renderings; see [maintainers.md](maintainers.md#snapshots). |

## Troubleshooting

- **"is a directory; pass --recursive to process it"**: folders need `--recursive`.
- **The image is `kept` and the report says "nothing scored at least ...":** no
  candidate was both smaller than the original and at or above the floor. Nothing
  is written and the exit status stays 0. Lower the preset (`--preset
  aggressive`), allow more formats (`--target smallest`), or accept that the file
  is already well compressed. The strict preset finds no smaller file for any of
  the six larger sample photos.
- **"output already exists, left as it is":** the image was skipped; pass
  `--overwrite` or another `--out-dir`.
- **An image is refused for its size:** raise `--max-pixels` or the other limits,
  within what the memory allows.
- **A format never appears:** check "Missing optional tools" in the report; run
  `powershell -ExecutionPolicy Bypass -File tools/fetch-tools.ps1`. The default
  `web` target does not try JPEG XL; add it with `--formats jpeg,webp,avif,png,jxl`.
- **Colour-related refusals:** the report names the PNG chunk, ICC profile or CICP
  triple that could not be carried; see
  [Colour, metadata and alpha](#colour-metadata-and-alpha).
- **The control panel opens in a browser tab, not a window:** Microsoft Edge was
  not found at its installed path; the tab behaves identically.
