# Third-party software

smart-compress runs the command-line programs below as separate processes. They
are not part of this repository and are not covered by its licence. Nothing is
linked into smart-compress.

`tools/fetch-tools.ps1` downloads the official Windows releases from the
upstream addresses below, checks each download against a pinned SHA-256, and
installs into `tools/` only the programs listed here together with the licence
files of each package. The programs are not stored in this repository.

| Package (programs installed) | Version | Licence | Upstream |
|---|---|---|---|
| libjxl (`cjxl`, `djxl`, `cjpegli`, `ssimulacra2`, `jxlinfo`) | 0.11.2 | BSD-3-Clause. The licences of the libraries it contains (brotli, giflib, highway, libjpeg-turbo, libpng, libwebp, sjpeg, skcms, zlib) are installed beside the programs as `LICENSE.*` | https://github.com/libjxl/libjxl/releases/tag/v0.11.2 |
| libavif (`avifenc`, `avifdec`) | 1.4.2 | BSD-2-Clause (installed as `LICENSE.libavif`, from the v1.4.2 source tag). The builds embed dav1d 1.5.3 (BSD-2-Clause) and libaom 3.14.1 (BSD-2-Clause with patent grant) | https://github.com/AOMediaCodec/libavif/releases/tag/v1.4.2 |
| libwebp (`cwebp`, `webpmux`) | 1.6.0 | BSD-3-Clause with patent grant (installed as `LICENSE.libwebp` and `PATENTS.libwebp`, from the v1.6.0 source tag) | https://developers.google.com/speed/webp/download |
| pngquant (`pngquant`), optional | 2.17.0 | GPL-3.0-or-later (`COPYRIGHT`, shipped in the archive) | https://pngquant.org/ |

## pngquant is GPL-3.0-or-later

pngquant is free software under the GNU General Public License, version 3 or
later. smart-compress only runs it as a separate program and does not link or
include it. It is optional: without it PNG output still works and only the
palette PNG candidates are lost. The fetch script treats a failed download of it
as a warning, because the only official download address is not versioned.

If you redistribute the pngquant binary you must also offer its source. The
source for the version used here is the 2.17.0 tag:
https://github.com/kornelski/pngquant/tree/2.17.0

## Programs used if present, never installed by the script

- `oxipng` (MIT, https://github.com/oxipng/oxipng) re-packs PNG candidates when
  it is found on `PATH`.

## Rust crates

The program is built from Rust crates that are listed with their exact versions
in `Cargo.lock`. Each carries its own licence; most are MIT or Apache-2.0.
`fast-ssim2`, which computes the in-process SSIMULACRA2 score, is BSD-2-Clause,
so a redistributed binary must carry its copyright notice.

## Sample photographs

The photographs in `samples/photos/` are published on Unsplash under the
[Unsplash License](https://unsplash.com/license). Their sources are listed in
[samples/photos/SOURCES.md](samples/photos/SOURCES.md).
