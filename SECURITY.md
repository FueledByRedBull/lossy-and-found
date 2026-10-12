# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately, through GitHub's advisory
feature: open the repository's **Security** tab, choose **Report a
vulnerability**, and describe the problem. Do not open a public issue for it.
Include the version, the operating system, what you did, and an input file or
command that reproduces the problem if you have one. This project is maintained
by one person; reports are handled on a best-effort basis.

## Supported versions

Only the latest commit on the default branch is supported. The project is at
version 0.1.0 and has no stable releases yet.

## What is in scope

- **Untrusted input files.** The program decodes, probes and re-encodes images it
  is given. Crashes, hangs, unbounded memory or disk use, or writes outside the
  chosen output folder caused by a crafted image are in scope. Size limits are
  checked before pixels are allocated, and they are configurable (see
  [docs/usage.md](docs/usage.md#options)).
- **The local control panel.** It listens on `127.0.0.1` only, requires a
  per-session token and a matching `Host` header on every request, serves at most
  16 connections, and reads only image files for previews. Anything that lets
  another local page or user drive it, read other files, or reach it from the
  network is in scope.
- **Running external programs.** Encoders, decoders and metric tools are found on
  absolute `PATH` entries or in `tools` folders next to the executable or the
  workspace, never in the working directory, and are started with argument lists,
  not shell strings. A way to make the program run a program the user did not
  intend is in scope.
- **The score cache.** Cached files are verified by hash; see the
  [roadmap](ROADMAP.md) for the known limit on stored scores.

## Out of scope

- Vulnerabilities in the third-party programs that `tools/fetch-tools.ps1`
  downloads (libjxl, libavif, libwebp, pngquant) or in `oxipng` themselves; report
  those upstream. Do tell us if a version we pin is affected.
- Denial of service by giving the program a very large image with the limits
  raised on purpose, or by pointing the output or cache folder at a shared
  writable location. The documentation says to keep those folders on trusted
  locations.

Reports the program writes include the paths of the inputs and outputs as you
gave them; check one before you share it.
