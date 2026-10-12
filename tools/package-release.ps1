param(
    [string]$Output = "target/release-package"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stage = Join-Path $root $Output
New-Item -ItemType Directory -Force -Path $stage | Out-Null

Push-Location $root
try {
    cargo build --release --bins
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
    Copy-Item "target/release/smart-compress.exe" $stage -Force
    Copy-Item "target/release/smart-compress-benchmark.exe" $stage -Force
    Copy-Item "target/release/smart-compress-app.exe" $stage -Force
    Copy-Item "README.md", "ROADMAP.md", "THIRD_PARTY_NOTICES.md", "LICENSE-MIT", "LICENSE-APACHE" $stage -Force
    Copy-Item "docs" (Join-Path $stage "docs") -Recurse -Force

    # Only what tools/fetch-tools.ps1 installs; nothing else that sits in tools/.
    New-Item -ItemType Directory -Force -Path (Join-Path $stage "tools") | Out-Null
    foreach ($name in "libjxl", "libavif", "libwebp", "pngquant") {
        $dir = Join-Path $root "tools/$name"
        if (Test-Path $dir) {
            Copy-Item $dir (Join-Path $stage "tools/$name") -Recurse -Force
        }
    }
    Write-Output "Release staging directory: $stage"
}
finally {
    Pop-Location
}
