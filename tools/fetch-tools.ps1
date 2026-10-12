<#
.SYNOPSIS
  Downloads the third-party encoders smart-compress runs (official releases) into the tools folder.
.DESCRIPTION
  Every download is pinned by SHA-256 and refused on mismatch. Nothing downloaded is executed.
  Only the executables the program uses and each package's licence files are kept.
  libjxl, libavif and libwebp are required: a failure for any of them stops the script with exit code 1.
  pngquant is optional (PNG output works without it, only palette candidates are lost) and its only official URL
  is not versioned, so a download or hash failure for it prints a warning and the script goes on.
.PARAMETER Destination  Folder to install into (default: the folder this script is in).
.PARAMETER Force        Reinstall packages that are already present.
.PARAMETER CacheDir     Keep verified downloads here (for CI caching) instead of a temporary folder.
#>
param(
    [string]$Destination = $PSScriptRoot,
    [switch]$Force,
    [string]$CacheDir
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.IO.Compression.FileSystem

$gh = 'https://github.com'
$webp = 'libwebp-1.6.0-windows-x64'
# Files = entries kept from the archive; Extra = licence files the archive lacks, fetched from the tagged source.
# Optional = a failure is a warning, not an error.
# Everything lands at <Destination>\<Dest>\<path>, which is where the program looks.
$packages = @(
    @{ Name = 'libjxl 0.11.2'; Dest = 'libjxl'
       Url = "$gh/libjxl/libjxl/releases/download/v0.11.2/jxl-x64-windows-static.zip"
       Sha256 = '97dc815bdd99ba243d8502050357342cf649251a5df069f8c3daee6828cbe0ce'
       Files = @('cjxl', 'djxl', 'cjpegli', 'ssimulacra2', 'butteraugli_main', 'jxlinfo' | ForEach-Object { "bin/$_.exe" }) +
               @('brotli', 'giflib', 'highway', 'libjpeg-turbo', 'libjxl', 'libpng', 'libwebp', 'sjpeg', 'skcms', 'zlib' | ForEach-Object { "bin/LICENSE.$_" })
       Extra = @() },
    @{ Name = 'libavif 1.4.2'; Dest = 'libavif'
       Url = "$gh/AOMediaCodec/libavif/releases/download/v1.4.2/windows-artifacts.zip"
       Sha256 = 'cb2d9fea43dcbab1d0707e3b37eb7b08070ad2fb60a2c188c39ec12382c0484a'
       Files = @('avifenc.exe', 'avifdec.exe')
       Extra = @(@{ Path = 'LICENSE.libavif'; Sha256 = '165abf92cc04b39e80d29cadea7a6a7e8fddf59407d4ad2616507a7ebe8216f9'
                    Url = 'https://raw.githubusercontent.com/AOMediaCodec/libavif/v1.4.2/LICENSE' }) },
    @{ Name = 'libwebp 1.6.0'; Dest = 'libwebp'
       Url = 'https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0-windows-x64.zip'
       Sha256 = '48886f506b21f62e4661f0f4cbfca19800897c385128e8902542d29a950c93f1'
       Files = @("$webp/bin/cwebp.exe", "$webp/bin/webpmux.exe")
       Extra = @(@{ Path = "$webp/bin/LICENSE.libwebp"; Sha256 = '5aec868f669e384a22372a4e8a1a6cd7d44c64cd451f960ca69cc170d1e13acf'
                    Url = 'https://raw.githubusercontent.com/webmproject/libwebp/v1.6.0/COPYING' },
                  @{ Path = "$webp/bin/PATENTS.libwebp"; Sha256 = 'cc3273e0694ea5896145e0677699b53471b03ea43021ddc50e7923fbb9f5023c'
                    Url = 'https://raw.githubusercontent.com/webmproject/libwebp/v1.6.0/PATENTS' }) },
    @{ Name = 'pngquant 2.17.0'; Dest = 'pngquant'
       Optional = $true
       Url = 'https://pngquant.org/pngquant-windows.zip'
       Sha256 = 'bd0257aeeccfe446a4cd764927e26f8af6051796f28abed104307284107b120d'
       Files = @('pngquant/pngquant.exe', 'pngquant/COPYRIGHT')
       Extra = @() }
)

function Get-Verified([string]$Url, [string]$Sha256, [string]$Dir) {
    $file = Join-Path $Dir "$Sha256.download"
    if (-not (Test-Path -LiteralPath $file)) {
        Write-Host "  downloading $Url"
        Invoke-WebRequest -Uri $Url -OutFile $file -UseBasicParsing
    }
    $actual = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Sha256) {
        Remove-Item -LiteralPath $file -Force
        throw "SHA-256 mismatch for $Url`n  expected $Sha256`n  got      $actual"
    }
    $file
}

try {
    $root = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Destination)
    $temp = $null
    if ($CacheDir) {
        $cache = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CacheDir)
    } else {
        $cache = $temp = Join-Path ([IO.Path]::GetTempPath()) ("smart-compress-tools-" + [guid]::NewGuid().ToString('N'))
    }
    New-Item -ItemType Directory -Force -Path $cache | Out-Null

    $summary = @()
    foreach ($p in $packages) {
        $base = Join-Path $root $p.Dest
        $wanted = @($p.Files) + @($p.Extra | ForEach-Object { $_.Path })
        $present = @($wanted | Where-Object { Test-Path -LiteralPath (Join-Path $base $_) }).Count -eq $wanted.Count
        if ($present -and -not $Force) {
            $summary += "skipped    $($p.Name) (already in $base)"
            continue
        }
        try {
            Write-Host "$($p.Name)"
            $archive = Get-Verified $p.Url $p.Sha256 $cache
            $extras = @($p.Extra | ForEach-Object { @{ Path = $_.Path; From = (Get-Verified $_.Url $_.Sha256 $cache) } })

            $zip = [IO.Compression.ZipFile]::OpenRead($archive)
            try {
                foreach ($name in $p.Files) {
                    $entry = $zip.GetEntry($name)
                    if (-not $entry) { throw "$($p.Name): '$name' not found in the archive" }
                    $target = Join-Path $base $name
                    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
                    [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
                }
            } finally { $zip.Dispose() }
            foreach ($e in $extras) {
                $target = Join-Path $base $e.Path
                New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
                Copy-Item -LiteralPath $e.From -Destination $target -Force
            }
            $summary += "installed  $($p.Name): $($wanted.Count) files in $base"
        } catch {
            if (-not $p.Optional) { throw }
            Write-Warning "$($p.Name) was not installed: $($_.Exception.Message)"
            Write-Warning "$($p.Name) is optional. PNG output still works; palette PNG candidates are not tried without it."
            $summary += "WARNING    $($p.Name) not installed (optional)"
        }
    }

    Write-Host ''
    $summary | ForEach-Object { Write-Host $_ }
    Write-Host 'Not installed (optional): oxipng (better PNG recompression) - winget install oxipng or https://github.com/oxipng/oxipng/releases;'
    Write-Host '                          cjpeg (JPEG fallback) - a mozjpeg or libjpeg-turbo build on PATH.'
} catch {
    [Console]::Error.WriteLine("fetch-tools: $($_.Exception.Message)")
    exit 1
} finally {
    if ($temp -and (Test-Path -LiteralPath $temp)) { Remove-Item -LiteralPath $temp -Recurse -Force }
}
