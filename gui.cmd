@echo off
rem Opens the smart-compress control panel in its own window.
rem Delete target\release\smart-compress-app.exe to force a rebuild after editing the source.
cd /d "%~dp0"
if not exist "target\release\smart-compress-app.exe" (
  cargo build --release --bins || exit /b 1
)
start "" "target\release\smart-compress-app.exe" %*
