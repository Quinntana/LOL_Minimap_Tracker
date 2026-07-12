$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VirtualEnvironment = Join-Path $ProjectRoot ".venv-build"
$Python = Join-Path $VirtualEnvironment "Scripts\python.exe"
$Ruff = Join-Path $VirtualEnvironment "Scripts\ruff.exe"
$Mypy = Join-Path $VirtualEnvironment "Scripts\mypy.exe"
$Pytest = Join-Path $VirtualEnvironment "Scripts\pytest.exe"
$PyInstaller = Join-Path $VirtualEnvironment "Scripts\pyinstaller.exe"
$ArchiveViewer = Join-Path $VirtualEnvironment "Scripts\pyi-archive_viewer.exe"
$Executable = Join-Path $ProjectRoot "dist\LoLMinimapTracker.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    python -m venv $VirtualEnvironment
    if ($LASTEXITCODE -ne 0) { throw "Could not create the build environment." }
}

& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }

& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.lock")
if ($LASTEXITCODE -ne 0) { throw "Could not install locked dependencies." }

& $Python -m pip install --no-deps -e $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Could not install the local package." }

& $Ruff format --check src tests tools main.py
if ($LASTEXITCODE -ne 0) { throw "Ruff formatting check failed." }

& $Ruff check src tests tools main.py
if ($LASTEXITCODE -ne 0) { throw "Ruff lint check failed." }

& $Mypy src
if ($LASTEXITCODE -ne 0) { throw "Mypy failed." }

& $Pytest
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }

& $PyInstaller --clean --noconfirm (Join-Path $ProjectRoot "LoLMinimapTracker.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

Copy-Item -LiteralPath (Join-Path $ProjectRoot "config.json") -Destination (Join-Path $ProjectRoot "dist\config.json") -Force

$Archive = & $ArchiveViewer -l -r $Executable
$ArchiveText = $Archive -join [Environment]::NewLine
foreach ($Required in @("lol_minimap_tracker.app", "lol_minimap_tracker.ui.calibration", "QtSvg.pyd", "cv2.pyd", "mss.windows", "skimage.metrics._structural_similarity", "windows_capture\windows_capture.pyd", "position-top.svg")) {
    if ($ArchiveText -notmatch [regex]::Escape($Required)) {
        throw "Packaged archive is missing $Required"
    }
}

$Hash = Get-FileHash -Algorithm SHA256 -LiteralPath $Executable
$ChecksumPath = "$Executable.sha256"
Set-Content -LiteralPath $ChecksumPath -Value "$($Hash.Hash) *LoLMinimapTracker.exe" -Encoding ascii
Write-Output "Built $Executable"
Write-Output "SHA256 $($Hash.Hash)"
Write-Output "Checksum $ChecksumPath"
