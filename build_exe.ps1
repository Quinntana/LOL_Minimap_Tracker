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
    $EnvironmentCreated = $false
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        & $PyLauncher.Source -3.13 -m venv $VirtualEnvironment
        $EnvironmentCreated = $LASTEXITCODE -eq 0
    }
    if (-not $EnvironmentCreated) {
        python -m venv $VirtualEnvironment
        $EnvironmentCreated = $LASTEXITCODE -eq 0
    }
    if (-not $EnvironmentCreated) { throw "Could not create the build environment." }
}

$BuildRuntime = (& $Python -c "import struct, sys; print('{0}.{1}|{2}'.format(sys.version_info.major, sys.version_info.minor, struct.calcsize('P') * 8))").Trim()
if ($BuildRuntime -ne "3.13|64") {
    throw "The locked Windows executable build requires 64-bit CPython 3.13 (found $BuildRuntime)."
}

& $Python -m pip install --upgrade "pip==26.1.2"
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }

& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.lock")
if ($LASTEXITCODE -ne 0) { throw "Could not install locked dependencies." }

& $Python -m pip install --no-deps --no-build-isolation -e $ProjectRoot
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

$Archive = & $ArchiveViewer -l -r $Executable
$ArchiveText = $Archive -join [Environment]::NewLine
foreach ($Required in @("lol_minimap_tracker.app", "lol_minimap_tracker.domain.cooldowns", "lol_minimap_tracker.integrations.capture", "lol_minimap_tracker.integrations.clickthrough", "lol_minimap_tracker.integrations.cooldown_data", "lol_minimap_tracker.integrations.cooldown_events", "lol_minimap_tracker.integrations.data_dragon", "lol_minimap_tracker.integrations.live_client", "lol_minimap_tracker.ui.calibration", "lol_minimap_tracker.ui.champion_portraits", "lol_minimap_tracker.ui.cooldown_panel", "lol_minimap_tracker.ui.overlay", "QtSvg.pyd", "cv2.pyd", "mss.windows", "windows_capture.pyd", "position-top.svg", "position-unknown.svg")) {
    if ($ArchiveText -notmatch [regex]::Escape($Required)) {
        throw "Packaged archive is missing $Required"
    }
}
foreach ($ForbiddenPrefix in @("'imageio", "'PIL", "'scipy", "'skimage", "'tifffile")) {
    if ($ArchiveText -match [regex]::Escape($ForbiddenPrefix)) {
        throw "Packaged archive unexpectedly contains $($ForbiddenPrefix.Trim("'"))"
    }
}

$Hash = Get-FileHash -Algorithm SHA256 -LiteralPath $Executable
$ChecksumPath = "$Executable.sha256"
Set-Content -LiteralPath $ChecksumPath -Value "$($Hash.Hash) *LoLMinimapTracker.exe" -Encoding ascii
Write-Output "Built $Executable"
Write-Output "SHA256 $($Hash.Hash)"
Write-Output "Checksum $ChecksumPath"
