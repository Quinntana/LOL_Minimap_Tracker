# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


project_root = Path(SPECPATH)
datas = collect_data_files("lol_minimap_tracker")
datas.append((str(project_root / "config.json"), "."))

analysis = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "keyboard",
        "mss",
        "PyQt5.QtSvg",
        "windows_capture",
        "windows_capture.windows_capture",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "coverage",
        "imageio",
        "lazy_loader",
        "matplotlib",
        "mypy",
        "networkx",
        "pandas",
        "PIL",
        "pytest",
        "ruff",
        "scipy",
        "seaborn",
        "skimage",
        "tifffile",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="LoLMinimapTracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    manifest=str(project_root / "LoLMinimapTracker.manifest"),
)
