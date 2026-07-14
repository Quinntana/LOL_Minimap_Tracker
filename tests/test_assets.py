from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

from lol_minimap_tracker.paths import AppPaths


def test_role_assets_match_manifest() -> None:
    asset_dir = AppPaths.discover().role_asset_dir
    manifest = json.loads((asset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"top", "jungle", "middle", "bottom", "utility", "unknown"}
    for name, entry in manifest.items():
        path = asset_dir / entry["file"]
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        assert ElementTree.fromstring(content).tag.rsplit("}", 1)[-1] == "svg"
        if name == "unknown":
            assert entry["source"] == "project-generated"
        else:
            assert entry["source"].startswith("https://raw.communitydragon.org/")


def test_build_script_checks_calibration_and_writes_checksum() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "build_exe.ps1").read_text(encoding="utf-8")
    assert "lol_minimap_tracker.ui.calibration" in script
    assert "Set-Content -LiteralPath $ChecksumPath" in script
    assert "LoLMinimapTracker.exe.sha256" not in script
