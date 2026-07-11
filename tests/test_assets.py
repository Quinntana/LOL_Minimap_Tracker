from __future__ import annotations

import hashlib
import json
from xml.etree import ElementTree

from lol_minimap_tracker.paths import AppPaths


def test_role_assets_match_manifest() -> None:
    asset_dir = AppPaths.discover().role_asset_dir
    manifest = json.loads((asset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"top", "jungle", "middle", "bottom", "utility"}
    for entry in manifest.values():
        path = asset_dir / entry["file"]
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        assert ElementTree.fromstring(content).tag.rsplit("}", 1)[-1] == "svg"
        assert entry["source"].startswith("https://raw.communitydragon.org/")
    assert (asset_dir / "position-unknown.svg").exists()
