"""Download and validate the role SVGs used by the overlay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree

ROLES = ("top", "jungle", "middle", "bottom", "utility")
BASE_URL = (
    "https://raw.communitydragon.org/latest/plugins/"
    "rcp-fe-lol-champ-select/global/default/svg/position-{role}.svg"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "src" / "lol_minimap_tracker" / "assets" / "roles"


def fetch_role_icons() -> dict[str, dict[str, str]]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}
    for role in ROLES:
        url = BASE_URL.format(role=role)
        request = Request(url, headers={"User-Agent": "LoLMinimapTracker-AssetFetcher/1.0"})
        with urlopen(request, timeout=20) as response:
            content_type = response.headers.get_content_type()
            content = response.read()
        if content_type != "image/svg+xml":
            raise ValueError(f"Unexpected content type for {role}: {content_type}")
        root = ElementTree.fromstring(content)
        if root.tag.rsplit("}", 1)[-1].lower() != "svg":
            raise ValueError(f"Downloaded {role} asset is not SVG")
        filename = f"position-{role}.svg"
        (ASSET_DIR / filename).write_bytes(content)
        manifest[role] = {
            "file": filename,
            "sha256": hashlib.sha256(content).hexdigest(),
            "source": url,
        }
    (ASSET_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    result = fetch_role_icons()
    print(f"Fetched {len(result)} validated role icons into {ASSET_DIR}")
