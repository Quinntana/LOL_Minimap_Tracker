from __future__ import annotations

import tomllib
from pathlib import Path
from xml.etree import ElementTree

from lol_minimap_tracker import __version__


def test_release_metadata_uses_the_package_version_source() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = ElementTree.parse(root / "LoLMinimapTracker.manifest").getroot()
    identity = manifest.find("{urn:schemas-microsoft-com:asm.v1}assemblyIdentity")

    assert "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"]["attr"] == (
        "lol_minimap_tracker._version.__version__"
    )
    assert identity is not None
    assert identity.attrib["version"] == f"{__version__}.0"


def test_release_build_is_pinned_and_does_not_ship_live_settings() -> None:
    root = Path(__file__).resolve().parents[1]
    build_script = (root / "build_exe.ps1").read_text(encoding="utf-8")
    lock = (root / "requirements.lock").read_text(encoding="utf-8")
    spec = (root / "LoLMinimapTracker.spec").read_text(encoding="utf-8")

    assert '"3.13|64"' in build_script
    assert '"pip==26.1.2"' in build_script
    assert "--no-build-isolation" in build_script
    assert "pip==26.1.2" in lock
    assert "wheel==0.47.0" in lock
    assert 'datas.append((str(project_root / "config.json")' not in spec
    assert "Copy-Item" not in build_script or "config.json" not in build_script
