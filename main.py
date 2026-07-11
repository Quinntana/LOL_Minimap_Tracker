"""PyInstaller-compatible application entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from lol_minimap_tracker.app import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run())
