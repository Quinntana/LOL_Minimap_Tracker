from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

import lol_minimap_tracker.integrations.capture as capture_module
from lol_minimap_tracker.config import CaptureRegion
from lol_minimap_tracker.integrations.capture import MssFrameSource
from lol_minimap_tracker.integrations.clock import SystemClock
from lol_minimap_tracker.logging_setup import configure_logging
from lol_minimap_tracker.paths import AppPaths


class Capture:
    def __init__(self) -> None:
        self.closed = False
        self.monitor: dict[str, int] | None = None

    def grab(self, monitor: dict[str, int]) -> np.ndarray[Any, Any]:
        self.monitor = monitor
        frame = np.zeros((20, 30, 4), dtype=np.uint8)
        frame[0, 0] = [7, 11, 223, 255]
        return frame

    def close(self) -> None:
        self.closed = True


def test_mss_frame_source_reuses_and_closes_capture(monkeypatch: Any) -> None:
    capture = Capture()
    monkeypatch.setattr(capture_module.mss, "mss", lambda: capture)
    source = MssFrameSource(CaptureRegion(1, 2, 30, 20))
    source.start()
    source.start()
    frame = source.capture()
    assert frame.shape == (20, 30, 3)
    assert frame[0, 0].tolist() == [7, 11, 223]
    assert capture.monitor == {"top": 1, "left": 2, "width": 30, "height": 20}
    updated = CaptureRegion(-20, -30, 40, 50)
    source.set_region(updated)
    assert source.region == updated
    source.capture()
    assert capture.monitor == {"top": -20, "left": -30, "width": 40, "height": 50}
    source.close()
    assert capture.closed


def test_capture_requires_start() -> None:
    source = MssFrameSource(CaptureRegion())
    try:
        source.capture()
    except RuntimeError as exc:
        assert "started" in str(exc)
    else:
        raise AssertionError("capture should require start")


def test_clock_logging_and_paths(tmp_path: Path, monkeypatch: Any) -> None:
    clock = SystemClock()
    assert clock.monotonic() > 0
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", clock.timestamp())

    logger = configure_logging(tmp_path / "logs", "DEBUG")
    logger.debug("written")
    assert (tmp_path / "logs" / "tracker.log").exists()
    assert logger.level == logging.DEBUG

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    paths = AppPaths.discover()
    paths.ensure_runtime_directories()
    assert paths.log_dir.exists()
    assert paths.cache_dir.exists()
    assert paths.timeline_path.parent == paths.user_data_dir
    assert paths.lock_path.parent == paths.user_data_dir
    assert paths.role_asset_dir.exists()


def test_frozen_paths_prefer_config_json_then_legacy(tmp_path: Path, monkeypatch: Any) -> None:
    executable = tmp_path / "portable" / "LoLMinimapTracker.exe"
    executable.parent.mkdir()
    legacy = executable.parent / "config.txt"
    legacy.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert AppPaths.discover().config_path == legacy
    canonical = executable.parent / "config.json"
    canonical.write_text("{}", encoding="utf-8")
    assert AppPaths.discover().config_path == canonical


def test_frozen_paths_use_local_appdata_when_portable_config_is_missing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    executable = tmp_path / "readonly" / "LoLMinimapTracker.exe"
    executable.parent.mkdir()
    local = tmp_path / "local"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    paths = AppPaths.discover()
    assert paths.config_path == local / "LoLMinimapTracker" / "config.json"
    assert paths.config_write_path == paths.config_path


def test_legacy_config_writes_to_canonical_sibling(tmp_path: Path, monkeypatch: Any) -> None:
    executable = tmp_path / "portable" / "LoLMinimapTracker.exe"
    executable.parent.mkdir()
    legacy = executable.parent / "config.txt"
    legacy.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    paths = AppPaths.discover()
    assert paths.config_path == legacy
    assert paths.config_write_path == legacy.with_name("config.json")
