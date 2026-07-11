from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import lol_minimap_tracker.config as config_module
from lol_minimap_tracker.config import (
    DEFAULT_CONFIG,
    CaptureRegion,
    config_from_mapping,
    ensure_config,
    load_config,
    save_config,
)


def test_config_loads_flat_capture_and_nested_hotkeys(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"top": 700, "width": 300, "hotkeys": {"quit": "alt+q"}}),
        encoding="utf-8",
    )
    config = load_config(path, logging.getLogger("test"))
    assert config.capture.top == 700
    assert config.capture.width == 300
    assert config.hotkeys.quit == "alt+q"
    assert config.hotkeys.toggle_arrows == "ctrl+a"


def test_config_rejects_invalid_values(caplog: object) -> None:
    config = config_from_mapping(
        {
            "width": 0,
            "ssim_threshold": 2,
            "ssim_margin": -1,
            "confirmation_frames": 0,
            "confirmation_position_tolerance_pixels": -1,
            "jump_confirmation_frames": 20,
            "max_position_jump_pixels": -1,
            "max_position_speed_pixels_per_second": -1,
            "health_stale_after_seconds": 0,
            "capture_recovery_failure_count": 0,
            "capture_recovery_backoff_seconds": -1,
            "circle_radius_min": 50,
            "circle_radius_max": 20,
            "log_level": "LOUD",
            "exclude_overlay_from_capture": "false",
            "show_arrows": 1,
            "show_last_seen": "yes",
            "show_notifications": None,
        },
        logging.getLogger("test"),
    )
    assert config.capture.width == DEFAULT_CONFIG.capture.width
    assert config.ssim_threshold == DEFAULT_CONFIG.ssim_threshold
    assert config.ssim_margin == DEFAULT_CONFIG.ssim_margin
    assert config.confirmation_frames == DEFAULT_CONFIG.confirmation_frames
    assert (
        config.confirmation_position_tolerance_pixels
        == DEFAULT_CONFIG.confirmation_position_tolerance_pixels
    )
    assert config.jump_confirmation_frames == DEFAULT_CONFIG.jump_confirmation_frames
    assert config.max_position_jump_pixels == DEFAULT_CONFIG.max_position_jump_pixels
    assert (
        config.max_position_speed_pixels_per_second
        == DEFAULT_CONFIG.max_position_speed_pixels_per_second
    )
    assert config.health_stale_after_seconds == DEFAULT_CONFIG.health_stale_after_seconds
    assert config.capture_recovery_failure_count == DEFAULT_CONFIG.capture_recovery_failure_count
    assert (
        config.capture_recovery_backoff_seconds == DEFAULT_CONFIG.capture_recovery_backoff_seconds
    )
    assert config.circle_radius_min == DEFAULT_CONFIG.circle_radius_min
    assert config.circle_radius_max == DEFAULT_CONFIG.circle_radius_max
    assert config.log_level == "INFO"
    assert config.exclude_overlay_from_capture is True
    assert config.show_arrows is True
    assert config.show_last_seen is True
    assert config.show_notifications is True


def test_legacy_json_is_read_without_rewrite(tmp_path: Path, caplog: object) -> None:
    path = tmp_path / "config.txt"
    original = '{"height": 321}'
    path.write_text(original, encoding="utf-8")
    config = load_config(path, logging.getLogger("test"))
    assert config.capture.height == 321
    assert path.read_text(encoding="utf-8") == original


def test_missing_or_malformed_config_uses_defaults(tmp_path: Path) -> None:
    logger = logging.getLogger("test")
    assert load_config(tmp_path / "missing.json", logger) == DEFAULT_CONFIG
    malformed = tmp_path / "bad.json"
    malformed.write_text("[]", encoding="utf-8")
    assert load_config(malformed, logger) == DEFAULT_CONFIG


def test_nested_capture_and_invalid_hotkeys_use_safe_defaults() -> None:
    config = config_from_mapping(
        {
            "capture": {"top": -200, "left": -1920, "width": 400, "height": 300},
            "hotkeys": {"quit": "", "pause": 123},
            "enable_global_hotkeys": False,
        },
        logging.getLogger("test"),
    )
    assert config.capture.top == -200
    assert config.capture.left == -1920
    assert config.capture.width == 400
    assert config.hotkeys.quit == DEFAULT_CONFIG.hotkeys.quit
    assert config.hotkeys.pause == DEFAULT_CONFIG.hotkeys.pause
    assert config.enable_global_hotkeys is False


def test_malformed_json_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{broken", encoding="utf-8")
    assert load_config(path, logging.getLogger("test")) == DEFAULT_CONFIG


def test_default_config_bootstrap_and_preference_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings" / "config.json"
    logger = logging.getLogger("test")
    assert ensure_config(path, logger)
    assert not ensure_config(path, logger)
    assert load_config(path, logger) == DEFAULT_CONFIG

    updated = replace(
        DEFAULT_CONFIG,
        capture=CaptureRegion(top=-100, left=-1800, width=320, height=280),
        show_arrows=False,
        show_last_seen=False,
        show_notifications=False,
    )
    assert save_config(path, updated, logger)
    assert load_config(path, logger) == updated


def test_failed_atomic_save_preserves_existing_config(tmp_path: Path, monkeypatch: Any) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"width": 321}', encoding="utf-8")
    original = path.read_text(encoding="utf-8")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("read only")

    monkeypatch.setattr(config_module.os, "replace", fail_replace)
    assert not save_config(path, DEFAULT_CONFIG, logging.getLogger("test"))
    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.tmp"))
