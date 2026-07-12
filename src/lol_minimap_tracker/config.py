"""Typed configuration loading and validation."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .domain.models import LastSeenMarkerStyle


@dataclass(frozen=True)
class CaptureRegion:
    top: int = 813
    left: int = 1655
    width: int = 252
    height: int = 252


@dataclass(frozen=True)
class HotkeyConfig:
    save_timeline: str = "ctrl+s"
    quit: str = "ctrl+d"
    toggle_arrows: str = "ctrl+a"
    pause: str = "ctrl+p"
    toggle_last_seen: str = "ctrl+l"
    toggle_timeline_logging: str = "ctrl+t"


@dataclass(frozen=True)
class TrackerConfig:
    capture: CaptureRegion = field(default_factory=CaptureRegion)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    ssim_threshold: float = 0.3
    ssim_margin: float = 0.04
    circle_radius_min: int = 12
    circle_radius_max: int = 40
    update_interval_ms: int = 100
    detection_timeout_seconds: float = 4.0
    confirmation_frames: int = 2
    confirmation_position_tolerance_pixels: float = 10.0
    jump_confirmation_frames: int = 3
    max_position_jump_pixels: float = 18.0
    max_position_speed_pixels_per_second: float = 140.0
    health_stale_after_seconds: float = 1.0
    capture_recovery_failure_count: int = 3
    capture_recovery_backoff_seconds: float = 1.0
    roster_refresh_interval_seconds: float = 5.0
    roster_missing_grace_polls: int = 3
    local_api_timeout_seconds: float = 1.0
    capture_backend: str = "league_window"
    capture_region_space: str = "screen"
    league_process_name: str = "League of Legends.exe"
    window_capture_timeout_seconds: float = 1.0
    log_level: str = "INFO"
    exclude_overlay_from_capture: bool = True
    enable_global_hotkeys: bool = True
    show_arrows: bool = True
    show_last_seen: bool = True
    last_seen_marker_style: LastSeenMarkerStyle = LastSeenMarkerStyle.PORTRAIT
    show_notifications: bool = True

    def to_mapping(self) -> dict[str, Any]:
        data = asdict(self)
        capture = data.pop("capture")
        data.update(capture)
        return data


DEFAULT_CONFIG = TrackerConfig()


def _boolean(values: dict[str, Any], key: str, default: bool, logger: logging.Logger) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        logger.warning("Invalid %s=%r; using %r", key, value, default)
        return default
    return value


def _number(
    values: dict[str, Any],
    key: str,
    default: int | float,
    logger: logging.Logger,
    minimum: float | None = None,
    maximum: float | None = None,
) -> int | float:
    value = values.get(key, default)
    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    if valid and minimum is not None:
        valid = value >= minimum
    if valid and maximum is not None:
        valid = value <= maximum
    if not valid:
        logger.warning("Invalid %s=%r; using %r", key, value, default)
        return default
    return type(default)(value)


def config_from_mapping(values: dict[str, Any], logger: logging.Logger) -> TrackerConfig:
    capture_values = values.get("capture", {})
    if not isinstance(capture_values, dict):
        capture_values = {}
    flat_capture = {
        **capture_values,
        **{key: values[key] for key in ("top", "left", "width", "height") if key in values},
    }

    hotkey_values = values.get("hotkeys", {})
    if not isinstance(hotkey_values, dict):
        logger.warning("hotkeys must be an object; using defaults")
        hotkey_values = {}
    default_hotkeys = asdict(HotkeyConfig())
    hotkeys = HotkeyConfig(
        **{
            name: value
            if isinstance((value := hotkey_values.get(name, default)), str) and value.strip()
            else default
            for name, default in default_hotkeys.items()
        }
    )

    capture = CaptureRegion(
        top=int(_number(flat_capture, "top", 813, logger)),
        left=int(_number(flat_capture, "left", 1655, logger)),
        width=int(_number(flat_capture, "width", 252, logger, 1)),
        height=int(_number(flat_capture, "height", 252, logger, 1)),
    )
    radius_min = int(_number(values, "circle_radius_min", 12, logger, 1))
    radius_max = int(_number(values, "circle_radius_max", 40, logger, 1))
    if radius_min > radius_max:
        logger.warning("circle_radius_min exceeds circle_radius_max; using defaults")
        radius_min, radius_max = 12, 40

    log_level = str(values.get("log_level", "INFO")).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        logger.warning("Invalid log_level=%r; using INFO", log_level)
        log_level = "INFO"

    capture_backend = values.get("capture_backend", "league_window")
    if not isinstance(capture_backend, str) or capture_backend not in {
        "league_window",
        "desktop_mss",
    }:
        logger.warning("Invalid capture_backend=%r; using league_window", capture_backend)
        capture_backend = "league_window"
    capture_region_space = values.get("capture_region_space", "screen")
    if not isinstance(capture_region_space, str) or capture_region_space not in {
        "screen",
        "client",
    }:
        logger.warning("Invalid capture_region_space=%r; using screen", capture_region_space)
        capture_region_space = "screen"
    if capture_backend == "desktop_mss" and capture_region_space == "client":
        logger.warning(
            "desktop_mss requires screen-space coordinates; keeping league_window backend"
        )
        capture_backend = "league_window"
    league_process_name = values.get("league_process_name", "League of Legends.exe")
    if not isinstance(league_process_name, str) or not league_process_name.strip():
        logger.warning("Invalid league_process_name=%r; using default", league_process_name)
        league_process_name = "League of Legends.exe"

    raw_marker_style = values.get("last_seen_marker_style", LastSeenMarkerStyle.PORTRAIT.value)
    if raw_marker_style == "ring":
        logger.warning("Legacy last_seen_marker_style='ring'; using portrait")
        marker_style = LastSeenMarkerStyle.PORTRAIT
    else:
        try:
            if not isinstance(raw_marker_style, str):
                raise ValueError
            marker_style = LastSeenMarkerStyle(raw_marker_style)
        except ValueError:
            logger.warning("Invalid last_seen_marker_style=%r; using portrait", raw_marker_style)
            marker_style = LastSeenMarkerStyle.PORTRAIT

    return TrackerConfig(
        capture=capture,
        hotkeys=hotkeys,
        ssim_threshold=float(_number(values, "ssim_threshold", 0.3, logger, 0.0, 1.0)),
        ssim_margin=float(_number(values, "ssim_margin", 0.04, logger, 0.0, 1.0)),
        circle_radius_min=radius_min,
        circle_radius_max=radius_max,
        update_interval_ms=int(_number(values, "update_interval_ms", 100, logger, 16)),
        detection_timeout_seconds=float(
            _number(values, "detection_timeout_seconds", 4.0, logger, 0.0)
        ),
        confirmation_frames=int(_number(values, "confirmation_frames", 2, logger, 1, 10)),
        confirmation_position_tolerance_pixels=float(
            _number(values, "confirmation_position_tolerance_pixels", 10.0, logger, 0.0)
        ),
        jump_confirmation_frames=int(_number(values, "jump_confirmation_frames", 3, logger, 1, 10)),
        max_position_jump_pixels=float(
            _number(values, "max_position_jump_pixels", 18.0, logger, 0.0)
        ),
        max_position_speed_pixels_per_second=float(
            _number(values, "max_position_speed_pixels_per_second", 140.0, logger, 0.0)
        ),
        health_stale_after_seconds=float(
            _number(values, "health_stale_after_seconds", 1.0, logger, 0.1)
        ),
        capture_recovery_failure_count=int(
            _number(values, "capture_recovery_failure_count", 3, logger, 1)
        ),
        capture_recovery_backoff_seconds=float(
            _number(values, "capture_recovery_backoff_seconds", 1.0, logger, 0.0)
        ),
        roster_refresh_interval_seconds=float(
            _number(values, "roster_refresh_interval_seconds", 5.0, logger, 1.0)
        ),
        roster_missing_grace_polls=int(_number(values, "roster_missing_grace_polls", 3, logger, 1)),
        local_api_timeout_seconds=float(
            _number(values, "local_api_timeout_seconds", 1.0, logger, 0.1)
        ),
        capture_backend=capture_backend,
        capture_region_space=capture_region_space,
        league_process_name=league_process_name.strip(),
        window_capture_timeout_seconds=float(
            _number(values, "window_capture_timeout_seconds", 1.0, logger, 0.05)
        ),
        log_level=log_level,
        exclude_overlay_from_capture=_boolean(values, "exclude_overlay_from_capture", True, logger),
        enable_global_hotkeys=_boolean(values, "enable_global_hotkeys", True, logger),
        show_arrows=_boolean(values, "show_arrows", True, logger),
        show_last_seen=_boolean(values, "show_last_seen", True, logger),
        last_seen_marker_style=marker_style,
        show_notifications=_boolean(values, "show_notifications", True, logger),
    )


def load_config(path: Path, logger: logging.Logger) -> TrackerConfig:
    if not path.exists():
        logger.warning("Configuration file %s was not found; using defaults", path)
        return DEFAULT_CONFIG
    try:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("configuration root must be an object")
        if path.name == "config.txt":
            logger.warning("config.txt is deprecated; rename it to config.json")
        return config_from_mapping(values, logger)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Could not load %s: %s; using defaults", path, exc)
        return DEFAULT_CONFIG


def save_config(path: Path, config: TrackerConfig, logger: logging.Logger) -> bool:
    """Atomically persist a complete canonical configuration."""
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(config.to_mapping(), indent=4) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        return True
    except OSError as exc:
        logger.error("Could not save %s: %s", path, exc)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        return False


def ensure_config(path: Path, logger: logging.Logger) -> bool:
    """Create a user-editable default configuration when none exists."""
    if path.exists():
        return False
    return save_config(path, DEFAULT_CONFIG, logger)
