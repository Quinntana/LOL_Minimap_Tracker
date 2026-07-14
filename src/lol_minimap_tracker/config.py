"""Typed configuration loading and validation."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .domain.models import ArrowDisplayMode, LastSeenMarkerStyle

# Keep untrusted JSON values comfortably inside the coordinate and allocation
# ranges used by Qt, MSS, OpenCV, and Windows Graphics Capture.  A League
# minimap is normally only a few hundred pixels wide; the larger limits retain
# room for high-DPI and unusual multi-monitor setups without permitting a
# malformed configuration to request a multi-gigabyte frame.
MAX_SCREEN_COORDINATE = 1_000_000
MAX_CAPTURE_DIMENSION = 4_096
MAX_CAPTURE_PIXELS = 4_194_304
MAX_TIMER_INTERVAL_MS = 60_000
MAX_DURATION_SECONDS = 3_600.0
MAX_PIXEL_DISTANCE = 16_384.0
MAX_POSITION_SPEED = 65_536.0
MAX_RETRY_COUNT = 10_000
MAX_API_TIMEOUT_SECONDS = 60.0
CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_SCHEMA_VERSION = 1_000_000


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
    schema_version: int = CONFIG_SCHEMA_VERSION
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
    arrow_display_mode: ArrowDisplayMode = ArrowDisplayMode.NEARBY
    arrow_nearby_range_ratio: float = 0.35
    show_last_seen: bool = True
    last_seen_marker_style: LastSeenMarkerStyle = LastSeenMarkerStyle.ROLE
    show_notifications: bool = True
    cooldown_tracker_enabled: bool = False
    cooldown_panel_locked: bool = False
    cooldown_panel_left: int | None = None
    cooldown_panel_top: int | None = None

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
    if valid:
        try:
            valid = math.isfinite(float(value))
        except (OverflowError, TypeError, ValueError):
            valid = False
    if valid and minimum is not None:
        valid = value >= minimum
    if valid and maximum is not None:
        valid = value <= maximum
    if not valid:
        logger.warning("Invalid %s=%r; using %r", key, value, default)
        return default
    return type(default)(value)


def _optional_integer(
    values: dict[str, Any],
    key: str,
    default: int | None,
    logger: logging.Logger,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    value = values.get(key, default)
    if value is None:
        return None
    valid = isinstance(value, int) and not isinstance(value, bool)
    if valid and minimum is not None:
        valid = value >= minimum
    if valid and maximum is not None:
        valid = value <= maximum
    if not valid:
        logger.warning("Invalid %s=%r; using %r", key, value, default)
        return default
    return int(value)


def config_from_mapping(values: dict[str, Any], logger: logging.Logger) -> TrackerConfig:
    schema_version = int(
        _number(
            values,
            "schema_version",
            CONFIG_SCHEMA_VERSION,
            logger,
            1,
            MAX_CONFIG_SCHEMA_VERSION,
        )
    )
    if schema_version > CONFIG_SCHEMA_VERSION:
        logger.warning(
            "Configuration schema %s is newer than supported schema %s; "
            "settings will be read but not overwritten",
            schema_version,
            CONFIG_SCHEMA_VERSION,
        )
    default_capture = CaptureRegion()
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

    capture_top = int(
        _number(
            flat_capture,
            "top",
            default_capture.top,
            logger,
            -MAX_SCREEN_COORDINATE,
            MAX_SCREEN_COORDINATE,
        )
    )
    capture_left = int(
        _number(
            flat_capture,
            "left",
            default_capture.left,
            logger,
            -MAX_SCREEN_COORDINATE,
            MAX_SCREEN_COORDINATE,
        )
    )
    capture_width = int(
        _number(
            flat_capture,
            "width",
            default_capture.width,
            logger,
            1,
            MAX_CAPTURE_DIMENSION,
        )
    )
    capture_height = int(
        _number(
            flat_capture,
            "height",
            default_capture.height,
            logger,
            1,
            MAX_CAPTURE_DIMENSION,
        )
    )
    if capture_width * capture_height > MAX_CAPTURE_PIXELS:
        logger.warning(
            "Capture area %sx%s exceeds the safe %s-pixel limit; using default size %sx%s",
            capture_width,
            capture_height,
            MAX_CAPTURE_PIXELS,
            default_capture.width,
            default_capture.height,
        )
        capture_width = default_capture.width
        capture_height = default_capture.height
    capture = CaptureRegion(
        top=capture_top,
        left=capture_left,
        width=capture_width,
        height=capture_height,
    )
    radius_min = int(_number(values, "circle_radius_min", 12, logger, 1, MAX_CAPTURE_DIMENSION))
    radius_max = int(_number(values, "circle_radius_max", 40, logger, 1, MAX_CAPTURE_DIMENSION))
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
    if capture_region_space == "client" and (capture.left < 0 or capture.top < 0):
        logger.warning(
            "Client-space capture coordinates cannot be negative; using the default screen region"
        )
        capture = default_capture
        capture_region_space = "screen"
    league_process_name = values.get("league_process_name", "League of Legends.exe")
    if not isinstance(league_process_name, str) or not league_process_name.strip():
        logger.warning("Invalid league_process_name=%r; using default", league_process_name)
        league_process_name = "League of Legends.exe"

    raw_arrow_mode = values.get("arrow_display_mode", ArrowDisplayMode.NEARBY.value)
    try:
        if not isinstance(raw_arrow_mode, str):
            raise ValueError
        arrow_display_mode = ArrowDisplayMode(raw_arrow_mode)
    except ValueError:
        logger.warning("Invalid arrow_display_mode=%r; using nearby", raw_arrow_mode)
        arrow_display_mode = ArrowDisplayMode.NEARBY

    raw_marker_style = values.get("last_seen_marker_style", LastSeenMarkerStyle.ROLE.value)
    if raw_marker_style == "ring":
        logger.warning("Legacy last_seen_marker_style='ring'; using role")
        marker_style = LastSeenMarkerStyle.ROLE
    else:
        try:
            if not isinstance(raw_marker_style, str):
                raise ValueError
            marker_style = LastSeenMarkerStyle(raw_marker_style)
        except ValueError:
            logger.warning("Invalid last_seen_marker_style=%r; using role", raw_marker_style)
            marker_style = LastSeenMarkerStyle.ROLE

    return TrackerConfig(
        schema_version=schema_version,
        capture=capture,
        hotkeys=hotkeys,
        ssim_threshold=float(_number(values, "ssim_threshold", 0.3, logger, 0.0, 1.0)),
        ssim_margin=float(_number(values, "ssim_margin", 0.04, logger, 0.0, 1.0)),
        circle_radius_min=radius_min,
        circle_radius_max=radius_max,
        update_interval_ms=int(
            _number(values, "update_interval_ms", 100, logger, 16, MAX_TIMER_INTERVAL_MS)
        ),
        detection_timeout_seconds=float(
            _number(
                values,
                "detection_timeout_seconds",
                4.0,
                logger,
                0.0,
                MAX_DURATION_SECONDS,
            )
        ),
        confirmation_frames=int(_number(values, "confirmation_frames", 2, logger, 1, 10)),
        confirmation_position_tolerance_pixels=float(
            _number(
                values,
                "confirmation_position_tolerance_pixels",
                10.0,
                logger,
                0.0,
                MAX_PIXEL_DISTANCE,
            )
        ),
        jump_confirmation_frames=int(_number(values, "jump_confirmation_frames", 3, logger, 1, 10)),
        max_position_jump_pixels=float(
            _number(
                values,
                "max_position_jump_pixels",
                18.0,
                logger,
                0.0,
                MAX_PIXEL_DISTANCE,
            )
        ),
        max_position_speed_pixels_per_second=float(
            _number(
                values,
                "max_position_speed_pixels_per_second",
                140.0,
                logger,
                0.0,
                MAX_POSITION_SPEED,
            )
        ),
        health_stale_after_seconds=float(
            _number(
                values,
                "health_stale_after_seconds",
                1.0,
                logger,
                0.1,
                MAX_DURATION_SECONDS,
            )
        ),
        capture_recovery_failure_count=int(
            _number(values, "capture_recovery_failure_count", 3, logger, 1, MAX_RETRY_COUNT)
        ),
        capture_recovery_backoff_seconds=float(
            _number(
                values,
                "capture_recovery_backoff_seconds",
                1.0,
                logger,
                0.0,
                MAX_DURATION_SECONDS,
            )
        ),
        roster_refresh_interval_seconds=float(
            _number(
                values,
                "roster_refresh_interval_seconds",
                5.0,
                logger,
                1.0,
                MAX_DURATION_SECONDS,
            )
        ),
        roster_missing_grace_polls=int(
            _number(values, "roster_missing_grace_polls", 3, logger, 1, MAX_RETRY_COUNT)
        ),
        local_api_timeout_seconds=float(
            _number(
                values,
                "local_api_timeout_seconds",
                1.0,
                logger,
                0.1,
                MAX_API_TIMEOUT_SECONDS,
            )
        ),
        capture_backend=capture_backend,
        capture_region_space=capture_region_space,
        league_process_name=league_process_name.strip(),
        window_capture_timeout_seconds=float(
            _number(
                values,
                "window_capture_timeout_seconds",
                1.0,
                logger,
                0.05,
                MAX_API_TIMEOUT_SECONDS,
            )
        ),
        log_level=log_level,
        exclude_overlay_from_capture=_boolean(values, "exclude_overlay_from_capture", True, logger),
        enable_global_hotkeys=_boolean(values, "enable_global_hotkeys", True, logger),
        show_arrows=_boolean(values, "show_arrows", True, logger),
        arrow_display_mode=arrow_display_mode,
        arrow_nearby_range_ratio=float(
            _number(values, "arrow_nearby_range_ratio", 0.35, logger, 0.05, 1.0)
        ),
        show_last_seen=_boolean(values, "show_last_seen", True, logger),
        last_seen_marker_style=marker_style,
        show_notifications=_boolean(values, "show_notifications", True, logger),
        cooldown_tracker_enabled=_boolean(values, "cooldown_tracker_enabled", False, logger),
        cooldown_panel_locked=_boolean(values, "cooldown_panel_locked", False, logger),
        cooldown_panel_left=_optional_integer(
            values,
            "cooldown_panel_left",
            None,
            logger,
            -MAX_SCREEN_COORDINATE,
            MAX_SCREEN_COORDINATE,
        ),
        cooldown_panel_top=_optional_integer(
            values,
            "cooldown_panel_top",
            None,
            logger,
            -MAX_SCREEN_COORDINATE,
            MAX_SCREEN_COORDINATE,
        ),
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
    if config.schema_version > CONFIG_SCHEMA_VERSION:
        logger.error(
            "Refusing to overwrite configuration schema %s with older schema %s",
            config.schema_version,
            CONFIG_SCHEMA_VERSION,
        )
        return False
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
