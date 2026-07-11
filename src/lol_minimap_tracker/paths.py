"""Portable application path discovery."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    app_dir: Path
    user_data_dir: Path
    config_path: Path
    role_asset_dir: Path

    @classmethod
    def discover(cls) -> AppPaths:
        if getattr(sys, "frozen", False):
            app_dir = Path(sys.executable).resolve().parent
        else:
            app_dir = Path(__file__).resolve().parents[2]
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        user_data_dir = local_app_data / "LoLMinimapTracker"
        canonical = app_dir / "config.json"
        legacy = app_dir / "config.txt"
        if canonical.exists():
            config_path = canonical
        elif legacy.exists():
            config_path = legacy
        else:
            config_path = user_data_dir / "config.json"
        role_asset_dir = Path(str(files("lol_minimap_tracker").joinpath("assets", "roles")))
        return cls(app_dir, user_data_dir, config_path, role_asset_dir)

    @property
    def log_dir(self) -> Path:
        return self.user_data_dir / "logs"

    @property
    def config_write_path(self) -> Path:
        if self.config_path.name.casefold() == "config.txt":
            return self.config_path.with_name("config.json")
        return self.config_path

    @property
    def cache_dir(self) -> Path:
        return self.user_data_dir / "cache"

    @property
    def timeline_path(self) -> Path:
        return self.user_data_dir / "timeline.csv"

    @property
    def lock_path(self) -> Path:
        return self.user_data_dir / "tracker.lock"

    def ensure_runtime_directories(self) -> None:
        for path in (self.user_data_dir, self.log_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)
