"""Data Dragon metadata and portrait cache."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..domain.interfaces import Image
from ..domain.models import RosterMember


class DataDragonClient:
    BASE_URL = "https://ddragon.leagueoflegends.com"

    def __init__(
        self,
        cache_dir: Path,
        logger: logging.Logger,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = cache_dir / "ddragon"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.session = session or requests.Session()
        if session is None:
            retry = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
            )
            self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.setdefault("User-Agent", "LoLMinimapTracker-PrivateResearch/1.0")
        self._version: str | None = None
        self._metadata: dict[str, str] | None = None

    @property
    def version_file(self) -> Path:
        return self.cache_dir / "version.txt"

    def _atomic_write(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

    def get_latest_version(self) -> str | None:
        if self._version:
            return self._version
        try:
            response = self.session.get(f"{self.BASE_URL}/api/versions.json", timeout=(2, 5))
            response.raise_for_status()
            versions = response.json()
            if not isinstance(versions, list) or not versions or not isinstance(versions[0], str):
                raise ValueError("versions.json had an unexpected shape")
            self._version = versions[0]
            self._atomic_write(self.version_file, self._version.encode("utf-8"))
        except (requests.RequestException, ValueError, OSError) as exc:
            self.logger.warning("Data Dragon version request failed: %s", exc)
            try:
                cached = self.version_file.read_text(encoding="utf-8").strip()
                self._version = cached or None
            except OSError:
                self._version = None
        return self._version

    def _metadata_file(self, version: str) -> Path:
        return self.cache_dir / version / "champion.json"

    def _parse_metadata(self, payload: Any) -> dict[str, str]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise ValueError("champion metadata had an unexpected shape")
        result: dict[str, str] = {}
        for key, champion in payload["data"].items():
            if not isinstance(champion, dict):
                continue
            name = champion.get("name")
            image = champion.get("image")
            filename = image.get("full") if isinstance(image, dict) else None
            if isinstance(filename, str):
                result[str(key).casefold()] = filename
                if isinstance(name, str):
                    result[name.casefold()] = filename
        if not result:
            raise ValueError("champion metadata contained no images")
        return result

    def get_metadata(self) -> dict[str, str]:
        if self._metadata is not None:
            return self._metadata
        version = self.get_latest_version()
        if not version:
            return {}
        path = self._metadata_file(version)
        payload: Any = None
        try:
            response = self.session.get(
                f"{self.BASE_URL}/cdn/{version}/data/en_US/champion.json",
                timeout=(2, 10),
            )
            response.raise_for_status()
            payload = response.json()
            self._metadata = self._parse_metadata(payload)
            self._atomic_write(
                path,
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            )
        except (requests.RequestException, ValueError, OSError) as exc:
            self.logger.warning("Data Dragon metadata request failed: %s", exc)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self._metadata = self._parse_metadata(payload)
            except (OSError, ValueError, json.JSONDecodeError):
                self._metadata = {}
        return self._metadata

    def _portrait_path(self, version: str, filename: str) -> Path:
        return self.cache_dir / version / "portraits" / filename

    def _load_portrait(self, filename: str) -> Image | None:
        version = self.get_latest_version()
        if not version:
            return None
        path = self._portrait_path(version, filename)
        if path.exists():
            image = cv2.imread(str(path))
            if image is not None:
                return image
        try:
            response = self.session.get(
                f"{self.BASE_URL}/cdn/{version}/img/champion/{filename}",
                timeout=(2, 10),
            )
            response.raise_for_status()
            content = response.content
            decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"Could not decode {filename}")
            self._atomic_write(path, content)
            return decoded
        except (requests.RequestException, ValueError, OSError) as exc:
            self.logger.warning("Portrait request failed for %s: %s", filename, exc)
            return None

    def get_portraits(self, members: tuple[RosterMember, ...]) -> dict[str, Image]:
        metadata = self.get_metadata()
        portraits: dict[str, Image] = {}
        for member in members:
            filename = metadata.get(member.champion_name.casefold())
            if not filename:
                self.logger.warning("No Data Dragon portrait for %s", member.champion_name)
                continue
            portrait = self._load_portrait(filename)
            if portrait is not None:
                portraits[member.champion_name] = portrait
        return portraits
