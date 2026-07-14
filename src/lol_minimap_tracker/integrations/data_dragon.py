"""Data Dragon metadata and portrait cache."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..domain.interfaces import Image
from ..domain.models import RosterMember


@dataclass(frozen=True)
class _RealmInfo:
    champion_version: str
    cdn: str


class DataDragonClient:
    BASE_URL = "https://ddragon.leagueoflegends.com"
    REALM = "vn"
    DEFAULT_REALM_TTL = 6 * 60 * 60.0
    DEFAULT_METADATA_RETRY_DELAY = 30.0
    _SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")

    def __init__(
        self,
        cache_dir: Path,
        logger: logging.Logger,
        session: requests.Session | None = None,
        *,
        realm_ttl: float = DEFAULT_REALM_TTL,
        metadata_retry_delay: float = DEFAULT_METADATA_RETRY_DELAY,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(realm_ttl, bool) or not math.isfinite(realm_ttl) or realm_ttl <= 0:
            raise ValueError("realm_ttl must be a positive finite number")
        if (
            isinstance(metadata_retry_delay, bool)
            or not math.isfinite(metadata_retry_delay)
            or metadata_retry_delay <= 0
        ):
            raise ValueError("metadata_retry_delay must be a positive finite number")
        self.cache_dir = cache_dir / "ddragon"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.realm_ttl = float(realm_ttl)
        self.metadata_retry_delay = float(metadata_retry_delay)
        self.monotonic = monotonic
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
        self._realm_cache_loaded = False
        self._realm_checked_at: float | None = None
        self._realm: _RealmInfo | None = None
        self._pending_realm: _RealmInfo | None = None
        self._metadata_retry_at = 0.0
        self._metadata: dict[str, str] | None = None

    @property
    def version_file(self) -> Path:
        """Legacy cache marker retained for offline upgrades."""

        return self.cache_dir / "version.txt"

    @property
    def realm_file(self) -> Path:
        return self.cache_dir / "realms" / f"{self.REALM}.json"

    @classmethod
    def _safe_segment(cls, value: str) -> bool:
        return bool(value and value not in {".", ".."} and cls._SAFE_SEGMENT.fullmatch(value))

    @classmethod
    def _safe_png_filename(cls, value: object) -> str | None:
        if not isinstance(value, str) or not cls._safe_segment(value):
            return None
        return value if value.casefold().endswith(".png") else None

    @classmethod
    def _valid_cdn(cls, value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        candidate = value.strip().rstrip("/")
        try:
            parts = urlsplit(candidate)
            port = parts.port
        except ValueError:
            return None
        if (
            parts.scheme.casefold() != "https"
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or port not in {None, 443}
            or bool(parts.query)
            or bool(parts.fragment)
        ):
            return None
        return candidate

    @classmethod
    def _parse_realm(cls, payload: Any) -> _RealmInfo:
        if not isinstance(payload, dict) or not isinstance(payload.get("n"), dict):
            raise ValueError("vn realm had an unexpected shape")
        version = payload["n"].get("champion")
        cdn = cls._valid_cdn(payload.get("cdn"))
        if not isinstance(version, str) or not cls._safe_segment(version) or cdn is None:
            raise ValueError("vn realm had an unexpected shape")
        return _RealmInfo(version, cdn)

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except OSError:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise

    def _read_cached_realm(self) -> _RealmInfo | None:
        try:
            return self._parse_realm(json.loads(self.realm_file.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if self.realm_file.exists():
                self.logger.warning("Cached Data Dragon vn realm is unusable: %s", exc)

        # Older releases cached only the global version.  This fallback keeps a
        # previously warmed portrait cache usable while the machine is offline.
        try:
            version = self.version_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not self._safe_segment(version):
            return None
        return _RealmInfo(version, f"{self.BASE_URL}/cdn")

    @staticmethod
    def _realm_payload(realm: _RealmInfo) -> dict[str, object]:
        return {"cdn": realm.cdn, "n": {"champion": realm.champion_version}}

    def _persist_realm(self, realm: _RealmInfo) -> bool:
        try:
            self._atomic_write(
                self.realm_file,
                json.dumps(self._realm_payload(realm), separators=(",", ":")).encode("utf-8"),
            )
        except OSError as exc:
            self.logger.warning("Could not cache Data Dragon vn realm: %s", exc)
            return False
        try:
            self._atomic_write(self.version_file, realm.champion_version.encode("utf-8"))
        except OSError as exc:
            # realm_file is authoritative; version.txt exists only to migrate
            # caches written by older releases.
            self.logger.warning("Could not update legacy Data Dragon version cache: %s", exc)
        return True

    def _load_cached_realm_once(self) -> None:
        if self._realm_cache_loaded:
            return
        self._realm_cache_loaded = True
        self._realm = self._read_cached_realm()

    def _refresh_realm_if_due(self) -> None:
        self._load_cached_realm_once()
        now = self.monotonic()
        if self._realm_checked_at is not None and now - self._realm_checked_at < self.realm_ttl:
            return
        self._realm_checked_at = now
        try:
            response = self.session.get(f"{self.BASE_URL}/realms/{self.REALM}.json", timeout=(2, 5))
            response.raise_for_status()
            candidate = self._parse_realm(response.json())
        except (requests.RequestException, ValueError, OSError) as exc:
            self.logger.warning("Data Dragon vn realm request failed: %s", exc)
            return

        active = self._realm
        if active is not None and candidate.champion_version == active.champion_version:
            # A same-version CDN change does not invalidate the validated
            # champion index, but the validated realm can still be refreshed.
            if self._persist_realm(candidate):
                self._realm = candidate
                self._pending_realm = None
                self._metadata_retry_at = 0.0
            return
        # Do not expose or persist a new version until its champion catalog has
        # also validated.  That keeps the active realm and metadata a pair.
        if candidate != self._pending_realm:
            self._metadata_retry_at = 0.0
        self._pending_realm = candidate

    def get_latest_version(self) -> str | None:
        self._refresh_realm_if_due()
        realm = self._realm or self._pending_realm
        return realm.champion_version if realm is not None else None

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
            filename = (
                self._safe_png_filename(image.get("full")) if isinstance(image, dict) else None
            )
            if filename is not None:
                result[str(key).casefold()] = filename
                identifier = champion.get("id")
                if isinstance(identifier, str):
                    result[identifier.casefold()] = filename
                if isinstance(name, str):
                    result[name.casefold()] = filename
        if not result:
            raise ValueError("champion metadata contained no images")
        return result

    def _load_metadata_for(self, realm: _RealmInfo) -> dict[str, str] | None:
        path = self._metadata_file(realm.champion_version)
        try:
            cached_payload: Any = json.loads(path.read_text(encoding="utf-8"))
            return self._parse_metadata(cached_payload)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        try:
            response = self.session.get(
                f"{realm.cdn}/{realm.champion_version}/data/en_US/champion.json",
                timeout=(2, 10),
            )
            response.raise_for_status()
            payload = response.json()
            metadata = self._parse_metadata(payload)
            # Validate first, then atomically publish the new catalog.
            self._atomic_write(
                path,
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            )
            return metadata
        except (requests.RequestException, ValueError, OSError) as exc:
            self.logger.warning("Data Dragon metadata request failed: %s", exc)
            return None

    def get_metadata(self) -> dict[str, str]:
        self._refresh_realm_if_due()

        candidate = self._pending_realm
        now = self.monotonic()
        if candidate is not None and now >= self._metadata_retry_at:
            metadata = self._load_metadata_for(candidate)
            if metadata is not None and self._persist_realm(candidate):
                # Swap the validated version/catalog pair together.  Until
                # this point every caller continues to see the old pair.
                self._realm = candidate
                self._metadata = metadata
                self._pending_realm = None
                self._metadata_retry_at = 0.0
                return metadata
            self._metadata_retry_at = now + self.metadata_retry_delay

        active = self._realm
        if active is None:
            return {}
        if self._metadata is not None:
            return self._metadata
        metadata = self._load_metadata_for(active)
        if metadata is None:
            return {}
        self._metadata = metadata
        return metadata

    def _portrait_path(self, version: str, filename: str) -> Path:
        return self.cache_dir / version / "portraits" / filename

    def _load_portrait(self, filename: str) -> Image | None:
        realm = self._realm
        if realm is None:
            return None
        version = realm.champion_version
        path = self._portrait_path(version, filename)
        if path.exists():
            image = cv2.imread(str(path))
            if image is not None:
                return image
        try:
            response = self.session.get(
                f"{realm.cdn}/{version}/img/champion/{filename}",
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
            filename = (
                metadata.get(member.champion_id.casefold())
                if member.champion_id is not None
                else None
            ) or metadata.get(member.champion_name.casefold())
            if not filename:
                self.logger.warning("No Data Dragon portrait for %s", member.champion_name)
                continue
            portrait = self._load_portrait(filename)
            if portrait is not None:
                portraits[member.champion_name] = portrait
        return portraits
