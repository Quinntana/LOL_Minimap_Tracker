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

from .._version import __version__
from ..domain.interfaces import Image
from ..domain.models import RosterMember


@dataclass(frozen=True)
class _RealmInfo:
    champion_version: str
    cdn: str


class DataDragonClient:
    BASE_URL = "https://ddragon.leagueoflegends.com"
    REALM = "vn"
    DEFAULT_REALM_TTL = 15 * 60.0
    DEFAULT_REALM_RETRY_DELAY = 30.0
    DEFAULT_METADATA_RETRY_DELAY = 30.0
    MIN_CHAMPION_RECORDS = 100
    MIN_VALID_RECORD_RATIO = 0.9
    MIN_CATALOG_RETENTION_RATIO = 0.8
    MAX_CHAMPION_RECORDS = 1_000
    MAX_TEXT_LENGTH = 128
    _SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")

    def __init__(
        self,
        cache_dir: Path,
        logger: logging.Logger,
        session: requests.Session | None = None,
        *,
        realm_ttl: float = DEFAULT_REALM_TTL,
        realm_retry_delay: float = DEFAULT_REALM_RETRY_DELAY,
        metadata_retry_delay: float = DEFAULT_METADATA_RETRY_DELAY,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(realm_ttl, bool) or not math.isfinite(realm_ttl) or realm_ttl <= 0:
            raise ValueError("realm_ttl must be a positive finite number")
        if (
            isinstance(realm_retry_delay, bool)
            or not math.isfinite(realm_retry_delay)
            or realm_retry_delay <= 0
        ):
            raise ValueError("realm_retry_delay must be a positive finite number")
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
        self.realm_retry_delay = float(realm_retry_delay)
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
        self.session.headers.setdefault(
            "User-Agent", f"LoLMinimapTracker-PrivateResearch/{__version__}"
        )
        self._realm_cache_loaded = False
        self._realm_checked_at: float | None = None
        self._realm_retry_at = 0.0
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
        return bool(
            value
            and len(value) <= cls.MAX_TEXT_LENGTH
            and value not in {".", ".."}
            and cls._SAFE_SEGMENT.fullmatch(value)
        )

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
        if not isinstance(payload, dict):
            raise ValueError("vn realm had an unexpected shape")
        versions = payload.get("n")
        default_version = payload.get("dd") or payload.get("v")
        version = versions.get("champion") if isinstance(versions, dict) else None
        version = version or default_version
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

    def _persist_realm(self, realm: _RealmInfo) -> None:
        try:
            self._atomic_write(
                self.realm_file,
                json.dumps(self._realm_payload(realm), separators=(",", ":")).encode("utf-8"),
            )
        except OSError as exc:
            self.logger.warning("Could not cache Data Dragon vn realm: %s", exc)
            return
        try:
            self._atomic_write(self.version_file, realm.champion_version.encode("utf-8"))
        except OSError as exc:
            # realm_file is authoritative; version.txt exists only to migrate
            # caches written by older releases.
            self.logger.warning("Could not update legacy Data Dragon version cache: %s", exc)

    def _load_cached_realm_once(self) -> None:
        if self._realm_cache_loaded:
            return
        self._realm_cache_loaded = True
        self._realm = self._read_cached_realm()

    def _refresh_realm_if_due(self) -> None:
        self._load_cached_realm_once()
        now = self.monotonic()
        if now < self._realm_retry_at:
            return
        if self._realm_checked_at is not None and now - self._realm_checked_at < self.realm_ttl:
            return
        try:
            response = self.session.get(f"{self.BASE_URL}/realms/{self.REALM}.json", timeout=(2, 5))
            response.raise_for_status()
            candidate = self._parse_realm(response.json())
        except (requests.RequestException, ValueError, OSError) as exc:
            self.logger.warning("Data Dragon vn realm request failed: %s", exc)
            self._realm_retry_at = now + self.realm_retry_delay
            return
        self._realm_checked_at = now
        self._realm_retry_at = 0.0

        active = self._realm
        if active is not None and candidate.champion_version == active.champion_version:
            # A same-version CDN change does not invalidate the validated
            # champion index, but the validated realm can still be refreshed.
            self._persist_realm(candidate)
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

    def _parse_metadata(self, payload: Any, expected_version: str) -> dict[str, str]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise ValueError("champion metadata had an unexpected shape")
        if (
            payload.get("type") != "champion"
            or payload.get("format") != "standAloneComplex"
            or payload.get("version") != expected_version
        ):
            raise ValueError("champion metadata did not match the requested Data Dragon version")
        raw_records = payload["data"]
        if not self.MIN_CHAMPION_RECORDS <= len(raw_records) <= self.MAX_CHAMPION_RECORDS:
            raise ValueError("champion metadata had an implausible record count")
        result: dict[str, str] = {}
        valid_records = 0
        for key, champion in raw_records.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key) > self.MAX_TEXT_LENGTH
                or not isinstance(champion, dict)
            ):
                continue
            name = champion.get("name")
            image = champion.get("image")
            filename = (
                self._safe_png_filename(image.get("full")) if isinstance(image, dict) else None
            )
            if filename is not None:
                valid_records += 1
                result[key.casefold()] = filename
                identifier = champion.get("id")
                if isinstance(identifier, str) and 0 < len(identifier) <= self.MAX_TEXT_LENGTH:
                    result[identifier.casefold()] = filename
                if isinstance(name, str) and 0 < len(name) <= self.MAX_TEXT_LENGTH:
                    result[name.casefold()] = filename
        if (
            valid_records < self.MIN_CHAMPION_RECORDS
            or valid_records / len(raw_records) < self.MIN_VALID_RECORD_RATIO
        ):
            raise ValueError(
                "champion metadata was incomplete "
                f"({valid_records} valid records; expected at least {self.MIN_CHAMPION_RECORDS})"
            )
        return result

    @staticmethod
    def _record_count(metadata: dict[str, str]) -> int:
        return len(set(metadata.values()))

    def _read_cached_metadata_for(self, realm: _RealmInfo) -> dict[str, str] | None:
        path = self._metadata_file(realm.champion_version)
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
            return self._parse_metadata(payload, realm.champion_version)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _load_metadata_for(
        self,
        realm: _RealmInfo,
        *,
        minimum_records: int = MIN_CHAMPION_RECORDS,
    ) -> dict[str, str] | None:
        path = self._metadata_file(realm.champion_version)
        cached = self._read_cached_metadata_for(realm)
        if cached is not None and self._record_count(cached) >= minimum_records:
            return cached
        try:
            response = self.session.get(
                f"{realm.cdn}/{realm.champion_version}/data/en_US/champion.json",
                timeout=(2, 10),
            )
            response.raise_for_status()
            payload = response.json()
            metadata = self._parse_metadata(payload, realm.champion_version)
            record_count = self._record_count(metadata)
            if record_count < minimum_records:
                raise ValueError(
                    "champion metadata shrank unexpectedly "
                    f"({record_count} records; expected at least {minimum_records})"
                )
        except (requests.RequestException, ValueError, OSError) as exc:
            self.logger.warning("Data Dragon metadata request failed: %s", exc)
            return None

        # Cache persistence is best effort. A full/read-only disk must not make
        # already validated network data unusable for the current process.
        try:
            self._atomic_write(
                path,
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            )
        except OSError as exc:
            self.logger.warning("Could not cache Data Dragon metadata: %s", exc)
        return metadata

    def get_metadata(self) -> dict[str, str]:
        self._refresh_realm_if_due()

        candidate = self._pending_realm
        now = self.monotonic()
        if candidate is not None and now >= self._metadata_retry_at:
            baseline = self._metadata
            if baseline is None and self._realm is not None:
                baseline = self._read_cached_metadata_for(self._realm)
            minimum_records = self.MIN_CHAMPION_RECORDS
            if baseline is not None:
                minimum_records = max(
                    minimum_records,
                    math.ceil(self._record_count(baseline) * self.MIN_CATALOG_RETENTION_RATIO),
                )
            metadata = self._load_metadata_for(
                candidate,
                minimum_records=minimum_records,
            )
            if metadata is not None:
                # Swap the validated version/catalog pair together.  Until
                # this point every caller continues to see the old pair.
                self._persist_realm(candidate)
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
        except (requests.RequestException, ValueError, OSError) as exc:
            self.logger.warning("Portrait request failed for %s: %s", filename, exc)
            return None
        try:
            self._atomic_write(path, content)
        except OSError as exc:
            self.logger.warning("Could not cache portrait %s: %s", filename, exc)
        return decoded

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
