"""Patch-scoped Data Dragon metadata for the manual cooldown panel.

The live client identifies participants and summoner spells, but it does not
provide enemy cooldown state.  This module resolves those identifiers against
Data Dragon and deliberately exposes only validated base cooldowns.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
import zlib
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Protocol, cast
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .._version import __version__
from ..domain.cooldowns import CooldownDefinition, EnemyCooldownLoadout
from ..domain.models import RosterMember


class CooldownMetadataUnavailable(RuntimeError):
    """Raised when no valid static catalog is available yet."""


class _HttpResponse(Protocol):
    @property
    def content(self) -> bytes: ...

    def raise_for_status(self) -> None: ...

    def json(self) -> object: ...


class _HttpSession(Protocol):
    def get(self, url: str, *, timeout: tuple[float, float]) -> _HttpResponse: ...


@dataclass(frozen=True)
class _RealmInfo:
    cdn: str
    champion_version: str
    summoner_version: str


@dataclass(frozen=True)
class _ChampionRecord:
    identifier: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class _SummonerRecord:
    identifier: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class _CatalogSnapshot:
    realm_info: _RealmInfo
    champions: Mapping[str, _ChampionRecord]
    summoners_by_id: Mapping[str, _SummonerRecord]
    summoners_by_name: Mapping[str, tuple[_SummonerRecord, ...]]


class CooldownDataDragonClient:
    """Resolve cooldown definitions and cache all required static assets."""

    BASE_URL = "https://ddragon.leagueoflegends.com"
    _ALLOWED_FOUR_RANK_ULTIMATES = frozenset({"elise", "karma", "nidalee"})
    _DYNAMIC_ULTIMATE_CHAMPIONS = frozenset(
        {
            "anivia",
            "belveth",
            "corki",
            "kassadin",
            "kogmaw",
            "quinn",
            "samira",
            "shyvana",
            "teemo",
        }
    )
    _STANDARD_SUMMONER_IDS = frozenset(
        {
            "summonerbarrier",
            "summonerboost",
            "summonerdot",
            "summonerexhaust",
            "summonerflash",
            "summonerhaste",
            "summonerheal",
            "summonermana",
            "summonersnowball",
            "summonerteleport",
        }
    )
    _SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
    _SAFE_LOCALE = re.compile(r"^[A-Za-z]{2}_[A-Za-z]{2}$")
    _PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
    _MAX_ICON_BYTES = 8 * 1024 * 1024
    _MAX_ICON_DIMENSION = 2048
    _MAX_DECODED_ICON_BYTES = 20 * 1024 * 1024
    _MAX_SAFE_SEGMENT_LENGTH = 128
    _MIN_CHAMPION_RECORDS = 100
    _MIN_SUMMONER_RECORDS = 8
    _MAX_CATALOG_RECORDS = 1_000
    _MIN_VALID_RECORD_RATIO = 0.9
    DEFAULT_REALM_TTL_SECONDS = 15 * 60.0
    DEFAULT_RETRY_BACKOFF_SECONDS = 60.0

    def __init__(
        self,
        cache_dir: Path,
        logger: logging.Logger,
        session: _HttpSession | None = None,
        realm: str = "vn",
        locale: str = "en_US",
        *,
        realm_ttl_seconds: float = DEFAULT_REALM_TTL_SECONDS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache_dir = cache_dir / "cooldowns" / "ddragon"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.realm = realm.casefold() if self._safe_segment(realm) else "vn"
        self.locale = locale if self._SAFE_LOCALE.fullmatch(locale) else "en_US"
        self._realm_ttl_seconds = self._nonnegative_interval(
            realm_ttl_seconds, self.DEFAULT_REALM_TTL_SECONDS
        )
        self._retry_backoff_seconds = self._nonnegative_interval(
            retry_backoff_seconds, self.DEFAULT_RETRY_BACKOFF_SECONDS
        )
        self._clock = clock
        self._next_refresh_at = float("-inf")

        if session is None:
            created = requests.Session()
            retry = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
            )
            created.mount("https://", HTTPAdapter(max_retries=retry))
            session = cast(_HttpSession, created)
        self.session = session
        headers = getattr(session, "headers", None)
        if isinstance(headers, MutableMapping):
            headers.setdefault("User-Agent", f"LoLMinimapTracker-PrivateResearch/{__version__}")

        self._cancelled = Event()
        self._catalog: _CatalogSnapshot | None = None
        self._pending_realm_payload: object | None = None

    @staticmethod
    def _nonnegative_interval(value: float, fallback: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return fallback
        converted = float(value)
        return converted if math.isfinite(converted) and converted >= 0.0 else fallback

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    @staticmethod
    def _safe_segment(value: str) -> bool:
        return bool(
            value
            and len(value) <= CooldownDataDragonClient._MAX_SAFE_SEGMENT_LENGTH
            and value not in {".", ".."}
            and CooldownDataDragonClient._SAFE_SEGMENT.fullmatch(value)
        )

    @staticmethod
    def _mapping(value: object) -> Mapping[str, object] | None:
        if not isinstance(value, Mapping):
            return None
        return {str(key): item for key, item in value.items()}

    @staticmethod
    def _sequence(value: object) -> tuple[object, ...] | None:
        return tuple(value) if isinstance(value, (list, tuple)) else None

    @staticmethod
    def _text(value: object) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(character for character in value.casefold() if character.isalnum())

    @classmethod
    def _filename(cls, value: object) -> str | None:
        filename = cls._text(value)
        if not filename or not cls._safe_segment(filename):
            return None
        if "/" in filename or "\\" in filename or filename in {".", ".."}:
            return None
        return filename

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

    def cancel(self) -> None:
        """Prevent further cache/network work during application shutdown."""

        self._cancelled.set()

    def _load_json(
        self,
        url: str,
        path: Path,
        validator: Callable[[object], bool],
        label: str,
        *,
        cache_first: bool = False,
    ) -> object | None:
        if self._cancelled.is_set():
            return None
        if cache_first:
            cached_payload = self._read_cached_json(path, validator, label, warn=False)
            if cached_payload is not None:
                return cached_payload
        try:
            response = self.session.get(url, timeout=(2.0, 10.0))
            response.raise_for_status()
            payload = response.json()
            if not validator(payload):
                raise ValueError(f"{label} had an unexpected shape")
            if self._cancelled.is_set():
                return None
        except (requests.RequestException, OSError, TypeError, ValueError) as exc:
            self.logger.warning("Data Dragon %s request failed: %s", label, exc)
            return self._read_cached_json(path, validator, label, warn=True)

        # A full, read-only, or temporarily locked cache must not discard a
        # catalog that already passed all structural/version checks.
        try:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self._atomic_write(path, encoded)
        except (OSError, TypeError, ValueError) as exc:
            self.logger.warning("Could not cache Data Dragon %s: %s", label, exc)
        return payload

    def _read_cached_json(
        self,
        path: Path,
        validator: Callable[[object], bool],
        label: str,
        *,
        warn: bool,
    ) -> object | None:
        if self._cancelled.is_set():
            return None
        try:
            cached_payload: object = json.loads(path.read_text(encoding="utf-8"))
            if not validator(cached_payload):
                raise ValueError(f"cached {label} had an unexpected shape")
            return cached_payload
        except (OSError, TypeError, ValueError) as exc:
            if warn or path.exists():
                self.logger.warning("No usable cached Data Dragon %s: %s", label, exc)
            return None

    def _load_realm_candidate(self, url: str, path: Path) -> object | None:
        """Fetch a realm without replacing the persistent last-known-good manifest."""

        if self._cancelled.is_set():
            return None
        try:
            response = self.session.get(url, timeout=(2.0, 10.0))
            response.raise_for_status()
            payload = response.json()
            if self._parse_realm(payload) is None:
                raise ValueError(f"{self.realm} realm had an unexpected shape")
            if self._cancelled.is_set():
                return None
            self._pending_realm_payload = payload
            return payload
        except (requests.RequestException, OSError, TypeError, ValueError) as exc:
            self.logger.warning("Data Dragon %s realm request failed: %s", self.realm, exc)

        pending = self._pending_realm_payload
        if pending is not None and self._parse_realm(pending) is not None:
            return pending
        return self._read_cached_json(
            path,
            lambda payload: self._parse_realm(payload) is not None,
            f"{self.realm} realm",
            warn=True,
        )

    def _persist_realm(self, path: Path, payload: object) -> None:
        try:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self._atomic_write(path, encoded)
        except (OSError, TypeError, ValueError) as exc:
            self.logger.warning("Could not persist last-known-good Data Dragon realm: %s", exc)

    @classmethod
    def _valid_cdn(cls, value: object) -> str | None:
        url = cls._text(value)
        if not url:
            return None
        try:
            parts = urlsplit(url)
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
        return url.rstrip("/")

    @classmethod
    def _parse_realm(cls, payload: object) -> _RealmInfo | None:
        root = cls._mapping(payload)
        if root is None:
            return None
        versions = cls._mapping(root.get("n")) or {}
        default_version = cls._text(root.get("dd")) or cls._text(root.get("v"))
        champion_version = cls._text(versions.get("champion")) or default_version
        summoner_version = cls._text(versions.get("summoner")) or default_version
        cdn = cls._valid_cdn(root.get("cdn"))
        if (
            not champion_version
            or not summoner_version
            or not cls._safe_segment(champion_version)
            or not cls._safe_segment(summoner_version)
            or not cdn
        ):
            return None
        return _RealmInfo(cdn, champion_version, summoner_version)

    @classmethod
    def _valid_catalog_header(
        cls,
        payload: object,
        *,
        expected_type: str,
        expected_version: str,
    ) -> Mapping[str, object] | None:
        root = cls._mapping(payload)
        data = cls._mapping(root.get("data")) if root is not None else None
        if (
            root is None
            or root.get("type") != expected_type
            or root.get("version") != expected_version
            or data is None
            or not 1 <= len(data) <= cls._MAX_CATALOG_RECORDS
        ):
            return None
        return data

    @classmethod
    def _valid_champion_catalog(
        cls,
        payload: object,
        expected_version: str,
        minimum_records: int = _MIN_CHAMPION_RECORDS,
    ) -> bool:
        data = cls._valid_catalog_header(
            payload,
            expected_type="champion",
            expected_version=expected_version,
        )
        required_records = max(cls._MIN_CHAMPION_RECORDS, minimum_records)
        if data is None or len(data) < required_records:
            return False
        valid = 0
        for raw_record in data.values():
            record = cls._mapping(raw_record)
            spells = cls._sequence(record.get("spells")) if record is not None else None
            if (
                record is not None
                and cls._text(record.get("id"))
                and cls._text(record.get("name"))
                and spells is not None
                and len(spells) >= 4
                and cls._mapping(spells[3]) is not None
            ):
                valid += 1
        return valid >= required_records and valid / len(data) >= cls._MIN_VALID_RECORD_RATIO

    @classmethod
    def _valid_summoner_catalog(cls, payload: object, expected_version: str) -> bool:
        data = cls._valid_catalog_header(
            payload,
            expected_type="summoner",
            expected_version=expected_version,
        )
        if data is None or len(data) < cls._MIN_SUMMONER_RECORDS:
            return False
        valid = sum(
            1
            for raw_record in data.values()
            if (record := cls._mapping(raw_record)) is not None
            and cls._text(record.get("id")) is not None
            and cls._text(record.get("name")) is not None
        )
        return (
            valid >= cls._MIN_SUMMONER_RECORDS and valid / len(data) >= cls._MIN_VALID_RECORD_RATIO
        )

    def _stage_catalog(
        self,
        realm_info: _RealmInfo,
        *,
        minimum_champion_records: int = _MIN_CHAMPION_RECORDS,
    ) -> _CatalogSnapshot | None:
        """Build a complete candidate without mutating the active catalog."""

        champion_path = (
            self.cache_dir
            / realm_info.champion_version
            / "data"
            / self.locale
            / "championFull.json"
        )
        champion_url = (
            f"{realm_info.cdn}/{realm_info.champion_version}/data/{self.locale}/championFull.json"
        )
        champion_payload = self._load_json(
            champion_url,
            champion_path,
            lambda payload: self._valid_champion_catalog(
                payload,
                realm_info.champion_version,
                minimum_champion_records,
            ),
            "champion cooldown metadata",
            cache_first=True,
        )
        if self._cancelled.is_set():
            return None

        summoner_path = (
            self.cache_dir / realm_info.summoner_version / "data" / self.locale / "summoner.json"
        )
        summoner_url = (
            f"{realm_info.cdn}/{realm_info.summoner_version}/data/{self.locale}/summoner.json"
        )
        summoner_payload = self._load_json(
            summoner_url,
            summoner_path,
            lambda payload: self._valid_summoner_catalog(payload, realm_info.summoner_version),
            "summoner spell metadata",
            cache_first=True,
        )
        if champion_payload is None or summoner_payload is None or self._cancelled.is_set():
            return None
        champions = self._index_champions(champion_payload)
        summoners_by_id, summoners_by_name = self._index_summoners(summoner_payload)
        if not champions or not summoners_by_id:
            self.logger.warning("Data Dragon catalogs contained no indexable records")
            return None
        if self._cancelled.is_set():
            return None
        return _CatalogSnapshot(
            realm_info=realm_info,
            champions=champions,
            summoners_by_id=summoners_by_id,
            summoners_by_name={name: tuple(records) for name, records in summoners_by_name.items()},
        )

    def _ensure_loaded(self) -> None:
        if self._cancelled.is_set():
            return
        now = self._now()
        if now < self._next_refresh_at:
            return

        # Install a retry deadline before doing I/O.  A failed refresh keeps the
        # last-known-good snapshot and will not block every UI refresh on the CDN.
        self._next_refresh_at = now + self._retry_backoff_seconds

        realm_url = f"{self.BASE_URL}/realms/{self.realm}.json"
        realm_path = self.cache_dir / "realms" / f"{self.realm}.json"
        realm_payload = self._load_realm_candidate(realm_url, realm_path)
        realm_info = self._parse_realm(realm_payload) if realm_payload is not None else None
        if realm_info is None or self._cancelled.is_set():
            return
        active_catalog = self._catalog
        if active_catalog is not None and realm_info == active_catalog.realm_info:
            self._pending_realm_payload = None
            self._next_refresh_at = now + self._realm_ttl_seconds
            return

        minimum_champion_records = self._MIN_CHAMPION_RECORDS
        if active_catalog is not None:
            active_champions = len(
                {record.identifier.casefold() for record in active_catalog.champions.values()}
            )
            minimum_champion_records = max(
                minimum_champion_records,
                math.ceil(active_champions * 0.8),
            )
        candidate = self._stage_catalog(
            realm_info,
            minimum_champion_records=minimum_champion_records,
        )
        if candidate is None:
            # A fresh process can still start from the persistent last-known-good
            # realm while a newly announced patch is only partly available.
            if active_catalog is None and not self._cancelled.is_set():
                fallback_payload = self._read_cached_json(
                    realm_path,
                    lambda payload: self._parse_realm(payload) is not None,
                    f"{self.realm} last-known-good realm",
                    warn=False,
                )
                fallback_info = (
                    self._parse_realm(fallback_payload) if fallback_payload is not None else None
                )
                if fallback_info is not None and fallback_info != realm_info:
                    self._catalog = self._stage_catalog(fallback_info)
            return

        # Do not expose a half-updated patch.  Both catalogs are indexed before
        # this immutable last-known-good snapshot is replaced with one assignment.
        self._catalog = candidate
        self._persist_realm(realm_path, realm_payload)
        self._pending_realm_payload = None
        self._next_refresh_at = now + self._realm_ttl_seconds

    @classmethod
    def _index_champions(cls, payload: object | None) -> dict[str, _ChampionRecord]:
        root = cls._mapping(payload)
        data = cls._mapping(root.get("data")) if root is not None else None
        result: dict[str, _ChampionRecord] = {}
        if data is None:
            return result
        for key, raw_record in data.items():
            record = cls._mapping(raw_record)
            if record is None:
                continue
            identifier = cls._text(record.get("id")) or key
            indexed = _ChampionRecord(identifier, record)
            for alias in (key, identifier, cls._text(record.get("name"))):
                if alias:
                    result[cls._normalize(alias)] = indexed
        return result

    @classmethod
    def _index_summoners(
        cls, payload: object | None
    ) -> tuple[dict[str, _SummonerRecord], dict[str, list[_SummonerRecord]]]:
        root = cls._mapping(payload)
        data = cls._mapping(root.get("data")) if root is not None else None
        by_id: dict[str, _SummonerRecord] = {}
        by_name: dict[str, list[_SummonerRecord]] = {}
        if data is None:
            return by_id, by_name
        for key, raw_record in data.items():
            record = cls._mapping(raw_record)
            if record is None:
                continue
            identifier = cls._text(record.get("id")) or key
            indexed = _SummonerRecord(identifier, record)
            for alias in (key, identifier):
                by_id[cls._normalize(alias)] = indexed
            name = cls._text(record.get("name"))
            if name:
                by_name.setdefault(cls._normalize(name), []).append(indexed)
        return by_id, by_name

    @classmethod
    def _image_filename(cls, payload: Mapping[str, object]) -> str | None:
        image = cls._mapping(payload.get("image"))
        return cls._filename(image.get("full")) if image is not None else None

    @classmethod
    def _valid_png(cls, content: bytes) -> bool:
        if not content.startswith(cls._PNG_SIGNATURE) or len(content) > cls._MAX_ICON_BYTES:
            return False
        offset = len(cls._PNG_SIGNATURE)
        seen_header = False
        image_data = bytearray()
        while offset + 12 <= len(content):
            length = int.from_bytes(content[offset : offset + 4], "big")
            chunk_end = offset + 12 + length
            if chunk_end > len(content):
                return False
            chunk_type = content[offset + 4 : offset + 8]
            chunk_data = content[offset + 8 : offset + 8 + length]
            expected_crc = int.from_bytes(content[offset + 8 + length : chunk_end], "big")
            actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                return False
            if not seen_header:
                if chunk_type != b"IHDR" or length != 13:
                    return False
                width = int.from_bytes(chunk_data[0:4], "big")
                height = int.from_bytes(chunk_data[4:8], "big")
                if not (
                    0 < width <= cls._MAX_ICON_DIMENSION and 0 < height <= cls._MAX_ICON_DIMENSION
                ):
                    return False
                seen_header = True
            elif chunk_type == b"IHDR":
                return False
            if chunk_type == b"IDAT":
                image_data.extend(chunk_data)
            if chunk_type == b"IEND":
                if length != 0 or not seen_header or not image_data or chunk_end != len(content):
                    return False
                try:
                    decoder = zlib.decompressobj()
                    decoded = decoder.decompress(image_data, cls._MAX_DECODED_ICON_BYTES + 1)
                except zlib.error:
                    return False
                return (
                    0 < len(decoded) <= cls._MAX_DECODED_ICON_BYTES
                    and decoder.eof
                    and not decoder.unconsumed_tail
                    and not decoder.unused_data
                )
            offset = chunk_end
        return False

    @classmethod
    def _valid_cached_png(cls, path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                content = handle.read(cls._MAX_ICON_BYTES + 1)
            return cls._valid_png(content)
        except OSError:
            return False

    def _icon_path(
        self,
        realm: _RealmInfo,
        version: str,
        category: str,
        filename: str,
    ) -> Path | None:
        if self._cancelled.is_set():
            return None
        path = self.cache_dir / version / "img" / category / filename
        if self._valid_cached_png(path):
            return path
        url = f"{realm.cdn}/{version}/img/{category}/{filename}"
        try:
            response = self.session.get(url, timeout=(2.0, 10.0))
            response.raise_for_status()
            content = response.content
            if not isinstance(content, bytes) or not self._valid_png(content):
                raise ValueError(f"{filename} was not a PNG image")
            if self._cancelled.is_set():
                return None
            self._atomic_write(path, content)
            return path
        except (requests.RequestException, OSError, TypeError, ValueError) as exc:
            self.logger.warning("Data Dragon icon request failed for %s: %s", filename, exc)
            return None

    @staticmethod
    def _positive_cooldowns(value: object) -> tuple[float, ...] | None:
        if not isinstance(value, (list, tuple)) or not value:
            return None
        result: list[float] = []
        for raw in value:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return None
            cooldown = float(raw)
            if not math.isfinite(cooldown) or cooldown <= 0.0:
                return None
            result.append(cooldown)
        return tuple(result)

    @staticmethod
    def _max_rank(value: object) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0

    @staticmethod
    def _unsupported_definition(
        identifier: str,
        display_name: str,
        reason: str,
        *,
        icon_path: Path | None = None,
        cooldowns: tuple[float, ...] = (),
        max_rank: int = 0,
    ) -> CooldownDefinition:
        return CooldownDefinition(
            identifier=identifier,
            display_name=display_name,
            icon_path=icon_path,
            cooldowns=cooldowns,
            max_rank=max_rank,
            unsupported_reason=reason,
        )

    def _ultimate_definition(
        self,
        catalog: _CatalogSnapshot,
        champion: _ChampionRecord,
    ) -> tuple[CooldownDefinition, Path | None]:
        realm = catalog.realm_info
        champion_payload = champion.payload
        portrait_filename = self._image_filename(champion_payload)
        portrait_path = (
            self._icon_path(realm, realm.champion_version, "champion", portrait_filename)
            if portrait_filename
            else None
        )

        raw_spells = champion_payload.get("spells")
        spells = raw_spells if isinstance(raw_spells, (list, tuple)) else ()
        ultimate_payload = self._mapping(spells[3]) if len(spells) > 3 else None
        fallback_id = f"{champion.identifier}R"
        if ultimate_payload is None:
            return (
                self._unsupported_definition(
                    fallback_id,
                    "Ultimate (R)",
                    "Ultimate metadata is unavailable",
                ),
                portrait_path,
            )

        identifier = self._text(ultimate_payload.get("id")) or fallback_id
        display_name = self._text(ultimate_payload.get("name")) or "Ultimate (R)"
        ultimate_filename = self._image_filename(ultimate_payload)
        icon_path = (
            self._icon_path(realm, realm.champion_version, "spell", ultimate_filename)
            if ultimate_filename
            else None
        )
        max_rank = self._max_rank(ultimate_payload.get("maxrank"))
        cooldowns = self._positive_cooldowns(ultimate_payload.get("cooldown"))
        normalized_champion = self._normalize(champion.identifier)

        supported_rank = normalized_champion != "udyr" and (
            max_rank == 3
            or (max_rank == 1 and normalized_champion == "jayce")
            or (max_rank == 4 and normalized_champion in self._ALLOWED_FOUR_RANK_ULTIMATES)
        )
        reason = None if supported_rank else "Ultimate rank cannot be inferred from champion level"
        if normalized_champion in self._DYNAMIC_ULTIMATE_CHAMPIONS:
            reason = "Ultimate uses charge, resource, toggle, or repeat-cast behavior"
        if cooldowns is None or len(cooldowns) != max_rank:
            cooldowns = ()
            reason = "Base ultimate cooldown is dynamic or unavailable"

        return (
            CooldownDefinition(
                identifier=identifier,
                display_name=display_name,
                icon_path=icon_path,
                cooldowns=cooldowns,
                max_rank=max_rank,
                unsupported_reason=reason,
            ),
            portrait_path,
        )

    def _preferred_summoner(self, candidates: list[_SummonerRecord]) -> _SummonerRecord:
        def score(candidate: _SummonerRecord) -> tuple[int, int, str]:
            normalized_id = self._normalize(candidate.identifier)
            standard = 0 if normalized_id in self._STANDARD_SUMMONER_IDS else 1
            raw_modes = candidate.payload.get("modes")
            modes = raw_modes if isinstance(raw_modes, (list, tuple)) else ()
            classic = 0 if any(mode == "CLASSIC" for mode in modes) else 1
            return standard, classic, normalized_id

        return min(candidates, key=score)

    def _resolve_summoner(
        self,
        catalog: _CatalogSnapshot,
        identifier: str | None,
        display_name: str,
    ) -> _SummonerRecord | None:
        if identifier:
            exact = catalog.summoners_by_id.get(self._normalize(identifier))
            if exact is not None:
                return exact
        candidates = catalog.summoners_by_name.get(self._normalize(display_name), ())
        return self._preferred_summoner(list(candidates)) if candidates else None

    @staticmethod
    def _reference_values(reference: object) -> tuple[str | None, str]:
        raw_identifier = getattr(reference, "identifier", None)
        raw_display_name = getattr(reference, "display_name", None)
        identifier = raw_identifier.strip() if isinstance(raw_identifier, str) else None
        display_name = raw_display_name.strip() if isinstance(raw_display_name, str) else ""
        return identifier or None, display_name

    def _summoner_definition(
        self,
        catalog: _CatalogSnapshot | None,
        reference: object | None,
        slot_number: int,
    ) -> CooldownDefinition:
        identifier, requested_name = self._reference_values(reference) if reference else (None, "")
        fallback_id = identifier or f"unknown-summoner-{slot_number}"
        fallback_name = requested_name or f"Summoner spell {slot_number}"
        record = self._resolve_summoner(catalog, identifier, requested_name) if catalog else None
        if record is None or catalog is None:
            return self._unsupported_definition(
                fallback_id,
                fallback_name,
                "Summoner spell metadata is unavailable",
            )

        payload = record.payload
        display_name = self._text(payload.get("name")) or fallback_name
        filename = self._image_filename(payload)
        realm = catalog.realm_info
        icon_path = (
            self._icon_path(realm, realm.summoner_version, "spell", filename) if filename else None
        )
        max_rank = self._max_rank(payload.get("maxrank"))
        cooldowns = self._positive_cooldowns(payload.get("cooldown"))
        normalized_id = self._normalize(record.identifier)
        is_smite = "smite" in normalized_id or self._normalize(display_name) == "smite"

        reason: str | None = None
        if is_smite:
            reason = "Smite uses charge-based cooldown behavior"
        elif max_rank != 1:
            reason = "Summoner spell rank metadata is unsupported"
        if cooldowns is None or len(cooldowns) != max_rank:
            cooldowns = ()
            reason = "Base summoner spell cooldown is dynamic or unavailable"

        return CooldownDefinition(
            identifier=record.identifier,
            display_name=display_name,
            icon_path=icon_path,
            cooldowns=cooldowns,
            max_rank=max_rank,
            unsupported_reason=reason,
        )

    def _unknown_loadout(
        self,
        catalog: _CatalogSnapshot | None,
        member: RosterMember,
        reason: str,
    ) -> EnemyCooldownLoadout:
        champion_name = member.champion_name.strip() or "Unknown champion"
        participant_id = str(getattr(member, "participant_id", "")).strip() or champion_name
        ultimate = self._unsupported_definition(
            f"{self._normalize(champion_name) or 'unknown'}:R",
            "Ultimate (R)",
            reason,
        )
        references = getattr(member, "summoner_spells", ())
        spell_refs = references if isinstance(references, (list, tuple)) else ()
        summoners = tuple(
            self._summoner_definition(
                catalog,
                spell_refs[index] if index < len(spell_refs) else None,
                index + 1,
            )
            for index in range(2)
        )
        return EnemyCooldownLoadout(
            participant_id=participant_id,
            champion_name=champion_name,
            champion_icon_path=None,
            ultimate=ultimate,
            summoner_spells=(summoners[0], summoners[1]),
        )

    def _loadout(
        self,
        catalog: _CatalogSnapshot,
        member: RosterMember,
    ) -> EnemyCooldownLoadout:
        champion_name = member.champion_name.strip() or "Unknown champion"
        participant_id = str(getattr(member, "participant_id", "")).strip() or champion_name
        champion_id = getattr(member, "champion_id", None)
        champion = (
            catalog.champions.get(self._normalize(champion_id))
            if isinstance(champion_id, str) and champion_id.strip()
            else None
        )
        champion = champion or catalog.champions.get(self._normalize(champion_name))
        if champion is None:
            return self._unknown_loadout(catalog, member, "Champion metadata is unavailable")

        ultimate, portrait_path = self._ultimate_definition(catalog, champion)
        references = getattr(member, "summoner_spells", ())
        spell_refs = references if isinstance(references, (list, tuple)) else ()
        summoners = tuple(
            self._summoner_definition(
                catalog,
                spell_refs[index] if index < len(spell_refs) else None,
                index + 1,
            )
            for index in range(2)
        )
        return EnemyCooldownLoadout(
            participant_id=participant_id,
            champion_name=champion_name,
            champion_icon_path=portrait_path,
            ultimate=ultimate,
            summoner_spells=(summoners[0], summoners[1]),
        )

    def get_loadouts(self, members: Iterable[RosterMember]) -> tuple[EnemyCooldownLoadout, ...]:
        """Return one safe, complete loadout per roster member.

        Missing and malformed Data Dragon entries are represented as disabled
        definitions so callers can keep the panel layout stable.
        """

        self._ensure_loaded()
        catalog = self._catalog
        if catalog is None and not self._cancelled.is_set():
            raise CooldownMetadataUnavailable(
                "Data Dragon realm or cooldown catalogs are temporarily unavailable"
            )
        if catalog is None:
            return tuple(
                self._unknown_loadout(None, member, "Champion metadata is unavailable")
                for member in members
            )
        return tuple(self._loadout(catalog, member) for member in members)
