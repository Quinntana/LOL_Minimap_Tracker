"""Perform an explicit, non-destructive Data Dragon contract check.

This script intentionally runs outside the ordinary test workflow: it checks
Riot's live VN realm, the realm-selected champion catalog, and one portrait.
The parsing helpers are deterministic so the contract rules remain covered by
offline unit tests.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REALM_URL = "https://ddragon.leagueoflegends.com/realms/vn.json"
MIN_CHAMPION_RECORDS = 100
MAX_CHAMPION_RECORDS = 1_000
MIN_VALID_RECORD_RATIO = 0.9
MAX_TEXT_LENGTH = 128
MAX_PORTRAIT_BYTES = 2 * 1024 * 1024
MAX_PORTRAIT_DIMENSION = 1_024
MAX_PORTRAIT_PIXELS = MAX_PORTRAIT_DIMENSION**2
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class RealmInfo:
    """The champion version and CDN selected by Riot's VN realm."""

    champion_version: str
    cdn: str


@dataclass(frozen=True)
class CatalogInfo:
    """Validated champion catalog facts needed by the live check."""

    record_count: int
    valid_record_count: int
    portrait_filename: str


@dataclass(frozen=True)
class VerificationResult:
    """Summary printed by the scheduled contract workflow."""

    champion_version: str
    cdn: str
    champion_count: int
    valid_champion_count: int
    portrait_filename: str
    portrait_width: int
    portrait_height: int


def safe_segment(value: object) -> str | None:
    """Return a path-safe Data Dragon segment, or ``None`` when unsafe."""

    if not isinstance(value, str) or len(value) > MAX_TEXT_LENGTH or value in {"", ".", ".."}:
        return None
    return value if _SAFE_SEGMENT.fullmatch(value) else None


def safe_png_filename(value: object) -> str | None:
    """Validate the untrusted portrait filename from champion metadata."""

    segment = safe_segment(value)
    if segment is None or not segment.casefold().endswith(".png"):
        return None
    return segment


def validate_cdn(value: object) -> str:
    """Validate a realm-provided HTTPS CDN without pinning a mutable hostname."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("VN realm did not provide a CDN URL")
    candidate = value.strip().rstrip("/")
    try:
        parts = urlsplit(candidate)
        port = parts.port
    except ValueError as exc:
        raise ValueError("VN realm provided an invalid CDN URL") from exc
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or port not in {None, 443}
        or bool(parts.query)
        or bool(parts.fragment)
    ):
        raise ValueError("VN realm provided an invalid CDN URL")
    return candidate


def parse_realm(payload: object) -> RealmInfo:
    """Parse Riot's VN realm contract, including its documented fallbacks."""

    if not isinstance(payload, Mapping):
        raise ValueError("VN realm had an unexpected shape")
    versions = payload.get("n")
    version = versions.get("champion") if isinstance(versions, Mapping) else None
    version = version or payload.get("dd") or payload.get("v")
    safe_version = safe_segment(version)
    if safe_version is None:
        raise ValueError("VN realm did not provide a safe champion version")
    return RealmInfo(champion_version=safe_version, cdn=validate_cdn(payload.get("cdn")))


def validate_catalog(payload: object, expected_version: str) -> CatalogInfo:
    """Validate the full champion index using the runtime client's invariants."""

    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), Mapping):
        raise ValueError("champion.json had an unexpected shape")
    if (
        payload.get("type") != "champion"
        or payload.get("format") != "standAloneComplex"
        or payload.get("version") != expected_version
    ):
        raise ValueError("champion.json did not match the realm-selected version")

    records = payload["data"]
    record_count = len(records)
    if not MIN_CHAMPION_RECORDS <= record_count <= MAX_CHAMPION_RECORDS:
        raise ValueError(f"champion.json had an implausible record count ({record_count})")

    filenames: list[str] = []
    for key, champion in records.items():
        if (
            not isinstance(key, str)
            or not key
            or len(key) > MAX_TEXT_LENGTH
            or not isinstance(champion, Mapping)
        ):
            continue
        image = champion.get("image")
        filename = safe_png_filename(image.get("full")) if isinstance(image, Mapping) else None
        if filename is not None:
            filenames.append(filename)

    valid_count = len(filenames)
    valid_ratio = valid_count / record_count
    if valid_count < MIN_CHAMPION_RECORDS or valid_ratio < MIN_VALID_RECORD_RATIO:
        raise ValueError(
            "champion.json was incomplete "
            f"({valid_count}/{record_count} valid portrait records; ratio={valid_ratio:.3f})"
        )
    return CatalogInfo(
        record_count=record_count,
        valid_record_count=valid_count,
        portrait_filename=min(filenames, key=lambda filename: (filename.casefold(), filename)),
    )


def read_bounded_png(response: requests.Response, limit: int = MAX_PORTRAIT_BYTES) -> bytes:
    """Read a PNG response without allowing an unbounded CDN payload."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("portrait byte limit must be a positive integer")
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
    if content_type != "image/png":
        raise ValueError(f"portrait had an unexpected content type ({content_type or 'missing'})")

    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise ValueError("portrait had an invalid Content-Length") from exc
        if declared_size < 0 or declared_size > limit:
            raise ValueError(f"portrait exceeded the {limit}-byte limit")

    content = bytearray()
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        content.extend(chunk)
        if len(content) > limit:
            raise ValueError(f"portrait exceeded the {limit}-byte limit")
    if not content:
        raise ValueError("portrait response was empty")
    return bytes(content)


def validate_portrait_png(content: bytes) -> tuple[int, int]:
    """Validate bounded PNG dimensions, decode it, and return width/height."""

    if len(content) < 24 or content[:8] != _PNG_SIGNATURE or content[12:16] != b"IHDR":
        raise ValueError("portrait was not a structurally valid PNG")
    width = int.from_bytes(content[16:20], "big")
    height = int.from_bytes(content[20:24], "big")
    if (
        width <= 0
        or height <= 0
        or width > MAX_PORTRAIT_DIMENSION
        or height > MAX_PORTRAIT_DIMENSION
        or width * height > MAX_PORTRAIT_PIXELS
    ):
        raise ValueError(f"portrait had unsafe dimensions ({width}x{height})")

    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.shape[0] != height or decoded.shape[1] != width:
        raise ValueError("portrait could not be decoded as the advertised PNG dimensions")
    return width, height


def build_session() -> requests.Session:
    """Build a retrying session for the opt-in live contract check."""

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "LoLMinimapTracker-DataDragonContract/1.0"
    return session


def _json_payload(response: requests.Response, label: str) -> Any:
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise ValueError(f"{label} did not contain valid JSON") from exc


def verify(session: requests.Session | None = None) -> VerificationResult:
    """Check the current VN realm, selected catalog, and one live portrait."""

    owns_session = session is None
    active_session = session or build_session()
    try:
        realm_response = active_session.get(REALM_URL, timeout=(5, 15))
        realm_response.raise_for_status()
        realm = parse_realm(_json_payload(realm_response, "VN realm"))

        metadata_url = f"{realm.cdn}/{realm.champion_version}/data/en_US/champion.json"
        metadata_response = active_session.get(metadata_url, timeout=(5, 20))
        metadata_response.raise_for_status()
        catalog = validate_catalog(
            _json_payload(metadata_response, "champion.json"), realm.champion_version
        )

        portrait_url = (
            f"{realm.cdn}/{realm.champion_version}/img/champion/{catalog.portrait_filename}"
        )
        portrait_response = active_session.get(
            portrait_url,
            timeout=(5, 20),
            stream=True,
        )
        try:
            portrait_response.raise_for_status()
            portrait = read_bounded_png(portrait_response)
        finally:
            portrait_response.close()
        width, height = validate_portrait_png(portrait)
        return VerificationResult(
            champion_version=realm.champion_version,
            cdn=realm.cdn,
            champion_count=catalog.record_count,
            valid_champion_count=catalog.valid_record_count,
            portrait_filename=catalog.portrait_filename,
            portrait_width=width,
            portrait_height=height,
        )
    finally:
        if owns_session:
            active_session.close()


if __name__ == "__main__":
    checked = verify()
    print(
        "Data Dragon VN contract OK: "
        f"version={checked.champion_version}, cdn={checked.cdn}, "
        f"champions={checked.valid_champion_count}/{checked.champion_count}, "
        f"portrait={checked.portrait_filename} "
        f"({checked.portrait_width}x{checked.portrait_height})"
    )
