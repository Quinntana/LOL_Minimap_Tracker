from __future__ import annotations

import base64
import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import requests


def _load_verifier() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "tools" / "verify_ddragon.py"
    spec = importlib.util.spec_from_file_location("verify_ddragon", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load tools/verify_ddragon.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()
MAX_PORTRAIT_BYTES = verifier.MAX_PORTRAIT_BYTES
REALM_URL = verifier.REALM_URL
CatalogInfo = verifier.CatalogInfo
parse_realm = verifier.parse_realm
read_bounded_png = verifier.read_bounded_png
safe_png_filename = verifier.safe_png_filename
validate_catalog = verifier.validate_catalog
validate_cdn = verifier.validate_cdn
validate_portrait_png = verifier.validate_portrait_png
verify = verifier.verify

_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _catalog(version: str = "16.13.1", count: int = 100) -> dict[str, object]:
    return {
        "type": "champion",
        "format": "standAloneComplex",
        "version": version,
        "data": {
            f"Champion{index}": {
                "id": f"Champion{index}",
                "name": f"Champion {index}",
                "image": {"full": f"Champion{index}.png"},
            }
            for index in range(count)
        },
    }


class _Response:
    def __init__(
        self,
        *,
        payload: object | None = None,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.content = content
        self.headers = headers or {}
        self.closed = False

    def json(self) -> object:
        return self.payload

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def test_parse_vn_realm_uses_champion_version_and_normalizes_cdn() -> None:
    realm = parse_realm(
        {
            "n": {"champion": "16.13.1"},
            "dd": "ignored",
            "cdn": "https://cdn.example.test/cdn/",
        }
    )

    assert realm.champion_version == "16.13.1"
    assert realm.cdn == "https://cdn.example.test/cdn"


def test_parse_vn_realm_supports_default_version_fallback() -> None:
    realm = parse_realm({"n": {}, "dd": "16.13.1", "cdn": "https://cdn.example.test"})

    assert realm.champion_version == "16.13.1"


@pytest.mark.parametrize(
    "cdn",
    [
        "http://cdn.example.test/cdn",
        "https://user@cdn.example.test/cdn",
        "https://cdn.example.test:444/cdn",
        "https://cdn.example.test/cdn?version=1",
        "//cdn.example.test/cdn",
        "not a URL",
    ],
)
def test_validate_cdn_rejects_unsafe_values(cdn: str) -> None:
    with pytest.raises(ValueError, match="invalid CDN URL"):
        validate_cdn(cdn)


@pytest.mark.parametrize(
    "filename",
    ["../Ahri.png", "nested/Ahri.png", "Ahri.jpg", "Ahri.png?x=1", "", None],
)
def test_safe_png_filename_rejects_unsafe_values(filename: object) -> None:
    assert safe_png_filename(filename) is None


def test_validate_catalog_returns_deterministic_portrait() -> None:
    payload = _catalog()
    records = payload["data"]
    assert isinstance(records, dict)
    records["Champion0"]["image"]["full"] = "Zed.png"
    records["Champion1"]["image"]["full"] = "aatrox.png"

    result = validate_catalog(payload, "16.13.1")

    assert result == CatalogInfo(100, 100, "aatrox.png")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "championFull"),
        ("format", "standAloneComplex2"),
        ("version", "99.1.1"),
    ],
)
def test_validate_catalog_rejects_contract_mismatch(field: str, value: str) -> None:
    payload = _catalog()
    payload[field] = value

    with pytest.raises(ValueError, match="realm-selected version"):
        validate_catalog(payload, "16.13.1")


def test_validate_catalog_rejects_implausibly_small_catalog() -> None:
    with pytest.raises(ValueError, match="implausible record count"):
        validate_catalog(_catalog(count=99), "16.13.1")


def test_validate_catalog_requires_minimum_valid_count_and_ratio() -> None:
    payload = _catalog(count=112)
    records = payload["data"]
    assert isinstance(records, dict)
    for index in range(12):
        records[f"Champion{index}"]["image"]["full"] = "../unsafe.png"

    with pytest.raises(ValueError, match=r"100/112 valid portrait records"):
        validate_catalog(payload, "16.13.1")


def test_validate_catalog_accepts_exact_minimum_ratio() -> None:
    payload = _catalog(count=120)
    records = payload["data"]
    assert isinstance(records, dict)
    for index in range(12):
        records[f"Champion{index}"]["image"]["full"] = "../unsafe.png"

    result = validate_catalog(payload, "16.13.1")

    assert result.valid_record_count == 108


def test_read_bounded_png_rejects_declared_or_streamed_oversize() -> None:
    declared = _Response(
        content=b"small",
        headers={"Content-Type": "image/png", "Content-Length": str(MAX_PORTRAIT_BYTES + 1)},
    )
    streamed = _Response(content=b"12345", headers={"Content-Type": "image/png"})

    with pytest.raises(ValueError, match="exceeded"):
        read_bounded_png(declared)
    with pytest.raises(ValueError, match="exceeded"):
        read_bounded_png(streamed, limit=4)


def test_read_bounded_png_requires_png_content_type() -> None:
    response = _Response(content=_ONE_PIXEL_PNG, headers={"Content-Type": "text/html"})

    with pytest.raises(ValueError, match="content type"):
        read_bounded_png(response)


def test_validate_portrait_png_decodes_bounded_dimensions() -> None:
    assert validate_portrait_png(_ONE_PIXEL_PNG) == (1, 1)

    oversized_header = bytearray(_ONE_PIXEL_PNG)
    oversized_header[16:20] = (2_000).to_bytes(4, "big")
    with pytest.raises(ValueError, match="unsafe dimensions"):
        validate_portrait_png(bytes(oversized_header))


def test_verify_uses_vn_realm_selected_version_and_cdn() -> None:
    portrait_response = _Response(
        content=_ONE_PIXEL_PNG,
        headers={"Content-Type": "image/png", "Content-Length": str(len(_ONE_PIXEL_PNG))},
    )
    session = _Session(
        [
            _Response(payload={"n": {"champion": "16.13.1"}, "cdn": "https://cdn.test/cdn"}),
            _Response(payload=_catalog()),
            portrait_response,
        ]
    )

    result = verify(session)  # type: ignore[arg-type]

    assert result.champion_version == "16.13.1"
    assert result.champion_count == 100
    assert result.portrait_filename == "Champion0.png"
    assert [url for url, _kwargs in session.calls] == [
        REALM_URL,
        "https://cdn.test/cdn/16.13.1/data/en_US/champion.json",
        "https://cdn.test/cdn/16.13.1/img/champion/Champion0.png",
    ]
    assert session.calls[-1][1]["stream"] is True
    assert portrait_response.closed is True
    assert session.closed is False


def test_read_bounded_png_rejects_invalid_content_length() -> None:
    response = _Response(
        content=_ONE_PIXEL_PNG,
        headers={"Content-Type": "image/png", "Content-Length": "many"},
    )

    with pytest.raises(ValueError, match="invalid Content-Length"):
        read_bounded_png(response)


def test_json_decode_failures_are_reported_as_contract_errors() -> None:
    response = _Response(payload=None)

    def invalid_json() -> object:
        raise requests.exceptions.JSONDecodeError("bad", "{", 0)

    response.json = invalid_json  # type: ignore[method-assign]
    session = _Session([response])

    with pytest.raises(ValueError, match="VN realm did not contain valid JSON"):
        verify(session)  # type: ignore[arg-type]
