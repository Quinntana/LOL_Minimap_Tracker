from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from lol_minimap_tracker.domain.models import Role, RosterMember
from lol_minimap_tracker.integrations.data_dragon import DataDragonClient

CDN = "https://ddragon.leagueoflegends.com/cdn"
REALM_URL = "https://ddragon.leagueoflegends.com/realms/vn.json"


class Response:
    def __init__(self, payload: Any = None, content: bytes = b"") -> None:
        self.payload = payload
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


class Session:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url: str, **_kwargs: Any) -> Response:
        self.calls.append(url)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now


def realm(version: str = "16.13.1", cdn: str = CDN) -> dict[str, Any]:
    return {"cdn": cdn, "n": {"champion": version}}


def write_realm(cache: Path, version: str = "16.13.1", cdn: str = CDN) -> None:
    path = cache / "realms" / "vn.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(realm(version, cdn)), encoding="utf-8")


def metadata(
    filename: str = "MonkeyKing.png",
    *,
    key: str = "MonkeyKing",
    name: str = "Wukong",
    version: str = "16.13.1",
    count: int = DataDragonClient.MIN_CHAMPION_RECORDS,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        key: {
            "id": key,
            "name": name,
            "image": {"full": filename},
        }
    }
    for index in range(count - 1):
        identifier = f"FixtureChampion{index}"
        data[identifier] = {
            "id": identifier,
            "name": f"Fixture Champion {index}",
            "image": {"full": f"{identifier}.png"},
        }
    return {
        "type": "champion",
        "format": "standAloneComplex",
        "version": version,
        "data": data,
    }


def png() -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((10, 10, 3), 120, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_metadata_filename_and_portrait_are_cached(tmp_path: Path) -> None:
    regional_cdn = "https://regional-cdn.example.test/cdn"
    session = Session(
        [Response(realm(cdn=regional_cdn)), Response(metadata()), Response(content=png())]
    )
    client = DataDragonClient(tmp_path, logging.getLogger("test"), session)  # type: ignore[arg-type]
    portraits = client.get_portraits((RosterMember("Wukong", Role.TOP),))
    assert portraits["Wukong"].shape == (10, 10, 3)
    assert (tmp_path / "ddragon" / "16.13.1" / "portraits" / "MonkeyKing.png").exists()
    assert session.calls == [
        REALM_URL,
        f"{regional_cdn}/16.13.1/data/en_US/champion.json",
        f"{regional_cdn}/16.13.1/img/champion/MonkeyKing.png",
    ]


def test_offline_metadata_falls_back_to_cache(tmp_path: Path) -> None:
    cache = tmp_path / "ddragon"
    (cache / "16.13.1").mkdir(parents=True)
    write_realm(cache)
    (cache / "16.13.1" / "champion.json").write_text(json.dumps(metadata()), encoding="utf-8")
    session = Session([requests.ConnectionError("offline")])
    client = DataDragonClient(tmp_path, logging.getLogger("test"), session)  # type: ignore[arg-type]
    assert client.get_metadata()["wukong"] == "MonkeyKing.png"


def test_legacy_version_cache_remains_usable_offline(tmp_path: Path) -> None:
    cache = tmp_path / "ddragon"
    (cache / "16.13.1").mkdir(parents=True)
    (cache / "version.txt").write_text("16.13.1", encoding="utf-8")
    (cache / "16.13.1" / "champion.json").write_text(json.dumps(metadata()), encoding="utf-8")
    client = DataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        Session([requests.ConnectionError("offline")]),  # type: ignore[arg-type]
    )

    assert client.get_metadata()["wukong"] == "MonkeyKing.png"


def test_malformed_realm_without_cache_returns_no_metadata(tmp_path: Path) -> None:
    client = DataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        Session([Response({"not": "a realm"})]),  # type: ignore[arg-type]
    )
    assert client.get_latest_version() is None
    assert client.get_metadata() == {}


def test_realm_default_version_fallback_survives_missing_component_key(tmp_path: Path) -> None:
    payload = {"cdn": CDN, "dd": "16.13.1", "n": {}}
    client = DataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        Session([Response(payload), Response(metadata())]),  # type: ignore[arg-type]
    )

    assert client.get_metadata()["wukong"] == "MonkeyKing.png"


def test_corrupt_remote_and_cached_metadata_returns_empty(tmp_path: Path) -> None:
    cache = tmp_path / "ddragon"
    (cache / "16.13.1").mkdir(parents=True)
    write_realm(cache)
    (cache / "16.13.1" / "champion.json").write_text("not-json", encoding="utf-8")
    session = Session([requests.ConnectionError("offline"), Response({"data": {}})])
    client = DataDragonClient(tmp_path, logging.getLogger("test"), session)  # type: ignore[arg-type]
    assert client.get_metadata() == {}


def test_cached_portrait_avoids_portrait_network_request(tmp_path: Path) -> None:
    cache = tmp_path / "ddragon"
    portrait_dir = cache / "16.13.1" / "portraits"
    portrait_dir.mkdir(parents=True)
    write_realm(cache)
    (cache / "16.13.1" / "champion.json").write_text(json.dumps(metadata()), encoding="utf-8")
    (portrait_dir / "MonkeyKing.png").write_bytes(png())
    session = Session([requests.ConnectionError("offline")])
    client = DataDragonClient(tmp_path, logging.getLogger("test"), session)  # type: ignore[arg-type]
    portraits = client.get_portraits((RosterMember("Wukong", Role.TOP),))
    assert portraits["Wukong"].shape == (10, 10, 3)
    assert session.responses == []


def test_transient_metadata_failure_retries_without_restart(tmp_path: Path) -> None:
    clock = Clock()
    session = Session(
        [
            Response(realm()),
            requests.ConnectionError("temporary"),
            Response(metadata()),
        ]
    )
    client = DataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,  # type: ignore[arg-type]
        metadata_retry_delay=5,
        monotonic=clock.monotonic,
    )

    assert client.get_metadata() == {}
    assert client.get_metadata() == {}
    assert len(session.calls) == 2
    clock.now += 5
    assert client.get_metadata()["wukong"] == "MonkeyKing.png"
    assert session.responses == []


def test_transient_realm_failure_uses_short_backoff_then_recovers(tmp_path: Path) -> None:
    clock = Clock()
    session = Session(
        [
            requests.ConnectionError("temporary"),
            Response(realm()),
            Response(metadata()),
        ]
    )
    client = DataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,  # type: ignore[arg-type]
        realm_retry_delay=5,
        monotonic=clock.monotonic,
    )

    assert client.get_metadata() == {}
    assert client.get_metadata() == {}
    assert session.calls == [REALM_URL]
    clock.now += 5
    assert client.get_metadata()["wukong"] == "MonkeyKing.png"
    assert session.responses == []


def test_realm_is_refreshed_only_after_injected_ttl(tmp_path: Path) -> None:
    clock = Clock()
    session = Session(
        [
            Response(realm("16.13.1")),
            Response(metadata()),
            Response(realm("16.13.1")),
        ]
    )
    client = DataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,  # type: ignore[arg-type]
        realm_ttl=10,
        metadata_retry_delay=5,
        monotonic=clock.monotonic,
    )

    assert client.get_metadata()["wukong"] == "MonkeyKing.png"
    assert len(session.calls) == 2
    clock.now += 9
    assert client.get_metadata()["wukong"] == "MonkeyKing.png"
    assert len(session.calls) == 2
    clock.now += 2
    assert client.get_metadata()["wukong"] == "MonkeyKing.png"
    assert session.calls == [
        REALM_URL,
        f"{CDN}/16.13.1/data/en_US/champion.json",
        REALM_URL,
    ]


def test_patch_switch_waits_for_valid_catalog_and_keeps_last_known_good(
    tmp_path: Path,
) -> None:
    clock = Clock()
    old_metadata = metadata("OldWukong.png")
    new_metadata = metadata("NewWukong.png", version="16.14.1")
    session = Session(
        [
            Response(realm("16.13.1")),
            Response(old_metadata),
            Response(realm("16.14.1")),
            Response(
                {
                    "type": "champion",
                    "format": "standAloneComplex",
                    "version": "16.14.1",
                    "data": {
                        "MonkeyKing": {
                            "id": "MonkeyKing",
                            "name": "Wukong",
                            "image": {"full": "PartialWukong.png"},
                        }
                    },
                }
            ),
            Response(new_metadata),
        ]
    )
    client = DataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,  # type: ignore[arg-type]
        realm_ttl=10,
        metadata_retry_delay=5,
        monotonic=clock.monotonic,
    )

    assert client.get_metadata()["wukong"] == "OldWukong.png"
    assert client.get_latest_version() == "16.13.1"
    clock.now += 11

    # The realm announces a new patch, but its first catalog is incomplete.
    # The in-memory pair and both authoritative cache markers remain old.
    assert client.get_metadata()["wukong"] == "OldWukong.png"
    assert client.get_latest_version() == "16.13.1"
    cached_realm = json.loads(client.realm_file.read_text(encoding="utf-8"))
    assert cached_realm["n"]["champion"] == "16.13.1"
    assert client.version_file.read_text(encoding="utf-8") == "16.13.1"
    assert not (tmp_path / "ddragon" / "16.14.1" / "champion.json").exists()

    # Regular roster refreshes must not hammer the missing new-patch catalog.
    call_count = len(session.calls)
    assert client.get_metadata()["wukong"] == "OldWukong.png"
    assert len(session.calls) == call_count

    # A later call retries the pending catalog without restarting.  Only after
    # it validates does the version/catalog pair switch and become durable.
    clock.now += 5
    assert client.get_metadata()["wukong"] == "NewWukong.png"
    assert client.get_latest_version() == "16.14.1"
    cached_realm = json.loads(client.realm_file.read_text(encoding="utf-8"))
    assert cached_realm["n"]["champion"] == "16.14.1"
    assert (tmp_path / "ddragon" / "16.14.1" / "champion.json").exists()


def test_mismatched_catalog_version_never_replaces_last_known_good(tmp_path: Path) -> None:
    clock = Clock()
    session = Session(
        [
            Response(realm("16.13.1")),
            Response(metadata("Old.png")),
            Response(realm("16.14.1")),
            Response(metadata("WrongPatch.png", version="16.13.1")),
        ]
    )
    client = DataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,  # type: ignore[arg-type]
        realm_ttl=5,
        monotonic=clock.monotonic,
    )

    assert client.get_metadata()["wukong"] == "Old.png"
    clock.now += 5
    assert client.get_metadata()["wukong"] == "Old.png"
    assert client.get_latest_version() == "16.13.1"
    assert not (tmp_path / "ddragon" / "16.14.1" / "champion.json").exists()


def test_candidate_with_severe_record_shrink_keeps_larger_catalog(tmp_path: Path) -> None:
    clock = Clock()
    session = Session(
        [
            Response(realm("16.13.1")),
            Response(metadata("Old.png", count=150)),
            Response(realm("16.14.1")),
            Response(metadata("Shrunk.png", version="16.14.1", count=100)),
            Response(metadata("Recovered.png", version="16.14.1", count=150)),
        ]
    )
    client = DataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,  # type: ignore[arg-type]
        realm_ttl=5,
        metadata_retry_delay=3,
        monotonic=clock.monotonic,
    )

    assert client.get_metadata()["wukong"] == "Old.png"
    clock.now += 5
    assert client.get_metadata()["wukong"] == "Old.png"
    assert client.get_latest_version() == "16.13.1"
    assert not (tmp_path / "ddragon" / "16.14.1" / "champion.json").exists()

    clock.now += 3
    assert client.get_metadata()["wukong"] == "Recovered.png"
    assert client.get_latest_version() == "16.14.1"


def test_mostly_malformed_catalog_is_rejected(tmp_path: Path) -> None:
    payload = metadata(count=120)
    data = payload["data"]
    assert isinstance(data, dict)
    for index, key in enumerate(tuple(data)):
        if index >= 13:
            break
        data[key] = {"id": key, "image": {"full": "../unsafe.png"}}
    client = DataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        Session([Response(realm()), Response(payload)]),  # type: ignore[arg-type]
    )

    assert client.get_metadata() == {}


def test_cache_write_failures_do_not_discard_valid_network_data(
    tmp_path: Path, monkeypatch: Any
) -> None:
    session = Session([Response(realm()), Response(metadata()), Response(content=png())])
    client = DataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,  # type: ignore[arg-type]
    )

    def fail_write(_path: Path, _content: bytes) -> None:
        raise OSError("read-only cache")

    monkeypatch.setattr(client, "_atomic_write", fail_write)

    portraits = client.get_portraits((RosterMember("Wukong", Role.TOP),))
    assert portraits["Wukong"].shape == (10, 10, 3)
    assert client.get_latest_version() == "16.13.1"


def test_champion_id_survives_localized_display_name(tmp_path: Path) -> None:
    session = Session([Response(realm()), Response(metadata()), Response(content=png())])
    client = DataDragonClient(tmp_path, logging.getLogger("test"), session)  # type: ignore[arg-type]

    portraits = client.get_portraits(
        (RosterMember("Localized Wukong", Role.TOP, champion_id="MonkeyKing"),)
    )

    assert portraits["Localized Wukong"].shape == (10, 10, 3)


def test_untrusted_version_and_image_paths_are_rejected(tmp_path: Path) -> None:
    invalid_version = DataDragonClient(
        tmp_path / "version",
        logging.getLogger("test"),
        Session([Response(realm("../escape"))]),  # type: ignore[arg-type]
    )
    assert invalid_version.get_latest_version() is None

    invalid_cdn = DataDragonClient(
        tmp_path / "cdn",
        logging.getLogger("test"),
        Session([Response(realm(cdn="file:///tmp/cdn"))]),  # type: ignore[arg-type]
    )
    assert invalid_cdn.get_latest_version() is None

    unsafe_metadata = {
        "data": {
            "Aatrox": {
                "id": "Aatrox",
                "name": "Aatrox",
                "image": {"full": "../Aatrox.png"},
            }
        }
    }
    unsafe_image = DataDragonClient(
        tmp_path / "image",
        logging.getLogger("test"),
        Session([Response(realm()), Response(unsafe_metadata)]),  # type: ignore[arg-type]
    )
    assert unsafe_image.get_metadata() == {}
    assert not (tmp_path / "image" / "Aatrox.png").exists()


def test_refresh_intervals_must_be_positive_and_finite(tmp_path: Path) -> None:
    for value in (0, -1, float("inf"), float("nan"), True):
        try:
            DataDragonClient(
                tmp_path,
                logging.getLogger("test"),
                Session([]),  # type: ignore[arg-type]
                realm_ttl=value,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"realm_ttl={value!r} should be rejected")

        try:
            DataDragonClient(
                tmp_path,
                logging.getLogger("test"),
                Session([]),  # type: ignore[arg-type]
                realm_retry_delay=value,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"realm_retry_delay={value!r} should be rejected")

        try:
            DataDragonClient(
                tmp_path,
                logging.getLogger("test"),
                Session([]),  # type: ignore[arg-type]
                metadata_retry_delay=value,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"metadata_retry_delay={value!r} should be rejected")


def test_unknown_champion_and_invalid_image_are_skipped(tmp_path: Path) -> None:
    unknown_session = Session([Response(realm()), Response(metadata())])
    unknown = DataDragonClient(
        tmp_path / "unknown",
        logging.getLogger("test"),
        unknown_session,  # type: ignore[arg-type]
    )
    assert unknown.get_portraits((RosterMember("NotAChampion", Role.TOP),)) == {}

    invalid_session = Session(
        [Response(realm()), Response(metadata()), Response(content=b"bad-png")]
    )
    invalid = DataDragonClient(
        tmp_path / "invalid",
        logging.getLogger("test"),
        invalid_session,  # type: ignore[arg-type]
    )
    assert invalid.get_portraits((RosterMember("Wukong", Role.TOP),)) == {}
