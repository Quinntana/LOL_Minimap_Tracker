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

    def get(self, *_args: Any, **_kwargs: Any) -> Response:
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def metadata() -> dict[str, Any]:
    return {
        "data": {
            "MonkeyKing": {
                "name": "Wukong",
                "image": {"full": "MonkeyKing.png"},
            }
        }
    }


def png() -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((10, 10, 3), 120, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_metadata_filename_and_portrait_are_cached(tmp_path: Path) -> None:
    session = Session([Response(["16.13.1"]), Response(metadata()), Response(content=png())])
    client = DataDragonClient(tmp_path, logging.getLogger("test"), session)  # type: ignore[arg-type]
    portraits = client.get_portraits((RosterMember("Wukong", Role.TOP),))
    assert portraits["Wukong"].shape == (10, 10, 3)
    assert (tmp_path / "ddragon" / "16.13.1" / "portraits" / "MonkeyKing.png").exists()


def test_offline_metadata_falls_back_to_cache(tmp_path: Path) -> None:
    cache = tmp_path / "ddragon"
    (cache / "16.13.1").mkdir(parents=True)
    (cache / "version.txt").write_text("16.13.1", encoding="utf-8")
    (cache / "16.13.1" / "champion.json").write_text(json.dumps(metadata()), encoding="utf-8")
    session = Session([requests.ConnectionError("offline"), requests.ConnectionError("offline")])
    client = DataDragonClient(tmp_path, logging.getLogger("test"), session)  # type: ignore[arg-type]
    assert client.get_metadata()["wukong"] == "MonkeyKing.png"


def test_malformed_version_without_cache_returns_no_metadata(tmp_path: Path) -> None:
    client = DataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        Session([Response({"not": "versions"}), Response({"still": "not versions"})]),  # type: ignore[arg-type]
    )
    assert client.get_latest_version() is None
    assert client.get_metadata() == {}


def test_corrupt_remote_and_cached_metadata_returns_empty(tmp_path: Path) -> None:
    cache = tmp_path / "ddragon"
    (cache / "16.13.1").mkdir(parents=True)
    (cache / "version.txt").write_text("16.13.1", encoding="utf-8")
    (cache / "16.13.1" / "champion.json").write_text("not-json", encoding="utf-8")
    session = Session([requests.ConnectionError("offline"), Response({"data": {}})])
    client = DataDragonClient(tmp_path, logging.getLogger("test"), session)  # type: ignore[arg-type]
    assert client.get_metadata() == {}


def test_cached_portrait_avoids_portrait_network_request(tmp_path: Path) -> None:
    cache = tmp_path / "ddragon"
    portrait_dir = cache / "16.13.1" / "portraits"
    portrait_dir.mkdir(parents=True)
    (cache / "version.txt").write_text("16.13.1", encoding="utf-8")
    (cache / "16.13.1" / "champion.json").write_text(json.dumps(metadata()), encoding="utf-8")
    (portrait_dir / "MonkeyKing.png").write_bytes(png())
    session = Session([requests.ConnectionError("offline"), requests.ConnectionError("offline")])
    client = DataDragonClient(tmp_path, logging.getLogger("test"), session)  # type: ignore[arg-type]
    portraits = client.get_portraits((RosterMember("Wukong", Role.TOP),))
    assert portraits["Wukong"].shape == (10, 10, 3)
    assert session.responses == []


def test_unknown_champion_and_invalid_image_are_skipped(tmp_path: Path) -> None:
    unknown_session = Session([Response(["16.13.1"]), Response(metadata())])
    unknown = DataDragonClient(
        tmp_path / "unknown",
        logging.getLogger("test"),
        unknown_session,  # type: ignore[arg-type]
    )
    assert unknown.get_portraits((RosterMember("NotAChampion", Role.TOP),)) == {}

    invalid_session = Session(
        [Response(["16.13.1"]), Response(metadata()), Response(content=b"bad-png")]
    )
    invalid = DataDragonClient(
        tmp_path / "invalid",
        logging.getLogger("test"),
        invalid_session,  # type: ignore[arg-type]
    )
    assert invalid.get_portraits((RosterMember("Wukong", Role.TOP),)) == {}
