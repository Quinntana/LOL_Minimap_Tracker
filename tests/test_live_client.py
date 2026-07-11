from __future__ import annotations

import logging
from typing import Any

import requests

from lol_minimap_tracker.domain.models import Role, RosterStatus
from lol_minimap_tracker.integrations.live_client import LiveClientClient


class Response:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses

    def get(self, *_args: Any, **_kwargs: Any) -> Response:
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return Response(response)


def test_active_roster_uses_riot_id_team_and_roles() -> None:
    players = [
        {"riotId": "Me#TAG", "team": "ORDER", "championName": "Lux"},
        {
            "riotId": "Enemy#1",
            "team": "CHAOS",
            "championName": "Aatrox",
            "position": "TOP",
        },
        {
            "riotId": "Enemy#2",
            "team": "CHAOS",
            "championName": "Nami",
            "position": "SUPPORT",
        },
    ]
    client = LiveClientClient(1, logging.getLogger("test"), Session(["Me#TAG", players]))  # type: ignore[arg-type]
    result = client.poll()
    assert result.status is RosterStatus.ACTIVE
    assert [(member.champion_name, member.role) for member in result.members] == [
        ("Aatrox", Role.TOP),
        ("Nami", Role.UTILITY),
    ]


def test_unavailable_and_invalid_responses_are_distinct() -> None:
    unavailable = LiveClientClient(
        1,
        logging.getLogger("test"),
        Session([requests.ConnectionError("no game")]),  # type: ignore[arg-type]
    ).poll()
    invalid = LiveClientClient(
        1,
        logging.getLogger("test"),
        Session(["Me#TAG", {"not": "a list"}]),  # type: ignore[arg-type]
    ).poll()
    assert unavailable.status is RosterStatus.UNAVAILABLE
    assert invalid.status is RosterStatus.INVALID_RESPONSE


def test_summoner_name_fallback_and_unknown_role() -> None:
    players = [
        {"summonerName": "Legacy Name", "team": "ORDER", "championName": "Lux"},
        {
            "riotId": "Enemy#1",
            "team": "CHAOS",
            "championName": "Shaco",
            "position": "ROAMER",
        },
    ]
    client = LiveClientClient(
        1,
        logging.getLogger("test"),
        Session(["Legacy Name", players]),  # type: ignore[arg-type]
    )
    result = client.poll()
    assert result.status is RosterStatus.ACTIVE
    assert result.members[0].role is Role.UNKNOWN


def test_missing_active_player_and_empty_enemy_team_are_invalid() -> None:
    missing = LiveClientClient(
        1,
        logging.getLogger("test"),
        Session(["Absent#TAG", [{"riotId": "Someone#1", "team": "ORDER"}]]),  # type: ignore[arg-type]
    ).poll()
    empty = LiveClientClient(
        1,
        logging.getLogger("test"),
        Session(
            [
                "Me#TAG",
                [{"riotId": "Me#TAG", "team": "ORDER", "championName": "Lux"}],
            ]
        ),  # type: ignore[arg-type]
    ).poll()
    assert missing.status is RosterStatus.INVALID_RESPONSE
    assert empty.status is RosterStatus.INVALID_RESPONSE


def test_json_decode_failure_is_unavailable() -> None:
    client = LiveClientClient(
        1,
        logging.getLogger("test"),
        Session([ValueError("bad json")]),  # type: ignore[arg-type]
    )
    assert client.poll().status is RosterStatus.UNAVAILABLE
