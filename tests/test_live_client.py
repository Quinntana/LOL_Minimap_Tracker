from __future__ import annotations

import logging
from typing import Any

import pytest
import requests

from lol_minimap_tracker.domain.models import Role, RosterStatus, SummonerSpellRef
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


def test_players_without_a_nonblank_team_are_not_treated_as_opponents() -> None:
    players = [
        {"riotId": "Me#TAG", "team": "ORDER", "championName": "Lux"},
        {"riotId": "Unknown#1", "championName": "Aatrox", "position": "TOP"},
        {
            "riotId": "Unknown#2",
            "team": "   ",
            "championName": "Nami",
            "position": "SUPPORT",
        },
        {
            "riotId": "Enemy#1",
            "team": "CHAOS",
            "championName": "Ahri",
            "position": "MIDDLE",
        },
    ]
    client = LiveClientClient(1, logging.getLogger("test"), Session(["Me#TAG", players]))  # type: ignore[arg-type]

    result = client.poll()

    assert result.status is RosterStatus.ACTIVE
    assert [member.champion_name for member in result.members] == ["Ahri"]


def test_full_live_schema_parses_privacy_safe_cooldown_fields() -> None:
    players = [
        {"riotId": "Me#TAG", "team": "ORDER", "championName": "Lux"},
        {
            "riotId": "Secret Enemy#NA1",
            "summonerName": "Also Secret",
            "team": "CHAOS",
            "championName": "Bel'Veth",
            "rawChampionName": "game_character_displayname_GameCharacter_Belveth",
            "position": "JUNGLE",
            "level": 11.0,
            "summonerSpells": {
                "summonerSpellOne": {
                    "displayName": "Flash",
                    "rawDisplayName": ("GeneratedTip_SummonerSpell_SummonerFlash_DisplayName"),
                },
                "summonerSpellTwo": {
                    "displayName": "Smite",
                    "rawDisplayName": ("GeneratedTip_SummonerSpell_SummonerSmite_DisplayName"),
                },
                "unrelatedFutureSpell": {
                    "displayName": "Do not parse me",
                    "rawDisplayName": (
                        "GeneratedTip_SummonerSpell_UnrelatedFutureSpell_DisplayName"
                    ),
                },
            },
        },
    ]
    client = LiveClientClient(
        1,
        logging.getLogger("test"),
        Session(["Me#TAG", players]),  # type: ignore[arg-type]
    )

    member = client.poll().members[0]

    assert member.champion_name == "Bel'Veth"
    assert member.champion_id == "Belveth"
    assert member.level == 11
    assert member.summoner_spells == (
        SummonerSpellRef("SummonerFlash", "Flash"),
        SummonerSpellRef("SummonerSmite", "Smite"),
    )
    assert member.participant_id.startswith("participant-")
    assert "Secret Enemy" not in member.participant_id
    assert "Secret Enemy" not in repr(member)
    assert "Also Secret" not in repr(member)


@pytest.mark.parametrize(
    ("raw_level", "expected"),
    [
        (1, 1),
        (18, 18),
        (7.0, 7),
        (0, None),
        (19, None),
        (6.5, None),
        (True, None),
        ("6", None),
        (float("nan"), None),
    ],
)
def test_level_accepts_only_whole_json_numbers_in_game_range(
    raw_level: object, expected: int | None
) -> None:
    players = [
        {"riotId": "Me#TAG", "team": "ORDER", "championName": "Lux"},
        {
            "riotId": "Enemy#1",
            "team": "CHAOS",
            "championName": "Aatrox",
            "level": raw_level,
        },
    ]
    result = LiveClientClient(
        1,
        logging.getLogger("test"),
        Session(["Me#TAG", players]),  # type: ignore[arg-type]
    ).poll()

    assert result.members[0].level == expected


def test_malformed_optional_cooldown_fields_are_omitted_safely() -> None:
    players = [
        {"riotId": "Me#TAG", "team": "ORDER", "championName": "Lux"},
        {
            "team": "CHAOS",
            "championName": " Aatrox ",
            "rawChampionName": {"unexpected": "shape"},
            "summonerSpells": {
                "summonerSpellOne": "not an object",
                "summonerSpellTwo": {
                    "displayName": "Barrier",
                    "rawDisplayName": "schema_changed_but_display_name_is_useful",
                },
            },
        },
    ]
    result = LiveClientClient(
        1,
        logging.getLogger("test"),
        Session(["Me#TAG", players]),  # type: ignore[arg-type]
    ).poll()

    member = result.members[0]
    assert member.champion_name == "Aatrox"
    assert member.champion_id is None
    assert member.level is None
    assert member.summoner_spells == (
        SummonerSpellRef(None, "Unknown spell"),
        SummonerSpellRef(None, "Barrier"),
    )


def test_participant_hash_and_fallback_are_deterministic_across_polls() -> None:
    identified_enemy = {
        "riotId": "Enemy#1",
        "team": "CHAOS",
        "championName": "Aatrox",
    }
    anonymous_enemy = {
        "team": "CHAOS",
        "championName": "Nami",
        "rawChampionName": "game_character_displayname_GameCharacter_Nami",
    }

    def poll(enemy: dict[str, object]) -> str:
        players = [
            {"riotId": "Me#TAG", "team": "ORDER", "championName": "Lux"},
            enemy,
        ]
        result = LiveClientClient(
            1,
            logging.getLogger("test"),
            Session(["Me#TAG", players]),  # type: ignore[arg-type]
        ).poll()
        return result.members[0].participant_id

    assert poll(identified_enemy) == poll(dict(identified_enemy))
    assert poll(anonymous_enemy) == poll(dict(anonymous_enemy))
    assert poll(identified_enemy) != poll(anonymous_enemy)


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
