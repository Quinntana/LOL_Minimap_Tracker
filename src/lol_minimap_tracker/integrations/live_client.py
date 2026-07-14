"""Riot Live Client Data API adapter."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping
from typing import Any

import requests
import urllib3

from ..domain.models import (
    Role,
    RosterMember,
    RosterResult,
    RosterStatus,
    SummonerSpellRef,
)

_RAW_CHAMPION_ID = re.compile(r"(?:^|_)(?P<identifier>[A-Za-z][A-Za-z0-9]*)$")
_RAW_SPELL_ID = re.compile(
    r"^GeneratedTip_SummonerSpell_(?P<identifier>[A-Za-z0-9_]+)_DisplayName$"
)
_SPELL_KEYS = ("summonerSpellOne", "summonerSpellTwo")


def _clean_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _parse_champion_id(value: object) -> str | None:
    raw_name = _clean_string(value)
    if raw_name is None:
        return None
    match = _RAW_CHAMPION_ID.search(raw_name)
    return match.group("identifier") if match is not None else None


def _parse_level(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        level = value
    elif isinstance(value, float) and value.is_integer():
        level = int(value)
    else:
        return None
    return level if 1 <= level <= 18 else None


def _parse_summoner_spells(value: object) -> tuple[SummonerSpellRef, ...]:
    if not isinstance(value, Mapping):
        return ()
    spells: list[SummonerSpellRef] = []
    for key in _SPELL_KEYS:
        raw_spell = value.get(key)
        if not isinstance(raw_spell, Mapping):
            spells.append(SummonerSpellRef(None, "Unknown spell"))
            continue
        raw_display_name = _clean_string(raw_spell.get("rawDisplayName"))
        match = _RAW_SPELL_ID.fullmatch(raw_display_name or "")
        identifier = match.group("identifier") if match is not None else None
        display_name = _clean_string(raw_spell.get("displayName"))
        if display_name is None and identifier is None:
            spells.append(SummonerSpellRef(None, "Unknown spell"))
            continue
        spells.append(
            SummonerSpellRef(
                identifier=identifier,
                display_name=display_name or identifier or "Unknown spell",
            )
        )
    return tuple(spells)


def _participant_id(
    player: Mapping[str, object], champion_name: str, champion_id: str | None
) -> str:
    raw_identity = _clean_string(player.get("riotId")) or _clean_string(player.get("summonerName"))
    if raw_identity is not None:
        material = f"identity:{raw_identity.casefold()}"
    else:
        team = _clean_string(player.get("team")) or "unknown-team"
        champion = champion_id or champion_name
        material = f"fallback:{team.casefold()}:{champion.casefold()}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"participant-{digest}"


class LiveClientClient:
    BASE_URL = "https://127.0.0.1:2999/liveclientdata"

    def __init__(
        self,
        timeout: float,
        logger: logging.Logger,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self.logger = logger
        self.session = session or requests.Session()
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _get_json(self, endpoint: str) -> Any:
        response = self.session.get(
            f"{self.BASE_URL}/{endpoint}",
            timeout=self.timeout,
            verify=False,
        )
        response.raise_for_status()
        return response.json()

    def poll(self) -> RosterResult:
        try:
            active_name = self._get_json("activeplayername")
            players = self._get_json("playerlist")
        except (requests.RequestException, ValueError) as exc:
            return RosterResult(RosterStatus.UNAVAILABLE, error=str(exc))

        if not isinstance(active_name, str) or not isinstance(players, list):
            return RosterResult(
                RosterStatus.INVALID_RESPONSE,
                error="Live Client returned an unexpected response shape",
            )

        active_player = next(
            (
                player
                for player in players
                if isinstance(player, dict)
                and (
                    active_name == player.get("riotId") or active_name == player.get("summonerName")
                )
            ),
            None,
        )
        if not isinstance(active_player, dict):
            return RosterResult(
                RosterStatus.INVALID_RESPONSE,
                error="Active player was not present in playerlist",
            )

        my_team = _clean_string(active_player.get("team"))
        if my_team is None:
            return RosterResult(
                RosterStatus.INVALID_RESPONSE,
                error="Active player did not have a valid team",
            )
        members: list[RosterMember] = []
        for player in players:
            if not isinstance(player, dict):
                continue
            player_team = _clean_string(player.get("team"))
            if player_team is None or player_team == my_team:
                continue
            champion_name = _clean_string(player.get("championName"))
            if champion_name is not None:
                champion_id = _parse_champion_id(player.get("rawChampionName"))
                members.append(
                    RosterMember(
                        champion_name=champion_name,
                        role=Role.from_api(player.get("position")),
                        participant_id=_participant_id(player, champion_name, champion_id),
                        champion_id=champion_id,
                        level=_parse_level(player.get("level")),
                        summoner_spells=_parse_summoner_spells(player.get("summonerSpells")),
                    )
                )
        if not members:
            return RosterResult(
                RosterStatus.INVALID_RESPONSE,
                error="No opponents were present in playerlist",
            )
        return RosterResult(RosterStatus.ACTIVE, tuple(members))
