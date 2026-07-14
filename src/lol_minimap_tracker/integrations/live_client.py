"""Riot Live Client Data API adapter."""

from __future__ import annotations

import hashlib
import logging
import math
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
_MAX_LOCAL_TIMEOUT_SECONDS = 5.0
_MAX_PLAYER_RECORDS = 64
_MAX_TEXT_LENGTH = 256
_OPPOSING_TEAM = {"ORDER": "CHAOS", "CHAOS": "ORDER"}


def _clean_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned and len(cleaned) <= _MAX_TEXT_LENGTH else None


def _team(value: object) -> str | None:
    team = _clean_string(value)
    if team is None:
        return None
    normalized = team.upper()
    return normalized if normalized in _OPPOSING_TEAM else None


def _identity_tokens(value: object) -> frozenset[str]:
    """Return only documented, exact player identifiers in normalized form."""

    if isinstance(value, str):
        cleaned = _clean_string(value)
        return frozenset((cleaned.casefold(),)) if cleaned is not None else frozenset()
    if not isinstance(value, Mapping):
        return frozenset()

    identities: set[str] = set()
    for field in ("riotId", "summonerName"):
        identity = _clean_string(value.get(field))
        if identity is not None:
            identities.add(identity.casefold())

    game_name = _clean_string(value.get("riotIdGameName"))
    tag_line = _clean_string(value.get("riotIdTagLine"))
    if game_name is not None and tag_line is not None:
        identities.add(f"{game_name}#{tag_line}".casefold())
    return frozenset(identities)


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
    raw_identity = _clean_string(player.get("riotId"))
    if raw_identity is None:
        game_name = _clean_string(player.get("riotIdGameName"))
        tag_line = _clean_string(player.get("riotIdTagLine"))
        if game_name is not None and tag_line is not None:
            raw_identity = f"{game_name}#{tag_line}"
    raw_identity = raw_identity or _clean_string(player.get("summonerName"))
    if raw_identity is not None:
        material = f"identity:{raw_identity.casefold()}"
    else:
        team = _team(player.get("team")) or "unknown-team"
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
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        self.timeout = min(timeout, _MAX_LOCAL_TIMEOUT_SECONDS)
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

    @staticmethod
    def _invalid(error: str) -> RosterResult:
        return RosterResult(RosterStatus.INVALID_RESPONSE, error=error)

    @staticmethod
    def _unavailable(exc: Exception) -> RosterResult:
        # Do not put response bodies or player identities into runtime health/log output.
        return RosterResult(
            RosterStatus.UNAVAILABLE,
            error=f"Live Client request failed ({type(exc).__name__})",
        )

    def _parse_roster(self, active_identity: object, players: object) -> RosterResult:
        active_tokens = _identity_tokens(active_identity)
        if not active_tokens:
            return RosterResult(
                RosterStatus.INVALID_RESPONSE,
                error="Live Client did not return a valid active-player identity",
            )
        if not isinstance(players, list) or len(players) > _MAX_PLAYER_RECORDS:
            return self._invalid("Live Client returned an invalid player list")

        active_players = [
            player
            for player in players
            if isinstance(player, Mapping) and active_tokens.intersection(_identity_tokens(player))
        ]
        if len(active_players) != 1:
            return self._invalid("Active player was not uniquely present in player list")

        my_team = _team(active_players[0].get("team"))
        if my_team is None:
            return self._invalid("Active player did not have a recognized team")
        opponent_team = _OPPOSING_TEAM[my_team]

        members: list[RosterMember] = []
        for player in players:
            if not isinstance(player, Mapping):
                continue
            if _team(player.get("team")) != opponent_team:
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
            return self._invalid("No valid opponents were present in player list")
        return RosterResult(RosterStatus.ACTIVE, tuple(members))

    def _parse_all_game_data(
        self,
        payload: object,
        expected_active_tokens: frozenset[str],
    ) -> RosterResult:
        if not isinstance(payload, Mapping):
            return self._invalid("Live Client returned invalid aggregate game data")
        active_player = payload.get("activePlayer")
        if expected_active_tokens and not expected_active_tokens.intersection(
            _identity_tokens(active_player)
        ):
            return self._invalid("Aggregate game data was for a different active player")
        return self._parse_roster(active_player, payload.get("allPlayers"))

    def poll(self) -> RosterResult:
        """Read a roster with one bounded aggregate fallback for endpoint drift/outages."""

        primary_result: RosterResult | None = None
        expected_active_tokens: frozenset[str] = frozenset()
        try:
            active_name = self._get_json("activeplayername")
            expected_active_tokens = _identity_tokens(active_name)
            players = self._get_json("playerlist")
        except (requests.RequestException, ValueError):
            pass
        else:
            primary_result = self._parse_roster(active_name, players)
            if primary_result.status is RosterStatus.ACTIVE:
                return primary_result

        # Riot documents /allgamedata as the aggregate of the subset endpoints. A
        # single fallback keeps polling bounded to at most three local requests.
        try:
            aggregate = self._get_json("allgamedata")
        except (requests.RequestException, ValueError) as exc:
            return primary_result or self._unavailable(exc)

        return self._parse_all_game_data(aggregate, expected_active_tokens)
