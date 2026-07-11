"""Riot Live Client Data API adapter."""

from __future__ import annotations

import logging
from typing import Any

import requests
import urllib3

from ..domain.models import Role, RosterMember, RosterResult, RosterStatus


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
                and active_name
                in {
                    player.get("riotId"),
                    player.get("summonerName"),
                }
            ),
            None,
        )
        if not isinstance(active_player, dict) or not active_player.get("team"):
            return RosterResult(
                RosterStatus.INVALID_RESPONSE,
                error="Active player was not present in playerlist",
            )

        my_team = active_player["team"]
        members: list[RosterMember] = []
        for player in players:
            if not isinstance(player, dict) or player.get("team") == my_team:
                continue
            champion_name = player.get("championName")
            if isinstance(champion_name, str) and champion_name:
                members.append(
                    RosterMember(
                        champion_name=champion_name,
                        role=Role.from_api(player.get("position")),
                    )
                )
        if not members:
            return RosterResult(
                RosterStatus.INVALID_RESPONSE,
                error="No opponents were present in playerlist",
            )
        return RosterResult(RosterStatus.ACTIVE, tuple(members))
