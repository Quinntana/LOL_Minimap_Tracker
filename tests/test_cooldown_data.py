from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import requests

from lol_minimap_tracker.domain.models import Role, RosterMember, SummonerSpellRef
from lol_minimap_tracker.integrations.cooldown_data import (
    CooldownDataDragonClient,
    CooldownMetadataUnavailable,
)

BASE_URL = "https://ddragon.leagueoflegends.com"
REALM_URL = f"{BASE_URL}/realms/vn.json"
CDN = "https://cdn.example.test/ddragon"
CHAMPION_VERSION = "16.13.1"
SUMMONER_VERSION = "16.12.1"
CHAMPION_URL = f"{CDN}/{CHAMPION_VERSION}/data/en_US/championFull.json"
SUMMONER_URL = f"{CDN}/{SUMMONER_VERSION}/data/en_US/summoner.json"
PNG = b"\x89PNG\r\n\x1a\nfixture"


class Response:
    def __init__(self, payload: object = None, content: bytes = b"") -> None:
        self.payload = payload
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class Session:
    def __init__(self, routes: dict[str, list[Response | Exception]]) -> None:
        self.routes = routes
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url: str, *, timeout: tuple[float, float]) -> Response:
        assert timeout[0] > 0
        self.calls.append(url)
        responses = self.routes.get(url)
        if not responses:
            raise requests.ConnectionError(f"offline: {url}")
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def realm_payload() -> dict[str, object]:
    return {
        "cdn": CDN,
        "v": CHAMPION_VERSION,
        "n": {"champion": CHAMPION_VERSION, "summoner": SUMMONER_VERSION},
    }


def ultimate(
    identifier: str,
    cooldowns: list[object],
    max_rank: object,
    *,
    image: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": identifier,
        "name": f"{identifier} name",
        "cooldown": cooldowns,
        "maxrank": max_rank,
    }
    if image:
        result["image"] = {"full": image}
    return result


def champion(
    identifier: str,
    name: str,
    ultimate_payload: dict[str, object],
    *,
    image: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": identifier,
        "name": name,
        "spells": [{}, {}, {}, ultimate_payload],
    }
    if image:
        result["image"] = {"full": image}
    return result


def summoner(
    identifier: str,
    name: str,
    cooldown: object,
    *,
    image: str | None = None,
    modes: tuple[str, ...] = ("CLASSIC",),
    max_rank: object = 1,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": identifier,
        "name": name,
        "cooldown": cooldown,
        "maxrank": max_rank,
        "modes": list(modes),
    }
    if image:
        result["image"] = {"full": image}
    return result


def standard_catalogs() -> tuple[dict[str, object], dict[str, object]]:
    champions: dict[str, object] = {
        "data": {
            "MonkeyKing": champion(
                "MonkeyKing",
                "Wukong",
                ultimate("MonkeyKingR", [130, 110, 90], 3, image="MonkeyKingR.png"),
                image="MonkeyKing.png",
            )
        }
    }
    summoners: dict[str, object] = {
        "data": {
            # Put the duplicate first to prove name fallback is not insertion-order based.
            "SummonerCherryFlash": summoner(
                "SummonerCherryFlash",
                "Flash",
                [0.25],
                image="CherryFlash.png",
                modes=("CHERRY",),
            ),
            "SummonerFlash": summoner(
                "SummonerFlash",
                "Flash",
                [300],
                image="SummonerFlash.png",
            ),
            "SummonerHeal": summoner(
                "SummonerHeal",
                "Heal",
                [240],
                image="SummonerHeal.png",
            ),
        }
    }
    return champions, summoners


def member(
    champion_name: str = "Wukong",
    *,
    champion_id: str | None = None,
    first: SummonerSpellRef | None = None,
    second: SummonerSpellRef | None = None,
) -> RosterMember:
    resolved_first = first or SummonerSpellRef(None, "Flash")
    resolved_second = second or SummonerSpellRef("SummonerHeal", "Heal")
    return RosterMember(
        champion_name=champion_name,
        role=Role.TOP,
        participant_id="enemy-1",
        champion_id=champion_id,
        level=11,
        summoner_spells=(resolved_first, resolved_second),
    )


def online_routes(*, image_bytes: bytes = PNG) -> dict[str, list[Response | Exception]]:
    champions, summoners = standard_catalogs()
    return {
        REALM_URL: [Response(realm_payload())],
        CHAMPION_URL: [Response(champions)],
        SUMMONER_URL: [Response(summoners)],
        f"{CDN}/{CHAMPION_VERSION}/img/champion/MonkeyKing.png": [Response(content=image_bytes)],
        f"{CDN}/{CHAMPION_VERSION}/img/spell/MonkeyKingR.png": [Response(content=image_bytes)],
        f"{CDN}/{SUMMONER_VERSION}/img/spell/SummonerFlash.png": [Response(content=image_bytes)],
        f"{CDN}/{SUMMONER_VERSION}/img/spell/SummonerHeal.png": [Response(content=image_bytes)],
    }


def test_regional_versions_aliases_duplicate_names_and_assets_are_cached(tmp_path: Path) -> None:
    session = Session(online_routes())
    client = CooldownDataDragonClient(tmp_path, logging.getLogger("test"), session)

    (loadout,) = client.get_loadouts((member(),))

    assert loadout.participant_id == "enemy-1"
    assert loadout.champion_name == "Wukong"
    assert loadout.champion_icon_path is not None
    assert loadout.champion_icon_path.read_bytes() == PNG
    assert loadout.ultimate.identifier == "MonkeyKingR"
    assert loadout.ultimate.cooldowns == (130.0, 110.0, 90.0)
    assert loadout.ultimate.duration_for_level(11) == 110.0
    assert loadout.summoner_spells[0].identifier == "SummonerFlash"
    assert loadout.summoner_spells[0].cooldowns == (300.0,)
    assert loadout.summoner_spells[1].cooldowns == (240.0,)
    assert CHAMPION_URL in session.calls
    assert SUMMONER_URL in session.calls

    cache = tmp_path / "cooldowns" / "ddragon"
    assert (cache / CHAMPION_VERSION / "data" / "en_US" / "championFull.json").exists()
    assert (cache / SUMMONER_VERSION / "data" / "en_US" / "summoner.json").exists()
    assert not tuple(cache.rglob("*.tmp"))


def test_raw_summoner_identifier_wins_before_duplicate_display_name(tmp_path: Path) -> None:
    routes = online_routes()
    routes[f"{CDN}/{SUMMONER_VERSION}/img/spell/CherryFlash.png"] = [Response(content=PNG)]
    client = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        Session(routes),
    )
    cherry = member(first=SummonerSpellRef("SummonerCherryFlash", "Flash"))

    (loadout,) = client.get_loadouts((cherry,))

    assert loadout.summoner_spells[0].identifier == "SummonerCherryFlash"
    assert loadout.summoner_spells[0].cooldowns == (0.25,)


def test_cached_catalogs_and_icons_work_when_fully_offline(tmp_path: Path) -> None:
    online = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        Session(online_routes()),
    )
    expected = online.get_loadouts((member(),))[0]

    offline_session = Session({})
    offline = CooldownDataDragonClient(tmp_path, logging.getLogger("test"), offline_session)
    actual = offline.get_loadouts((member(),))[0]

    assert actual == expected
    assert offline_session.calls == [REALM_URL]


def test_malformed_remote_catalogs_fall_back_to_valid_cache(tmp_path: Path) -> None:
    warm = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        Session(online_routes()),
    )
    warm.get_loadouts((member(),))
    session = Session(
        {
            REALM_URL: [Response(realm_payload())],
            CHAMPION_URL: [Response({"data": []})],
            SUMMONER_URL: [Response({"wrong": "shape"})],
        }
    )

    (loadout,) = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,
    ).get_loadouts((member(),))

    assert loadout.ultimate.cooldowns == (130.0, 110.0, 90.0)
    assert loadout.summoner_spells[0].identifier == "SummonerFlash"


def test_rank_exceptions_and_dynamic_cooldowns_are_safe(tmp_path: Path) -> None:
    champions = {
        "data": {
            "Aatrox": champion("Aatrox", "Aatrox", ultimate("AatroxR", [120, 100, 80], 3)),
            "Jayce": champion("Jayce", "Jayce", ultimate("JayceStanceHtG", [6], 1)),
            "Elise": champion("Elise", "Elise", ultimate("EliseR", [4, 3, 2, 1], 4)),
            "Shyvana": champion("Shyvana", "Shyvana", ultimate("ShyvanaR", [0, 0, 0], 3)),
            "Udyr": champion("Udyr", "Udyr", ultimate("UdyrR", [6, 6, 6, 6, 6, 6], 6)),
            "Teemo": champion("Teemo", "Teemo", ultimate("TeemoRCast", [0.25, 0.25, 0.25], 3)),
            "Belveth": champion("Belveth", "Bel'Veth", ultimate("BelvethR", [1, 1, 1], 3)),
            "Glitch": champion("Glitch", "Glitch", ultimate("GlitchR", [100, 80], 3)),
        }
    }
    summoners = {
        "data": {
            "SummonerSmite": summoner("SummonerSmite", "Smite", [15]),
            "SummonerGhost": summoner("SummonerGhost", "Ghost", [240]),
        }
    }
    session = Session(
        {
            REALM_URL: [Response(realm_payload())],
            CHAMPION_URL: [Response(champions)],
            SUMMONER_URL: [Response(summoners)],
        }
    )
    client = CooldownDataDragonClient(tmp_path, logging.getLogger("test"), session)
    smite = SummonerSpellRef("SummonerSmite", "Smite")
    ghost = SummonerSpellRef("SummonerGhost", "Ghost")
    members = tuple(
        member(name, first=smite, second=ghost)
        for name in (
            "Aatrox",
            "Jayce",
            "Elise",
            "Shyvana",
            "Udyr",
            "Teemo",
            "Belveth",
            "Glitch",
        )
    )

    loadouts = {loadout.champion_name: loadout for loadout in client.get_loadouts(members)}

    assert loadouts["Aatrox"].ultimate.duration_for_level(16) == 80.0
    assert loadouts["Jayce"].ultimate.duration_for_level(1) == 6.0
    assert loadouts["Elise"].ultimate.duration_for_level(11) == 2.0
    assert loadouts["Shyvana"].ultimate.cooldowns == ()
    assert "dynamic" in (loadouts["Shyvana"].ultimate.unsupported_reason or "").casefold()
    assert loadouts["Udyr"].ultimate.duration_for_level(18) is None
    assert "rank" in (loadouts["Udyr"].ultimate.unsupported_reason or "").casefold()
    assert loadouts["Teemo"].ultimate.duration_for_level(18) is None
    assert "charge" in (loadouts["Teemo"].ultimate.unsupported_reason or "").casefold()
    assert loadouts["Belveth"].ultimate.duration_for_level(11) is None
    assert "resource" in (loadouts["Belveth"].ultimate.unsupported_reason or "").casefold()
    assert loadouts["Glitch"].ultimate.cooldowns == ()
    assert loadouts["Aatrox"].summoner_spells[0].duration_for_level(18) is None
    assert "charge" in (loadouts["Aatrox"].summoner_spells[0].unsupported_reason or "").casefold()
    assert loadouts["Aatrox"].summoner_spells[1].duration_for_level(1) == 240.0


def test_champion_id_is_used_before_display_name(tmp_path: Path) -> None:
    routes = online_routes()
    client = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        Session(routes),
    )

    (loadout,) = client.get_loadouts((member("Localized Wukong", champion_id="MonkeyKing"),))

    assert loadout.champion_name == "Localized Wukong"
    assert loadout.ultimate.identifier == "MonkeyKingR"


def test_unknown_and_incomplete_records_return_disabled_placeholders(tmp_path: Path) -> None:
    session = Session(
        {
            REALM_URL: [Response(realm_payload())],
            CHAMPION_URL: [Response({"data": {"Broken": "not-a-record"}})],
            SUMMONER_URL: [Response({"data": {"Broken": "not-a-record"}})],
        }
    )
    roster_member = member(
        "Unknown",
        first=SummonerSpellRef("NoSuchSpell", "Mystery"),
        second=SummonerSpellRef(None, "Also Missing"),
    )

    (loadout,) = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,
    ).get_loadouts((roster_member,))

    assert loadout.champion_icon_path is None
    assert loadout.ultimate.unsupported_reason == "Champion metadata is unavailable"
    assert all(spell.unsupported_reason is not None for spell in loadout.summoner_spells)


def test_invalid_icon_content_does_not_disable_valid_cooldowns(tmp_path: Path) -> None:
    client = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        Session(online_routes(image_bytes=b"not-a-png")),
    )

    (loadout,) = client.get_loadouts((member(),))

    assert loadout.champion_icon_path is None
    assert loadout.ultimate.icon_path is None
    assert loadout.ultimate.duration_for_level(11) == 110.0
    assert all(spell.icon_path is None for spell in loadout.summoner_spells)


def test_invalid_realm_raises_retryable_error_without_corrupting_state(tmp_path: Path) -> None:
    cache = tmp_path / "cooldowns" / "ddragon" / "realms"
    cache.mkdir(parents=True)
    (cache / "vn.json").write_text(json.dumps({"cdn": "not-a-url"}), encoding="utf-8")
    session = Session({REALM_URL: [Response({"n": {"champion": "../bad"}})]})

    client = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,
    )

    with pytest.raises(CooldownMetadataUnavailable):
        client.get_loadouts((member(),))


def test_transient_initial_failure_retries_and_recovers(tmp_path: Path) -> None:
    routes = online_routes()
    routes[REALM_URL].insert(0, requests.ConnectionError("temporary outage"))
    session = Session(routes)
    client = CooldownDataDragonClient(tmp_path, logging.getLogger("test"), session)

    with pytest.raises(CooldownMetadataUnavailable):
        client.get_loadouts((member(),))

    (loadout,) = client.get_loadouts((member(),))
    assert loadout.ultimate.identifier == "MonkeyKingR"
    assert session.calls.count(REALM_URL) == 2


def test_cancel_prevents_new_network_and_cache_work(tmp_path: Path) -> None:
    session = Session(online_routes())
    client = CooldownDataDragonClient(tmp_path, logging.getLogger("test"), session)
    client.cancel()

    (loadout,) = client.get_loadouts((member(),))

    assert session.calls == []
    assert loadout.ultimate.unsupported_reason == "Champion metadata is unavailable"
    assert not (tmp_path / "cooldowns" / "ddragon" / "realms" / "vn.json").exists()
