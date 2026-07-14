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
NEXT_CDN = "https://next-cdn.example.test/ddragon"
NEXT_CHAMPION_VERSION = "16.14.1"
NEXT_SUMMONER_VERSION = "16.14.1"
NEXT_CHAMPION_URL = f"{NEXT_CDN}/{NEXT_CHAMPION_VERSION}/data/en_US/championFull.json"
NEXT_SUMMONER_URL = f"{NEXT_CDN}/{NEXT_SUMMONER_VERSION}/data/en_US/summoner.json"
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010804000000b51c0c02"
    "0000000b4944415478da6364f80f00010501012718e3660000000049454e44ae426082"
)
MIN_CHAMPION_RECORDS = 100
MIN_SUMMONER_RECORDS = 9


class Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


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


def next_realm_payload() -> dict[str, object]:
    return {
        "cdn": NEXT_CDN,
        "v": NEXT_CHAMPION_VERSION,
        "n": {
            "champion": NEXT_CHAMPION_VERSION,
            "summoner": NEXT_SUMMONER_VERSION,
        },
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


def champion_catalog(
    data: dict[str, object],
    *,
    version: str = CHAMPION_VERSION,
    count: int = MIN_CHAMPION_RECORDS,
) -> dict[str, object]:
    records = dict(data)
    index = 0
    while len(records) < count:
        identifier = f"FixtureChampion{index}"
        if identifier not in records:
            records[identifier] = champion(
                identifier,
                f"Fixture Champion {index}",
                ultimate(f"{identifier}R", [100, 80, 60], 3),
            )
        index += 1
    return {
        "type": "champion",
        "format": "full",
        "version": version,
        "data": records,
    }


def summoner_catalog(
    data: dict[str, object], *, version: str = SUMMONER_VERSION
) -> dict[str, object]:
    records = dict(data)
    for index in range(max(0, MIN_SUMMONER_RECORDS - len(records))):
        identifier = f"FixtureSummoner{index}"
        records[identifier] = summoner(identifier, f"Fixture Summoner {index}", [180])
    return {"type": "summoner", "version": version, "data": records}


def standard_catalogs() -> tuple[dict[str, object], dict[str, object]]:
    champions = champion_catalog(
        {
            "MonkeyKing": champion(
                "MonkeyKing",
                "Wukong",
                ultimate("MonkeyKingR", [130, 110, 90], 3, image="MonkeyKingR.png"),
                image="MonkeyKing.png",
            ),
        }
    )
    summoners = summoner_catalog(
        {
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
    )
    return champions, summoners


def next_catalogs() -> tuple[dict[str, object], dict[str, object]]:
    champions, summoners = standard_catalogs()
    champions["version"] = NEXT_CHAMPION_VERSION
    summoners["version"] = NEXT_SUMMONER_VERSION
    champion_data = champions["data"]
    assert isinstance(champion_data, dict)
    monkey_king = champion_data["MonkeyKing"]
    assert isinstance(monkey_king, dict)
    spells = monkey_king["spells"]
    assert isinstance(spells, list)
    ultimate_payload = spells[3]
    assert isinstance(ultimate_payload, dict)
    ultimate_payload["cooldown"] = [120, 100, 80]
    return champions, summoners


def next_online_routes() -> dict[str, list[Response | Exception]]:
    champions, summoners = next_catalogs()
    return {
        NEXT_CHAMPION_URL: [Response(champions)],
        NEXT_SUMMONER_URL: [Response(summoners)],
        f"{NEXT_CDN}/{NEXT_CHAMPION_VERSION}/img/champion/MonkeyKing.png": [Response(content=PNG)],
        f"{NEXT_CDN}/{NEXT_CHAMPION_VERSION}/img/spell/MonkeyKingR.png": [Response(content=PNG)],
        f"{NEXT_CDN}/{NEXT_SUMMONER_VERSION}/img/spell/SummonerFlash.png": [Response(content=PNG)],
        f"{NEXT_CDN}/{NEXT_SUMMONER_VERSION}/img/spell/SummonerHeal.png": [Response(content=PNG)],
    }


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
    champions = champion_catalog(
        {
            "Aatrox": champion("Aatrox", "Aatrox", ultimate("AatroxR", [120, 100, 80], 3)),
            "Jayce": champion("Jayce", "Jayce", ultimate("JayceStanceHtG", [6], 1)),
            "Elise": champion("Elise", "Elise", ultimate("EliseR", [4, 3, 2, 1], 4)),
            "Shyvana": champion("Shyvana", "Shyvana", ultimate("ShyvanaR", [0, 0, 0], 3)),
            "Udyr": champion("Udyr", "Udyr", ultimate("UdyrR", [6, 6, 6, 6, 6, 6], 6)),
            "Teemo": champion("Teemo", "Teemo", ultimate("TeemoRCast", [0.25, 0.25, 0.25], 3)),
            "Belveth": champion("Belveth", "Bel'Veth", ultimate("BelvethR", [1, 1, 1], 3)),
            "Glitch": champion("Glitch", "Glitch", ultimate("GlitchR", [100, 80], 3)),
        }
    )
    summoners = summoner_catalog(
        {
            "SummonerSmite": summoner("SummonerSmite", "Smite", [15]),
            "SummonerGhost": summoner("SummonerGhost", "Ghost", [240]),
        }
    )
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


def test_unknown_member_and_incomplete_records_return_disabled_placeholders(
    tmp_path: Path,
) -> None:
    champions, summoners = standard_catalogs()
    champion_data = champions["data"]
    summoner_data = summoners["data"]
    assert isinstance(champion_data, dict)
    assert isinstance(summoner_data, dict)
    champion_data["Broken"] = "not-a-record"
    summoner_data["Broken"] = "not-a-record"
    session = Session(
        {
            REALM_URL: [Response(realm_payload())],
            CHAMPION_URL: [Response(champions)],
            SUMMONER_URL: [Response(summoners)],
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
    client = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,
        retry_backoff_seconds=0.0,
    )

    with pytest.raises(CooldownMetadataUnavailable):
        client.get_loadouts((member(),))

    (loadout,) = client.get_loadouts((member(),))
    assert loadout.ultimate.identifier == "MonkeyKingR"
    assert session.calls.count(REALM_URL) == 2


def test_malformed_nonempty_catalogs_are_not_cached_as_loaded_and_retry(
    tmp_path: Path,
) -> None:
    champions, summoners = standard_catalogs()
    partial_champions = {
        "type": "champion",
        "format": "full",
        "version": CHAMPION_VERSION,
        "data": {
            "MonkeyKing": champion(
                "MonkeyKing",
                "Wukong",
                ultimate("MonkeyKingR", [130, 110, 90], 3),
            )
        },
    }
    partial_summoners = {
        "type": "summoner",
        "version": SUMMONER_VERSION,
        "data": {"SummonerFlash": summoner("SummonerFlash", "Flash", [300])},
    }
    session = Session(
        {
            REALM_URL: [Response(realm_payload()), Response(realm_payload())],
            CHAMPION_URL: [
                Response(partial_champions),
                Response(champions),
            ],
            SUMMONER_URL: [
                Response(partial_summoners),
                Response(summoners),
            ],
            f"{CDN}/{CHAMPION_VERSION}/img/champion/MonkeyKing.png": [Response(content=PNG)],
            f"{CDN}/{CHAMPION_VERSION}/img/spell/MonkeyKingR.png": [Response(content=PNG)],
            f"{CDN}/{SUMMONER_VERSION}/img/spell/SummonerFlash.png": [Response(content=PNG)],
            f"{CDN}/{SUMMONER_VERSION}/img/spell/SummonerHeal.png": [Response(content=PNG)],
        }
    )
    client = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,
        retry_backoff_seconds=0.0,
    )

    with pytest.raises(CooldownMetadataUnavailable):
        client.get_loadouts((member(),))

    (loadout,) = client.get_loadouts((member(),))

    assert loadout.ultimate.identifier == "MonkeyKingR"
    assert session.calls.count(CHAMPION_URL) == 2
    assert session.calls.count(SUMMONER_URL) == 2


def test_read_only_cache_does_not_discard_valid_catalogs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        Session(online_routes()),
    )

    def fail_write(_path: Path, _content: bytes) -> None:
        raise OSError("cache is read-only")

    monkeypatch.setattr(client, "_atomic_write", fail_write)

    (loadout,) = client.get_loadouts((member(),))

    assert loadout.ultimate.identifier == "MonkeyKingR"
    assert loadout.ultimate.cooldowns == (130.0, 110.0, 90.0)
    assert loadout.summoner_spells[0].identifier == "SummonerFlash"
    assert loadout.champion_icon_path is None


def test_long_running_client_atomically_switches_versions_after_realm_ttl(
    tmp_path: Path,
) -> None:
    clock = Clock()
    routes = online_routes()
    routes[REALM_URL] = [Response(realm_payload()), Response(next_realm_payload())]
    routes.update(next_online_routes())
    session = Session(routes)
    client = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,
        realm_ttl_seconds=10.0,
        retry_backoff_seconds=3.0,
        clock=clock,
    )

    first = client.get_loadouts((member(),))[0]
    clock.advance(9.0)
    before_ttl = client.get_loadouts((member(),))[0]

    assert first.ultimate.cooldowns == (130.0, 110.0, 90.0)
    assert before_ttl.ultimate.cooldowns == first.ultimate.cooldowns
    assert session.calls.count(REALM_URL) == 1

    clock.advance(1.0)
    refreshed = client.get_loadouts((member(),))[0]

    assert refreshed.ultimate.cooldowns == (120.0, 100.0, 80.0)
    assert refreshed.champion_icon_path is not None
    assert NEXT_CHAMPION_VERSION in str(refreshed.champion_icon_path)
    assert session.calls.count(REALM_URL) == 2
    assert NEXT_CHAMPION_URL in session.calls
    assert NEXT_SUMMONER_URL in session.calls


def test_partial_new_patch_outage_keeps_last_known_good_and_retries_after_backoff(
    tmp_path: Path,
) -> None:
    clock = Clock()
    routes = online_routes()
    routes[REALM_URL] = [
        Response(realm_payload()),
        Response(next_realm_payload()),
        requests.ConnectionError("realm endpoint briefly unavailable during retry"),
    ]
    next_routes = next_online_routes()
    next_summoners = next_routes[NEXT_SUMMONER_URL][0]
    next_routes[NEXT_SUMMONER_URL] = [
        requests.ConnectionError("summoner catalog is still rolling out"),
        next_summoners,
    ]
    routes.update(next_routes)
    session = Session(routes)
    client = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,
        realm_ttl_seconds=10.0,
        retry_backoff_seconds=5.0,
        clock=clock,
    )

    original = client.get_loadouts((member(),))[0]
    clock.advance(10.0)
    during_partial_rollout = client.get_loadouts((member(),))[0]
    calls_after_failure = tuple(session.calls)
    realm_cache = tmp_path / "cooldowns" / "ddragon" / "realms" / "vn.json"

    assert during_partial_rollout.ultimate.cooldowns == original.ultimate.cooldowns
    assert during_partial_rollout.champion_icon_path == original.champion_icon_path
    assert json.loads(realm_cache.read_text(encoding="utf-8"))["cdn"] == CDN
    assert NEXT_CHAMPION_URL in session.calls
    assert session.calls.count(NEXT_SUMMONER_URL) == 1

    restarted_session = Session(
        {
            REALM_URL: [Response(next_realm_payload())],
            NEXT_SUMMONER_URL: [requests.ConnectionError("still unavailable after restart")],
        }
    )
    restarted = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        restarted_session,
        retry_backoff_seconds=5.0,
        clock=Clock(),
    )
    restarted_loadout = restarted.get_loadouts((member(),))[0]
    assert restarted_loadout.ultimate.cooldowns == original.ultimate.cooldowns
    assert NEXT_CHAMPION_URL not in restarted_session.calls
    assert NEXT_SUMMONER_URL in restarted_session.calls
    assert json.loads(realm_cache.read_text(encoding="utf-8"))["cdn"] == CDN

    clock.advance(4.0)
    still_backing_off = client.get_loadouts((member(),))[0]
    assert still_backing_off.ultimate.cooldowns == original.ultimate.cooldowns
    assert tuple(session.calls) == calls_after_failure

    clock.advance(1.0)
    recovered = client.get_loadouts((member(),))[0]

    assert recovered.ultimate.cooldowns == (120.0, 100.0, 80.0)
    assert json.loads(realm_cache.read_text(encoding="utf-8"))["cdn"] == NEXT_CDN
    assert session.calls.count(NEXT_CHAMPION_URL) == 1
    assert session.calls.count(NEXT_SUMMONER_URL) == 2


def test_severely_shrunk_patch_catalog_is_not_cached_and_can_recover(tmp_path: Path) -> None:
    clock = Clock()
    old_champions, old_summoners = standard_catalogs()
    old_data = old_champions["data"]
    assert isinstance(old_data, dict)
    old_champions = champion_catalog(old_data, count=150)

    shrunk_champions, next_summoners = next_catalogs()
    next_data = shrunk_champions["data"]
    assert isinstance(next_data, dict)
    recovered_champions = champion_catalog(
        next_data,
        version=NEXT_CHAMPION_VERSION,
        count=150,
    )
    routes = online_routes()
    routes[REALM_URL] = [
        Response(realm_payload()),
        Response(next_realm_payload()),
        requests.ConnectionError("realm retry unavailable"),
    ]
    routes[CHAMPION_URL] = [Response(old_champions)]
    routes[SUMMONER_URL] = [Response(old_summoners)]
    routes[NEXT_CHAMPION_URL] = [
        Response(shrunk_champions),
        Response(recovered_champions),
    ]
    routes[NEXT_SUMMONER_URL] = [Response(next_summoners)]
    routes.update(
        {
            key: value
            for key, value in next_online_routes().items()
            if key not in {NEXT_CHAMPION_URL, NEXT_SUMMONER_URL}
        }
    )
    session = Session(routes)
    client = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,
        realm_ttl_seconds=10.0,
        retry_backoff_seconds=5.0,
        clock=clock,
    )

    original = client.get_loadouts((member(),))[0]
    clock.advance(10.0)
    rejected = client.get_loadouts((member(),))[0]
    next_cache = (
        tmp_path
        / "cooldowns"
        / "ddragon"
        / NEXT_CHAMPION_VERSION
        / "data"
        / "en_US"
        / "championFull.json"
    )

    assert rejected.ultimate.cooldowns == original.ultimate.cooldowns
    assert not next_cache.exists()

    clock.advance(5.0)
    recovered = client.get_loadouts((member(),))[0]

    assert recovered.ultimate.cooldowns == (120.0, 100.0, 80.0)
    assert next_cache.exists()


def test_initial_failure_is_rate_limited_until_retry_backoff_expires(tmp_path: Path) -> None:
    clock = Clock(100.0)
    routes = online_routes()
    routes[REALM_URL].insert(0, requests.ConnectionError("temporary outage"))
    session = Session(routes)
    client = CooldownDataDragonClient(
        tmp_path,
        logging.getLogger("test"),
        session,
        retry_backoff_seconds=5.0,
        clock=clock,
    )

    with pytest.raises(CooldownMetadataUnavailable):
        client.get_loadouts((member(),))
    with pytest.raises(CooldownMetadataUnavailable):
        client.get_loadouts((member(),))
    assert session.calls.count(REALM_URL) == 1

    clock.advance(5.0)
    recovered = client.get_loadouts((member(),))[0]

    assert recovered.ultimate.identifier == "MonkeyKingR"
    assert session.calls.count(REALM_URL) == 2


def test_truncated_cached_png_is_replaced_instead_of_trusted(tmp_path: Path) -> None:
    cached_icon = (
        tmp_path
        / "cooldowns"
        / "ddragon"
        / CHAMPION_VERSION
        / "img"
        / "champion"
        / "MonkeyKing.png"
    )
    cached_icon.parent.mkdir(parents=True)
    cached_icon.write_bytes(PNG[:24])
    session = Session(online_routes())
    client = CooldownDataDragonClient(tmp_path, logging.getLogger("test"), session)

    loadout = client.get_loadouts((member(),))[0]

    assert loadout.champion_icon_path == cached_icon
    assert cached_icon.read_bytes() == PNG
    assert f"{CDN}/{CHAMPION_VERSION}/img/champion/MonkeyKing.png" in session.calls


def test_cancel_prevents_new_network_and_cache_work(tmp_path: Path) -> None:
    session = Session(online_routes())
    client = CooldownDataDragonClient(tmp_path, logging.getLogger("test"), session)
    client.cancel()

    (loadout,) = client.get_loadouts((member(),))

    assert session.calls == []
    assert loadout.ultimate.unsupported_reason == "Champion metadata is unavailable"
    assert not (tmp_path / "cooldowns" / "ddragon" / "realms" / "vn.json").exists()
