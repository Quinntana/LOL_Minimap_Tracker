from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from lol_minimap_tracker.domain.cooldowns import (
    CooldownAction,
    CooldownDefinition,
    CooldownEvent,
    CooldownKey,
    CooldownSlot,
    CooldownTimerStore,
    EnemyCooldownLoadout,
)
from lol_minimap_tracker.integrations.cooldown_events import CsvCooldownEventSink


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now

    def timestamp(self) -> str:
        return f"2026-07-14 12:00:{int(self.now) % 60:02d}"


class Recorder:
    def __init__(self) -> None:
        self.events: list[CooldownEvent] = []

    def record(self, event: CooldownEvent) -> None:
        self.events.append(event)


def definition(
    cooldowns: tuple[float, ...] = (120.0, 100.0, 80.0),
    max_rank: int = 3,
    unsupported_reason: str | None = None,
) -> CooldownDefinition:
    return CooldownDefinition(
        identifier="AatroxR",
        display_name="World Ender",
        icon_path=Path("AatroxR.png"),
        cooldowns=cooldowns,
        max_rank=max_rank,
        unsupported_reason=unsupported_reason,
    )


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (1, None),
        (5, None),
        (6, 120.0),
        (10, 120.0),
        (11, 100.0),
        (15, 100.0),
        (16, 80.0),
        (18, 80.0),
    ],
)
def test_three_rank_duration_boundaries(level: int, expected: float | None) -> None:
    assert definition().duration_for_level(level) == expected


@pytest.mark.parametrize(
    ("level", "expected"),
    [(1, 40.0), (5, 40.0), (6, 38.0), (10, 38.0), (11, 36.0), (15, 36.0), (16, 34.0), (18, 34.0)],
)
def test_four_rank_duration_boundaries(level: int, expected: float) -> None:
    assert definition((40.0, 38.0, 36.0, 34.0), 4).duration_for_level(level) == expected


def test_one_rank_and_invalid_definitions_are_conservative() -> None:
    assert definition((300.0,), 1).duration_for_level(1) == 300.0
    assert definition((300.0,), 1).duration_for_level(18) == 300.0
    assert definition().duration_for_level(0) is None
    assert definition().duration_for_level(19) is None
    assert definition((6.0,) * 6, 6).duration_for_level(18) is None
    assert definition((120.0, 100.0), 3).duration_for_level(16) is None
    assert definition((120.0, 0.0, 80.0), 3).duration_for_level(16) is None
    assert definition((120.0, float("nan"), 80.0), 3).duration_for_level(16) is None
    assert definition(unsupported_reason="Resource based").duration_for_level(16) is None


def test_loadout_requires_two_spells_and_maps_slots() -> None:
    ultimate = definition()
    flash = CooldownDefinition("SummonerFlash", "Flash", None, (300.0,), 1, None)
    heal = CooldownDefinition("SummonerHeal", "Heal", None, (240.0,), 1, None)
    loadout = EnemyCooldownLoadout("enemy-1", "Aatrox", None, ultimate, (flash, heal))
    assert loadout.definition_for(CooldownSlot.ULTIMATE) is ultimate
    assert loadout.definition_for(CooldownSlot.SPELL_ONE) is flash
    assert loadout.definition_for(CooldownSlot.SPELL_TWO) is heal
    with pytest.raises(ValueError, match="exactly two"):
        EnemyCooldownLoadout("enemy-1", "Aatrox", None, ultimate, (flash,))  # type: ignore[arg-type]


def test_start_snapshots_duration_level_and_emits_start() -> None:
    clock = Clock()
    recorder = Recorder()
    store = CooldownTimerStore(clock, recorder, session_id="match-1")
    key = CooldownKey("enemy-1", CooldownSlot.ULTIMATE)
    snapshot = store.start(key, "Aatrox", definition(), 11)
    assert snapshot is not None
    assert snapshot.duration == 100.0
    assert snapshot.level == 11
    assert snapshot.started_at == 100.0
    assert snapshot.ready_at == 200.0
    assert snapshot.remaining == 100.0
    assert not snapshot.is_ready
    assert recorder.events == [
        CooldownEvent(
            "2026-07-14 12:00:40",
            "match-1",
            key,
            "Aatrox",
            "AatroxR",
            CooldownAction.START,
            100.0,
            11,
            100.0,
        )
    ]


def test_restart_uses_new_level_and_new_full_duration() -> None:
    clock = Clock()
    recorder = Recorder()
    store = CooldownTimerStore(clock, recorder)
    key = CooldownKey("enemy-1", CooldownSlot.ULTIMATE)
    store.start(key, "Aatrox", definition(), 6)
    clock.now += 20.0
    snapshot = store.start(key, "Aatrox", definition(), 16)
    assert snapshot is not None
    assert snapshot.duration == 80.0
    assert snapshot.remaining == 80.0
    assert snapshot.ready_at == 200.0
    assert [event.action for event in recorder.events] == [
        CooldownAction.START,
        CooldownAction.RESTART,
    ]


def test_unsupported_start_does_not_replace_existing_timer() -> None:
    clock = Clock()
    store = CooldownTimerStore(clock)
    key = CooldownKey("enemy-1", CooldownSlot.ULTIMATE)
    original = store.start(key, "Aatrox", definition(), 6)
    assert original is not None
    assert store.start(key, "Aatrox", definition(), 5) is None
    assert store.snapshot(key) == original


def test_remaining_reaches_zero_and_ready_is_emitted_only_once() -> None:
    clock = Clock()
    recorder = Recorder()
    store = CooldownTimerStore(clock, recorder)
    key = CooldownKey("enemy-1", CooldownSlot.ULTIMATE)
    store.start(key, "Aatrox", definition(), 16)
    clock.now = 179.999
    assert store.remaining(key) == pytest.approx(0.001)
    assert [event.action for event in recorder.events] == [CooldownAction.START]
    clock.now = 180.0
    snapshot = store.snapshot(key)
    assert snapshot is not None and snapshot.is_ready and snapshot.remaining == 0.0
    store.snapshot(key)
    store.snapshots()
    assert [event.action for event in recorder.events] == [
        CooldownAction.START,
        CooldownAction.READY,
    ]


def test_clear_emits_cancel_with_remaining_and_removes_one() -> None:
    clock = Clock()
    recorder = Recorder()
    store = CooldownTimerStore(clock, recorder)
    key = CooldownKey("enemy-1", CooldownSlot.ULTIMATE)
    store.start(key, "Aatrox", definition(), 16)
    clock.now += 30.0
    assert store.clear(key)
    assert not store.clear(key)
    assert store.snapshot(key) is None
    assert recorder.events[-1].action is CooldownAction.CANCEL
    assert recorder.events[-1].remaining == 50.0


def test_clear_all_cancels_each_timer() -> None:
    clock = Clock()
    recorder = Recorder()
    store = CooldownTimerStore(clock, recorder)
    ultimate = CooldownKey("enemy-1", CooldownSlot.ULTIMATE)
    flash = CooldownKey("enemy-1", CooldownSlot.SPELL_ONE)
    store.start(ultimate, "Aatrox", definition(), 16)
    store.start(flash, "Aatrox", definition((300.0,), 1), 16)
    assert store.clear_all() == 2
    assert len(store) == 0
    assert [event.action for event in recorder.events[-2:]] == [
        CooldownAction.CANCEL,
        CooldownAction.CANCEL,
    ]


def test_reset_emits_old_session_events_then_changes_session() -> None:
    clock = Clock()
    recorder = Recorder()
    store = CooldownTimerStore(clock, recorder, session_id="old-match")
    key = CooldownKey("enemy-1", CooldownSlot.ULTIMATE)
    store.start(key, "Aatrox", definition(), 16)
    assert store.reset_session("new-match") == 1
    assert store.session_id == "new-match"
    reset = recorder.events[-1]
    assert reset.action is CooldownAction.RESET
    assert reset.session_id == "old-match"
    assert len(store) == 0
    store.start(key, "Aatrox", definition(), 16)
    assert recorder.events[-1].session_id == "new-match"


def test_restart_after_expiry_records_ready_before_restart() -> None:
    clock = Clock()
    recorder = Recorder()
    store = CooldownTimerStore(clock, recorder)
    key = CooldownKey("enemy-1", CooldownSlot.ULTIMATE)
    store.start(key, "Aatrox", definition(), 16)
    clock.now = 200.0
    store.start(key, "Aatrox", definition(), 16)
    assert [event.action for event in recorder.events] == [
        CooldownAction.START,
        CooldownAction.READY,
        CooldownAction.RESTART,
    ]


def test_snapshots_are_stably_sorted() -> None:
    clock = Clock()
    store = CooldownTimerStore(clock)
    second = CooldownKey("enemy-2", CooldownSlot.ULTIMATE)
    first_spell = CooldownKey("enemy-1", CooldownSlot.SPELL_ONE)
    first_ultimate = CooldownKey("enemy-1", CooldownSlot.ULTIMATE)
    for key in (second, first_spell, first_ultimate):
        store.start(key, "Aatrox", definition((300.0,), 1), 1)
    assert [snapshot.key for snapshot in store.snapshots()] == [
        first_spell,
        first_ultimate,
        second,
    ]


def test_csv_sink_writes_required_schema_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "cooldowns" / "events.csv"
    sink = CsvCooldownEventSink(path)
    event = CooldownEvent(
        timestamp="2026-07-14 12:00:00",
        session_id="match-1",
        key=CooldownKey("enemy-1", CooldownSlot.ULTIMATE),
        champion_name="Aatrox",
        identifier="AatroxR",
        action=CooldownAction.START,
        duration=120.0,
        level=6,
        remaining=120.0,
    )
    sink.record(event)
    sink.flush()
    sink.record(event)
    sink.flush()
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == list(CsvCooldownEventSink.FIELDNAMES)
    assert len(rows) == 2
    assert rows[0] == {
        "timestamp": "2026-07-14 12:00:00",
        "session": "match-1",
        "participant": "enemy-1",
        "champion": "Aatrox",
        "slot": "ultimate",
        "identifier": "AatroxR",
        "action": "start",
        "duration": "120.0",
        "level": "6",
        "remaining": "120.0",
    }


def test_empty_csv_flush_does_not_create_file(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    CsvCooldownEventSink(path).flush()
    assert not path.exists()


def test_failed_csv_flush_restores_buffer(tmp_path: Path, monkeypatch: Any) -> None:
    path = tmp_path / "events.csv"
    sink = CsvCooldownEventSink(path)
    sink.record(
        CooldownEvent(
            "now",
            "match",
            CooldownKey("enemy", CooldownSlot.SPELL_TWO),
            "Aatrox",
            "SummonerFlash",
            CooldownAction.START,
            300.0,
            1,
            300.0,
        )
    )
    original_open = Path.open

    def fail_open(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(OSError, match="disk unavailable"):
        sink.flush()
    monkeypatch.setattr(Path, "open", original_open)
    sink.flush()
    assert "SummonerFlash" in path.read_text(encoding="utf-8")
