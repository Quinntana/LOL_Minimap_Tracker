from __future__ import annotations

import logging
import time
from threading import Event
from typing import Any

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest

from lol_minimap_tracker.domain.cooldowns import (
    CooldownDefinition,
    CooldownKey,
    CooldownSlot,
    CooldownTimerStore,
    EnemyCooldownLoadout,
)
from lol_minimap_tracker.domain.models import (
    AffinityResult,
    AffinityStatus,
    Role,
    RosterMember,
    RosterState,
    SummonerSpellRef,
)
from lol_minimap_tracker.ui.cooldown_panel import CooldownPanel, format_remaining


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now

    def timestamp(self) -> str:
        return "2026-07-13 12:00:00"


class Affinity:
    def __init__(self, status: AffinityStatus = AffinityStatus.ACTIVE) -> None:
        self.status = status
        self.handles: list[int] = []

    def apply(self, handle: int, _enabled: bool) -> AffinityResult:
        self.handles.append(handle)
        return AffinityResult(self.status)


class Catalog:
    def __init__(self, loadouts: tuple[EnemyCooldownLoadout, ...]) -> None:
        self.loadouts = loadouts
        self.calls: list[tuple[RosterMember, ...]] = []

    def get_loadouts(self, members: tuple[RosterMember, ...]) -> tuple[EnemyCooldownLoadout, ...]:
        self.calls.append(members)
        return self.loadouts


def definition(
    identifier: str,
    name: str,
    cooldowns: tuple[float, ...],
    max_rank: int,
    reason: str | None = None,
) -> CooldownDefinition:
    return CooldownDefinition(identifier, name, None, cooldowns, max_rank, reason)


def fixture_data() -> tuple[tuple[RosterMember, ...], tuple[EnemyCooldownLoadout, ...]]:
    members = (
        RosterMember("Nami", Role.UTILITY, "support", level=11),
        RosterMember("Aatrox", Role.TOP, "top", level=11),
    )
    ultimate = definition("AatroxR", "World Ender", (120.0, 100.0, 80.0), 3)
    flash = definition("SummonerFlash", "Flash", (300.0,), 1)
    ignite = definition("SummonerDot", "Ignite", (180.0,), 1)
    loadouts = (
        EnemyCooldownLoadout("top", "Aatrox", None, ultimate, (flash, ignite)),
        EnemyCooldownLoadout("support", "Nami", None, ultimate, (flash, ignite)),
    )
    return members, loadouts


def settle(panel: CooldownPanel, qapp: Any) -> None:
    panel.refresh()
    assert panel._worker.wait_until_idle(2)
    panel.refresh()
    qapp.processEvents()


def test_remaining_display_rounds_up() -> None:
    assert format_remaining(125) == "2:05"
    assert format_remaining(60) == "1:00"
    assert format_remaining(59.1) == "1:00"
    assert format_remaining(0) == "0"


def test_panel_is_interactive_compact_and_capture_excluded(qapp: Any) -> None:
    members, loadouts = fixture_data()
    affinity = Affinity()
    panel = CooldownPanel(
        lambda: RosterState(1, members),
        Catalog(loadouts),
        CooldownTimerStore(Clock()),
        affinity,
        logging.getLogger("test"),
        capture_isolated=lambda: True,
        enabled=True,
        session_prefix="process-a",
    )
    panel.show()
    settle(panel, qapp)
    assert affinity.handles == [int(panel.winId())]
    assert not panel.windowFlags() & Qt.WindowTransparentForInput
    assert not panel.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert panel.windowFlags() & Qt.WindowDoesNotAcceptFocus
    assert panel.width() < 300
    assert [row.member.champion_name for row in panel.rows[:2] if row.member] == [
        "Aatrox",
        "Nami",
    ]
    assert len(panel.rows) == 5
    assert panel.timer_store.session_id == "process-a-1"
    assert panel.rows[2].member is None
    panel.shutdown()
    panel.close()


def test_left_click_starts_and_restarts_right_click_clears(qapp: Any) -> None:
    members, loadouts = fixture_data()
    clock = Clock()
    store = CooldownTimerStore(clock)
    panel = CooldownPanel(
        lambda: RosterState(7, members),
        Catalog(loadouts),
        store,
        Affinity(),
        logging.getLogger("test"),
        capture_isolated=lambda: True,
        enabled=True,
    )
    panel.show()
    settle(panel, qapp)
    button = panel.rows[0].buttons[CooldownSlot.ULTIMATE]
    key = CooldownKey("top", CooldownSlot.ULTIMATE)
    QTest.mouseClick(button, Qt.LeftButton)
    assert not button.isDown()
    first = store.snapshot(key)
    assert first is not None
    assert first.duration == 100.0
    clock.now += 5
    QTest.mouseClick(button, Qt.LeftButton)
    assert not button.isDown()
    restarted = store.snapshot(key)
    assert restarted is not None
    assert restarted.started_at == 105.0
    QTest.mouseClick(button, Qt.RightButton)
    assert not button.isDown()
    assert store.snapshot(key) is None
    panel.shutdown()
    panel.close()


def test_unsupported_slot_ignores_left_click_and_explains_reason(qapp: Any) -> None:
    members, loadouts = fixture_data()
    unsupported = definition("SummonerSmite", "Smite", (), 0, "Charge timer unsupported")
    changed = EnemyCooldownLoadout(
        "top",
        "Aatrox",
        None,
        loadouts[0].ultimate,
        (unsupported, loadouts[0].summoner_spells[1]),
    )
    store = CooldownTimerStore(Clock())
    panel = CooldownPanel(
        lambda: RosterState(2, members),
        Catalog((changed, loadouts[1])),
        store,
        Affinity(),
        logging.getLogger("test"),
        capture_isolated=lambda: True,
        enabled=True,
    )
    panel.show()
    settle(panel, qapp)
    button = panel.rows[0].buttons[CooldownSlot.SPELL_ONE]
    QTest.mouseClick(button, Qt.LeftButton)
    assert not button.isDown()
    assert len(store) == 0
    assert "Charge timer unsupported" in button.toolTip()
    panel.shutdown()
    panel.close()


def test_roster_generation_resets_timers_and_discards_old_rows(qapp: Any) -> None:
    members, loadouts = fixture_data()
    state = {"value": RosterState(1, members)}
    store = CooldownTimerStore(Clock())
    panel = CooldownPanel(
        lambda: state["value"],
        Catalog(loadouts),
        store,
        Affinity(),
        logging.getLogger("test"),
        capture_isolated=lambda: True,
        enabled=True,
    )
    panel.show()
    settle(panel, qapp)
    QTest.mouseClick(panel.rows[0].buttons[CooldownSlot.ULTIMATE], Qt.LeftButton)
    assert len(store) == 1
    state["value"] = RosterState(2, ())
    panel.refresh()
    assert len(store) == 0
    assert all(row.member is None for row in panel.rows)
    panel.shutdown()
    panel.close()


def test_same_match_spell_refresh_reloads_metadata_and_clears_only_changed_slot(
    qapp: Any,
) -> None:
    flash = definition("SummonerFlash", "Flash", (300.0,), 1)
    ignite = definition("SummonerDot", "Ignite", (180.0,), 1)
    barrier = definition("SummonerBarrier", "Barrier", (180.0,), 1)
    ultimate = definition("AatroxR", "World Ender", (120.0, 100.0, 80.0), 3)
    initial_member = RosterMember(
        "Aatrox",
        Role.TOP,
        "top",
        level=11,
        summoner_spells=(
            SummonerSpellRef("SummonerFlash", "Flash"),
            SummonerSpellRef("SummonerDot", "Ignite"),
        ),
    )
    updated_member = RosterMember(
        "Aatrox",
        Role.TOP,
        "top",
        level=12,
        summoner_spells=(
            SummonerSpellRef("SummonerFlash", "Flash"),
            SummonerSpellRef("SummonerBarrier", "Barrier"),
        ),
    )
    initial_loadout = EnemyCooldownLoadout("top", "Aatrox", None, ultimate, (flash, ignite))
    updated_loadout = EnemyCooldownLoadout("top", "Aatrox", None, ultimate, (flash, barrier))

    class RefreshingCatalog:
        def __init__(self) -> None:
            self.calls = 0

        def get_loadouts(
            self, _members: tuple[RosterMember, ...]
        ) -> tuple[EnemyCooldownLoadout, ...]:
            self.calls += 1
            return (initial_loadout,) if self.calls == 1 else (updated_loadout,)

    state = {"value": RosterState(4, (initial_member,))}
    store = CooldownTimerStore(Clock())
    catalog = RefreshingCatalog()
    panel = CooldownPanel(
        lambda: state["value"],
        catalog,
        store,
        Affinity(),
        logging.getLogger("test"),
        capture_isolated=lambda: True,
        enabled=True,
    )
    panel.show()
    settle(panel, qapp)
    for slot in CooldownSlot:
        QTest.mouseClick(panel.rows[0].buttons[slot], Qt.LeftButton)
    assert len(store) == 3

    state["value"] = RosterState(4, (updated_member,))
    panel.refresh()
    assert panel._worker.wait_until_idle(2)
    panel.refresh()

    assert catalog.calls == 2
    assert store.snapshot(CooldownKey("top", CooldownSlot.ULTIMATE)) is not None
    assert store.snapshot(CooldownKey("top", CooldownSlot.SPELL_ONE)) is not None
    assert store.snapshot(CooldownKey("top", CooldownSlot.SPELL_TWO)) is None
    assert panel.rows[0].buttons[CooldownSlot.SPELL_TWO].definition.identifier == "SummonerBarrier"
    panel.shutdown()
    panel.close()


def test_header_lock_hide_and_drag_signals(qapp: Any) -> None:
    panel = CooldownPanel(
        RosterState,
        Catalog(()),
        CooldownTimerStore(Clock()),
        Affinity(),
        logging.getLogger("test"),
        capture_isolated=lambda: True,
        enabled=True,
    )
    panel.show()
    qapp.processEvents()
    lock_values: list[bool] = []
    visibility: list[bool] = []
    positions: list[tuple[int, int]] = []
    panel.lock_requested.connect(lock_values.append)
    panel.visibility_requested.connect(visibility.append)
    panel.position_changed.connect(positions.append)
    panel.lock_button.click()
    panel.hide_button.click()
    assert lock_values == [True]
    assert visibility == [False]

    start = panel.pos()
    QTest.mousePress(panel, Qt.LeftButton, pos=QPoint(3, 10))
    QTest.mouseMove(panel, QPoint(23, 30))
    QTest.mouseRelease(panel, Qt.LeftButton, pos=QPoint(23, 30))
    assert positions
    assert positions[-1] == (panel.x(), panel.y())
    assert panel._clamped(start + QPoint(20, 20)) != start
    panel.set_locked(True)
    locked_position = panel.pos()
    QTest.mousePress(panel, Qt.LeftButton, pos=QPoint(3, 10))
    QTest.mouseMove(panel, QPoint(43, 50))
    QTest.mouseRelease(panel, Qt.LeftButton, pos=QPoint(43, 50))
    assert panel.pos() == locked_position
    panel.shutdown()
    panel.close()


def test_catalog_failure_retries_without_restarting_panel(qapp: Any) -> None:
    members, loadouts = fixture_data()

    class FlakyCatalog:
        def __init__(self) -> None:
            self.calls = 0

        def get_loadouts(
            self, _members: tuple[RosterMember, ...]
        ) -> tuple[EnemyCooldownLoadout, ...]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary CDN failure")
            return loadouts

    catalog = FlakyCatalog()
    panel = CooldownPanel(
        lambda: RosterState(1, members),
        catalog,
        CooldownTimerStore(Clock()),
        Affinity(),
        logging.getLogger("test"),
        capture_isolated=lambda: True,
        enabled=True,
    )
    panel.refresh()
    assert panel._worker.wait_until_idle(2)
    panel.refresh()
    assert catalog.calls == 1
    assert not panel._loadouts
    panel._catalog_retry_at = 0.0
    panel.refresh()
    assert panel._worker.wait_until_idle(2)
    panel.refresh()
    assert catalog.calls == 2
    assert set(panel._loadouts) == {"top", "support"}
    panel.shutdown()
    panel.close()


def test_desktop_capture_fails_closed_without_affinity(qapp: Any) -> None:
    blocked: list[str] = []
    panel = CooldownPanel(
        RosterState,
        Catalog(()),
        CooldownTimerStore(Clock()),
        Affinity(AffinityStatus.FAILED),
        logging.getLogger("test"),
        capture_isolated=lambda: False,
        enabled=True,
    )
    panel.safety_blocked.connect(blocked.append)
    panel.show()
    qapp.processEvents()
    assert blocked == ["failed"]
    assert not panel.isVisible()
    assert not panel.timer.isActive()
    panel.set_panel_visible(True)
    qapp.processEvents()
    assert blocked == ["failed", "failed"]
    assert not panel.isVisible()
    assert not panel.timer.isActive()
    panel.shutdown()
    panel.close()


def test_shutdown_cancels_catalog_without_waiting_for_slow_network(qapp: Any) -> None:
    class SlowCatalog:
        def __init__(self) -> None:
            self.started = Event()
            self.release = Event()
            self.cancelled = False

        def get_loadouts(
            self, _members: tuple[RosterMember, ...]
        ) -> tuple[EnemyCooldownLoadout, ...]:
            self.started.set()
            self.release.wait(2)
            return ()

        def cancel(self) -> None:
            self.cancelled = True

    catalog = SlowCatalog()
    member = RosterMember("Aatrox", Role.TOP, "top", level=11)
    panel = CooldownPanel(
        lambda: RosterState(1, (member,)),
        catalog,
        CooldownTimerStore(Clock()),
        Affinity(),
        logging.getLogger("test"),
        capture_isolated=lambda: True,
        enabled=True,
    )
    panel.refresh()
    assert catalog.started.wait(1)
    started = time.perf_counter()
    panel.shutdown()
    assert time.perf_counter() - started < 0.25
    assert catalog.cancelled
    catalog.release.set()
    panel._worker._thread.join(timeout=1)
    assert not panel._worker._thread.is_alive()
    panel.close()
    qapp.processEvents()
