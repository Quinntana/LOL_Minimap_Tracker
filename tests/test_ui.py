from __future__ import annotations

from typing import Any

from PyQt5.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PyQt5.QtGui import QColor, QImage, QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QSystemTrayIcon

from lol_minimap_tracker.config import CaptureRegion, TrackerConfig
from lol_minimap_tracker.domain.models import (
    AffinityResult,
    AffinityStatus,
    ChampionView,
    EnemyIdentity,
    Role,
    RosterStatus,
    TrackerMode,
    TrackerSnapshot,
)
from lol_minimap_tracker.paths import AppPaths
from lol_minimap_tracker.ui.calibration import (
    CalibrationController,
    RegionSelector,
    region_from_selection,
)
from lol_minimap_tracker.ui.overlay import TransparentOverlay
from lol_minimap_tracker.ui.role_icons import RoleIconRenderer
from lol_minimap_tracker.ui.tray import TrayController


class Affinity:
    def apply(self, _handle: int, _enabled: bool) -> AffinityResult:
        return AffinityResult(AffinityStatus.ACTIVE)


def snapshot() -> TrackerSnapshot:
    top = EnemyIdentity("Aatrox", Role.TOP, "#E69F00", "position-top.svg")
    jungle = EnemyIdentity("Nidalee", Role.JUNGLE, "#009E73", "position-jungle.svg")
    return TrackerSnapshot(
        mode=TrackerMode.ACTIVE,
        roster_status=RosterStatus.ACTIVE,
        champions=(
            ChampionView(top, (30, 40), True, 0.1),
            ChampionView(jungle, (70, 80), False, 6.0),
        ),
        camera_center=(50, 50),
        timeline_logging=True,
        message="Tracking Aatrox, Nidalee",
    )


def test_role_icons_render_and_cache(qapp: object) -> None:
    del qapp
    renderer = RoleIconRenderer(AppPaths.discover().role_asset_dir)
    first = renderer.render("position-top.svg", "#E69F00", 16, 0.35)
    second = renderer.render("position-top.svg", "#E69F00", 16, 0.35)
    assert not first.isNull()
    assert first.cacheKey() == second.cacheKey()
    assert not renderer.render("position-unknown.svg", "#B8B8B8", 18).isNull()


def test_overlay_renders_safe_and_fallback_modes(qapp: Any) -> None:
    current = snapshot()
    changed: list[AffinityResult] = []
    overlay = TransparentOverlay(
        lambda: current,
        TrackerConfig(capture=CaptureRegion(200, 300, 100, 100)),
        RoleIconRenderer(AppPaths.discover().role_asset_dir),
        Affinity(),
        changed.append,
    )
    overlay.snapshot = current
    overlay.show()
    qapp.processEvents()
    assert changed[-1].status is AffinityStatus.ACTIVE
    overlay.affinity_result = AffinityResult(AffinityStatus.ACTIVE)
    image = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(0)
    overlay.render(image)
    assert not image.isNull()
    assert not overlay.toggle_arrows()
    assert not overlay.toggle_last_seen()
    overlay.update_overlay()
    overlay.set_capture_region(CaptureRegion(-100, -200, 120, 110))
    assert overlay.capture_region == CaptureRegion(-100, -200, 120, 110)

    overlay.affinity_result = AffinityResult(AffinityStatus.FAILED)
    fallback = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    fallback.fill(0)
    overlay.render(fallback)
    overlay.close()


def test_tray_updates_and_dispatches(qapp: Any) -> None:
    dispatched: list[str] = []
    tray = TrayController(qapp, dispatched.append)
    tray.toggle_actions["toggle_arrows"].trigger()
    assert dispatched == ["toggle_arrows"]
    tray.update_action("toggle_arrows", True)
    assert tray.toggle_actions["toggle_arrows"].isChecked()
    tray.update_snapshot(snapshot())
    assert "Tracking Aatrox" in tray.status_action.text()
    assert "waiting for live frames" in tray.analysis_action.text().lower()
    assert tray.region_action.text() == "Minimap: 252 x 252 at 1655, 813"
    tray.update_affinity(AffinityResult(AffinityStatus.ACTIVE))
    assert tray.affinity_action.text() == "Capture exclusion: active"
    top_level = [action.text() for action in tray.menu.actions()]
    assert "Select minimap area..." in top_level
    assert "Direction arrows" in top_level
    assert "Last-seen markers" in top_level
    assert "Pause detection" in top_level
    assert "Save timeline" not in top_level
    save_action = next(
        action for action in tray.advanced_menu.actions() if action.text() == "Save timeline"
    )
    assert any(action.text() == "Open configuration" for action in tray.advanced_menu.actions())
    save_action.trigger()
    assert dispatched[-1] == "save_timeline"
    tray._activated(QSystemTrayIcon.Trigger)
    assert tray.menu.isVisible()
    tray.menu.hide()


def test_tray_respects_persisted_preferences_and_notifications(qapp: Any) -> None:
    tray = TrayController(
        qapp,
        lambda _name: None,
        TrackerConfig(show_arrows=False, show_last_seen=False, show_notifications=False),
    )
    assert not tray.toggle_actions["toggle_arrows"].isChecked()
    assert not tray.toggle_actions["toggle_last_seen"].isChecked()
    assert not tray.toggle_actions["toggle_notifications"].isChecked()


def test_region_selection_normalizes_virtual_desktop_coordinates() -> None:
    virtual = QRect(-1920, -200, 3840, 1280)
    region = region_from_selection(QRect(20, 30, 300, 250), virtual)
    assert region == CaptureRegion(top=-170, left=-1900, width=300, height=250)
    assert region_from_selection(QRect(10, 10, 20, 20), virtual) is None


def test_region_selector_emits_selection_and_escape_cancels(qapp: Any) -> None:
    del qapp
    selector = RegionSelector(minimum_size=50, geometry=QRect(0, 0, 400, 300))
    selected: list[CaptureRegion] = []
    cancelled: list[bool] = []
    selector.selected.connect(selected.append)
    selector.cancelled.connect(lambda: cancelled.append(True))
    selector.show()

    QTest.mousePress(selector, Qt.LeftButton, pos=QPoint(40, 50))
    QTest.mouseMove(selector, QPoint(180, 190))
    QTest.mouseRelease(selector, Qt.LeftButton, pos=QPoint(180, 190))
    assert selected == [CaptureRegion(top=50, left=40, width=141, height=141)]
    assert not selector.isVisible()

    selector.show()
    QTest.keyClick(selector, Qt.Key_Escape)
    assert cancelled == [True]
    assert not selector.isVisible()


def test_region_selector_paint_keeps_selected_center_transparent(qapp: Any) -> None:
    selector = RegionSelector(minimum_size=50, geometry=QRect(0, 0, 400, 300))
    selector.show()
    press = QMouseEvent(
        QEvent.MouseButtonPress,
        QPointF(80, 70),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    move = QMouseEvent(
        QEvent.MouseMove,
        QPointF(250, 240),
        Qt.NoButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    qapp.sendEvent(selector, press)
    qapp.sendEvent(selector, move)
    image = QImage(selector.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor(80, 80, 80))
    selector.render(image)
    assert image.pixelColor(150, 150).alpha() == 0
    assert image.pixelColor(80, 70).blue() > 150
    assert image.pixelColor(20, 20).alpha() > 0
    selector.close()


def test_calibration_controller_restores_previous_pause_state(qapp: Any) -> None:
    del qapp
    selector = RegionSelector(minimum_size=50, geometry=QRect(0, 0, 400, 300))
    state = {"paused": False}
    visibility: list[str] = []
    applied: list[CaptureRegion] = []
    pause_updates: list[bool] = []

    def set_paused(value: bool) -> None:
        state["paused"] = value

    controller = CalibrationController(
        selector=selector,
        is_paused=lambda: state["paused"],
        set_paused=set_paused,
        hide_overlay=lambda: visibility.append("hidden"),
        show_overlay=lambda: visibility.append("shown"),
        apply_region=applied.append,
        pause_changed=pause_updates.append,
    )
    controller.start()
    assert state["paused"]
    assert selector.isVisible()
    selected = CaptureRegion(10, 20, 200, 180)
    selector.selected.emit(selected)
    assert applied == [selected]
    assert not state["paused"]
    assert visibility == ["hidden", "shown"]
    assert pause_updates == [True, False]

    state["paused"] = True
    controller.start()
    selector.cancel()
    assert state["paused"]
