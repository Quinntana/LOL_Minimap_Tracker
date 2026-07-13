from __future__ import annotations

from typing import Any

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PyQt5.QtGui import QColor, QImage, QMouseEvent, QPen
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QSystemTrayIcon

from lol_minimap_tracker.config import CaptureRegion, TrackerConfig
from lol_minimap_tracker.domain.models import (
    AffinityResult,
    AffinityStatus,
    ChampionView,
    EnemyIdentity,
    LastSeenMarkerStyle,
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
    square_selection_rect,
)
from lol_minimap_tracker.ui.champion_portraits import ChampionPortraitRenderer
from lol_minimap_tracker.ui.overlay import TransparentOverlay
from lol_minimap_tracker.ui.role_icons import RoleIconRenderer
from lol_minimap_tracker.ui.tray import TrayController


class Affinity:
    def apply(self, _handle: int, _enabled: bool) -> AffinityResult:
        return AffinityResult(AffinityStatus.ACTIVE)


class InputController:
    def __init__(self) -> None:
        self.handles: list[int] = []

    def apply(self, handle: int) -> bool:
        self.handles.append(handle)
        return True


class SequencedInputController:
    def __init__(self, outcomes: list[bool]) -> None:
        self.outcomes = outcomes
        self.handles: list[int] = []

    def apply(self, handle: int) -> bool:
        self.handles.append(handle)
        return self.outcomes.pop(0) if self.outcomes else False


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


def test_champion_portraits_render_bgr_as_cached_circles(qapp: object) -> None:
    del qapp
    portrait = np.full((20, 20, 3), (220, 30, 10), dtype=np.uint8)
    renderer = ChampionPortraitRenderer()
    first = renderer.render("Aatrox", portrait, 18)
    second = renderer.render("Aatrox", portrait, 18)
    image = first.toImage()
    center = image.pixelColor(9, 9)
    assert not first.isNull()
    assert first.cacheKey() == second.cacheKey()
    assert center.blue() > center.red()
    assert image.pixelColor(0, 0).alpha() == 0


def test_overlay_renders_safe_and_fallback_modes(qapp: Any) -> None:
    current = snapshot()
    changed: list[AffinityResult] = []
    input_changed: list[bool] = []
    input_controller = InputController()
    isolated = {"value": False}
    region = CaptureRegion(200, 300, 100, 100)
    portraits = {
        "Nidalee": np.full((24, 24, 3), (220, 30, 10), dtype=np.uint8),
    }
    overlay = TransparentOverlay(
        snapshot_provider=lambda: current,
        config=TrackerConfig(capture=region),
        icons=RoleIconRenderer(AppPaths.discover().role_asset_dir),
        affinity_controller=Affinity(),
        affinity_changed=changed.append,
        capture_isolated=lambda: isolated["value"],
        portrait_provider=lambda: portraits,
        input_controller=input_controller,
        input_changed=input_changed.append,
    )
    overlay.snapshot = current
    overlay.portraits = portraits
    overlay.show()
    qapp.processEvents()
    assert changed[-1].status is AffinityStatus.ACTIVE
    assert input_changed == [True]
    assert input_controller.handles == [int(overlay.winId())]
    assert overlay.windowFlags() & Qt.WindowTransparentForInput
    assert overlay.windowFlags() & Qt.WindowDoesNotAcceptFocus
    assert overlay.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert overlay.testAttribute(Qt.WA_ShowWithoutActivating)
    assert overlay.focusPolicy() == Qt.NoFocus
    overlay.affinity_result = AffinityResult(AffinityStatus.ACTIVE)
    overlay.show_arrows = False
    image = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(0)
    overlay.render(image)
    assert not image.isNull()
    marker_x = region.left - overlay._virtual_geometry.left() + 70
    marker_y = region.top - overlay._virtual_geometry.top() + 80
    current_x = region.left - overlay._virtual_geometry.left() + 30
    current_y = region.top - overlay._virtual_geometry.top() + 40
    missing_cross = image.pixelColor(marker_x, marker_y)
    faded_portrait = image.pixelColor(marker_x, marker_y - 5)
    assert overlay.last_seen_marker_style is LastSeenMarkerStyle.PORTRAIT
    assert missing_cross.red() > missing_cross.green()
    assert faded_portrait.blue() > faded_portrait.red()
    assert image.pixelColor(marker_x + 12, marker_y + 12).alpha() == 0
    assert image.pixelColor(current_x, current_y).alpha() == 0

    overlay.affinity_result = AffinityResult(AffinityStatus.DISABLED)
    unsafe = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    unsafe.fill(0)
    overlay.render(unsafe)
    assert unsafe.pixelColor(marker_x, marker_y).alpha() == 0

    overlay.set_last_seen_marker_style(LastSeenMarkerStyle.ROLE)
    unsafe_role = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    unsafe_role.fill(0)
    overlay.render(unsafe_role)
    assert unsafe_role.pixelColor(marker_x, marker_y).alpha() == 0

    isolated["value"] = True
    isolated_role = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    isolated_role.fill(0)
    overlay.render(isolated_role)
    role_pixels = [
        isolated_role.pixelColor(x, y)
        for x in range(marker_x - 10, marker_x + 11)
        for y in range(marker_y - 10, marker_y + 11)
    ]
    assert any(pixel.alpha() > 0 for pixel in role_pixels)
    assert not any(
        pixel.red() > pixel.green() + 40 and pixel.red() > pixel.blue() + 40
        for pixel in role_pixels
    )
    assert isolated_role.pixelColor(marker_x + 12, marker_y + 12).alpha() == 0

    overlay.set_last_seen_marker_style(LastSeenMarkerStyle.PORTRAIT)
    overlay.portraits = {}
    portrait_fallback = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    portrait_fallback.fill(0)
    overlay.render(portrait_fallback)
    assert portrait_fallback.pixelColor(marker_x, marker_y).alpha() > 0
    assert portrait_fallback.pixelColor(marker_x + 4, marker_y).alpha() == 0

    isolated["value"] = False
    overlay.set_last_seen_marker_style(LastSeenMarkerStyle.DOT)
    overlay.affinity_result = AffinityResult(AffinityStatus.FAILED)
    dot_fallback = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    dot_fallback.fill(0)
    overlay.render(dot_fallback)
    dot = dot_fallback.pixelColor(marker_x, marker_y)
    assert dot.alpha() > 0
    assert dot.green() > dot.red()
    assert dot_fallback.pixelColor(marker_x + 4, marker_y).alpha() == 0

    assert overlay.last_seen_marker_style is LastSeenMarkerStyle.DOT
    failed_affinity = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    failed_affinity.fill(0)
    overlay.render(failed_affinity)
    assert failed_affinity.pixelColor(marker_x, marker_y).alpha() > 0

    top = current.champions[0].identity
    jungle = current.champions[1].identity
    overlay.snapshot = TrackerSnapshot(
        champions=(
            ChampionView(top, (70, 80), False, 8.0),
            ChampionView(jungle, (70, 80), False, 5.0),
        )
    )
    clustered = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    clustered.fill(0)
    overlay.render(clustered)
    left_dot = clustered.pixelColor(marker_x - 3, marker_y)
    right_dot = clustered.pixelColor(marker_x + 3, marker_y)
    assert left_dot.red() > left_dot.green()
    assert right_dot.green() > right_dot.red()

    overlay.snapshot = current
    overlay.show_arrows = True
    assert not overlay.toggle_arrows()
    assert not overlay.toggle_last_seen()
    overlay.set_last_seen_marker_style(LastSeenMarkerStyle.DOT)
    hidden = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    hidden.fill(0)
    overlay.render(hidden)
    assert hidden.pixelColor(marker_x, marker_y).alpha() == 0
    overlay.update_overlay()
    overlay.set_capture_region(CaptureRegion(-100, -200, 120, 110))
    assert overlay.capture_region == CaptureRegion(-100, -200, 120, 110)

    overlay.affinity_result = AffinityResult(AffinityStatus.FAILED)
    fallback = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    fallback.fill(0)
    overlay.render(fallback)
    overlay.close()


def test_overlay_arrows_use_range_colors_and_game_window_origin(qapp: Any) -> None:
    class RecordingPainter:
        def __init__(self) -> None:
            self.current_pen = QPen()
            self.line_pens: list[QPen] = []
            self.labels: list[str] = []

        def setPen(self, pen: QPen | QColor) -> None:
            self.current_pen = QPen(pen)

        def drawLine(self, *_coordinates: object) -> None:
            self.line_pens.append(QPen(self.current_pen))

        def drawText(self, *_arguments: object) -> None:
            self.labels.append(str(_arguments[-1]))

    identity = EnemyIdentity("Aatrox", Role.TOP, "#E69F00", "position-top.svg")
    state = {"snapshot": TrackerSnapshot(), "origin": (0, 0)}
    region = CaptureRegion(200, 300, 100, 100)
    overlay = TransparentOverlay(
        snapshot_provider=lambda: state["snapshot"],
        config=TrackerConfig(capture=region, show_last_seen=False),
        icons=RoleIconRenderer(AppPaths.discover().role_asset_dir),
        affinity_controller=Affinity(),
        affinity_changed=lambda _result: None,
        arrow_origin_provider=lambda: state["origin"],
    )
    origin_local = (overlay.width() // 3, overlay.height() // 3)
    state["origin"] = (
        origin_local[0] + overlay._virtual_geometry.left(),
        origin_local[1] + overlay._virtual_geometry.top(),
    )
    overlay.show()
    qapp.processEvents()
    overlay.affinity_result = AffinityResult(AffinityStatus.ACTIVE)

    state["snapshot"] = TrackerSnapshot(
        champions=(ChampionView(identity, (65, 50), True, 0.1),),
        camera_center=(50, 50),
    )
    overlay.snapshot = state["snapshot"]
    current_painter = RecordingPainter()
    overlay._draw_arrows(current_painter)  # type: ignore[arg-type]
    assert current_painter.line_pens[0].style() == Qt.SolidLine
    assert current_painter.labels == ["Aatrox"]
    close_image = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    close_image.fill(0)
    overlay.render(close_image)
    close_pixel = close_image.pixelColor(origin_local[0] + 7, origin_local[1])
    assert close_pixel.red() > close_pixel.green()

    state["snapshot"] = TrackerSnapshot(
        champions=(ChampionView(identity, (90, 90), True, 0.1),),
        camera_center=(50, 50),
    )
    overlay.snapshot = state["snapshot"]
    far_painter = RecordingPainter()
    overlay._draw_arrows(far_painter)  # type: ignore[arg-type]
    assert far_painter.line_pens[0].color().green() > far_painter.line_pens[0].color().red()
    far_image = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    far_image.fill(0)
    overlay.render(far_image)
    far_pixel = far_image.pixelColor(origin_local[0] + 20, origin_local[1] + 20)
    assert far_pixel.green() > far_pixel.red()

    state["snapshot"] = TrackerSnapshot(
        champions=(ChampionView(identity, (65, 50), False, 8.0),),
        camera_center=(50, 50),
    )
    overlay.snapshot = state["snapshot"]
    stale_painter = RecordingPainter()
    overlay._draw_arrows(stale_painter)  # type: ignore[arg-type]
    assert stale_painter.line_pens[0].style() == Qt.DashLine
    assert stale_painter.line_pens[0].color().red() > stale_painter.line_pens[0].color().green()
    assert stale_painter.labels == ["Aatrox"]
    stale_image = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    stale_image.fill(0)
    overlay.render(stale_image)
    stale_pixels = [
        stale_image.pixelColor(origin_local[0] + offset, origin_local[1]) for offset in range(1, 14)
    ]
    assert any(pixel.red() > pixel.green() for pixel in stale_pixels)
    assert origin_local != (overlay.rect().center().x(), overlay.rect().center().y())

    overlay.capture_isolated = lambda: True
    state["origin"] = None
    assert overlay._arrow_origin_local() is None
    no_origin_image = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    no_origin_image.fill(0)
    overlay.render(no_origin_image)
    assert no_origin_image.pixelColor(origin_local[0] + 7, origin_local[1]).alpha() == 0
    overlay.close()


def test_unsafe_arrow_labels_and_heads_are_clipped_out_of_minimap(qapp: Any) -> None:
    del qapp
    identity = EnemyIdentity("Aatrox", Role.TOP, "#E69F00", "position-top.svg")
    origin_local = (100, 100)
    virtual = TransparentOverlay._get_virtual_geometry()
    region = CaptureRegion(
        top=virtual.top() + origin_local[1] - 20,
        left=virtual.left() + origin_local[0] + 20,
        width=100,
        height=100,
    )
    overlay = TransparentOverlay(
        snapshot_provider=TrackerSnapshot,
        config=TrackerConfig(capture=region, show_last_seen=False),
        icons=RoleIconRenderer(AppPaths.discover().role_asset_dir),
        affinity_controller=Affinity(),
        affinity_changed=lambda _result: None,
        arrow_origin_provider=lambda: (
            virtual.left() + origin_local[0],
            virtual.top() + origin_local[1],
        ),
    )
    overlay.affinity_result = AffinityResult(AffinityStatus.FAILED)
    overlay.snapshot = TrackerSnapshot(
        champions=(ChampionView(identity, (68, 50), True, 0.1),),
        camera_center=(50, 50),
    )
    image = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(0)
    overlay.render(image)
    map_rect = overlay._map_rect_local()
    assert all(
        image.pixelColor(x, y).alpha() == 0
        for y in range(map_rect.top(), map_rect.bottom() + 1)
        for x in range(map_rect.left(), map_rect.right() + 1)
    )
    overlay.close()


def test_overlay_retries_clickthrough_then_keeps_verified_overlay(qapp: Any) -> None:
    controller = SequencedInputController([False, True])
    changed: list[bool] = []
    overlay = TransparentOverlay(
        snapshot_provider=TrackerSnapshot,
        config=TrackerConfig(),
        icons=RoleIconRenderer(AppPaths.discover().role_asset_dir),
        affinity_controller=Affinity(),
        affinity_changed=lambda _result: None,
        input_controller=controller,
        input_changed=changed.append,
    )
    overlay.show()
    QTest.qWait(75)
    qapp.processEvents()
    assert len(controller.handles) == 2
    assert changed == [True]
    assert overlay.isVisible()
    overlay.close()


def test_overlay_hides_after_clickthrough_retry_fails(qapp: Any) -> None:
    controller = SequencedInputController([False, False])
    changed: list[bool] = []
    overlay = TransparentOverlay(
        snapshot_provider=TrackerSnapshot,
        config=TrackerConfig(),
        icons=RoleIconRenderer(AppPaths.discover().role_asset_dir),
        affinity_controller=Affinity(),
        affinity_changed=lambda _result: None,
        input_controller=controller,
        input_changed=changed.append,
    )
    overlay.show()
    QTest.qWait(75)
    qapp.processEvents()
    assert len(controller.handles) == 2
    assert changed == [False]
    assert not overlay.isVisible()
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
    tray.update_affinity(AffinityResult(AffinityStatus.FAILED))
    assert "dot fallback available" in tray.affinity_action.text()
    tray.update_affinity(AffinityResult(AffinityStatus.FAILED), capture_isolated=True)
    assert tray.affinity_action.text() == "Overlay capture: isolated (League window)"
    assert tray.marker_style_actions[LastSeenMarkerStyle.PORTRAIT].isChecked()
    assert not tray.marker_style_actions[LastSeenMarkerStyle.ROLE].isChecked()
    assert not tray.marker_style_actions[LastSeenMarkerStyle.DOT].isChecked()
    assert "red X" not in tray.marker_style_actions[LastSeenMarkerStyle.ROLE].toolTip()
    tray.marker_style_actions[LastSeenMarkerStyle.ROLE].trigger()
    assert dispatched[-1] == "set_marker_style_role"
    assert tray.marker_style_actions[LastSeenMarkerStyle.ROLE].isChecked()
    tray.marker_style_actions[LastSeenMarkerStyle.DOT].trigger()
    assert dispatched[-1] == "set_marker_style_dot"
    assert tray.marker_style_actions[LastSeenMarkerStyle.DOT].isChecked()
    assert not tray.marker_style_actions[LastSeenMarkerStyle.PORTRAIT].isChecked()
    assert not tray.marker_style_actions[LastSeenMarkerStyle.ROLE].isChecked()
    tray.update_marker_style(LastSeenMarkerStyle.DOT)
    assert tray.marker_style_menu.title() == "Missing marker: Minimal dot"
    top_level = [action.text() for action in tray.menu.actions()]
    assert "Select minimap area..." in top_level
    assert "Direction arrows" in top_level
    assert "Missing-enemy markers" in top_level
    assert "Missing marker: Minimal dot" in top_level
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
        TrackerConfig(
            show_arrows=False,
            show_last_seen=False,
            last_seen_marker_style=LastSeenMarkerStyle.DOT,
            show_notifications=False,
        ),
    )
    assert not tray.toggle_actions["toggle_arrows"].isChecked()
    assert not tray.toggle_actions["toggle_last_seen"].isChecked()
    assert tray.marker_style_actions[LastSeenMarkerStyle.DOT].isChecked()
    assert not tray.marker_style_actions[LastSeenMarkerStyle.PORTRAIT].isChecked()
    assert not tray.marker_style_actions[LastSeenMarkerStyle.ROLE].isChecked()
    assert not tray.toggle_actions["toggle_notifications"].isChecked()


def test_region_selection_normalizes_virtual_desktop_coordinates() -> None:
    virtual = QRect(-1920, -200, 3840, 1280)
    region = region_from_selection(QRect(20, 30, 300, 250), virtual)
    assert region == CaptureRegion(top=-170, left=-1900, width=300, height=250)
    assert region_from_selection(QRect(10, 10, 20, 20), virtual) is None


def test_square_selection_uses_dominant_axis_in_every_quadrant() -> None:
    bounds = QRect(0, 0, 240, 200)
    start = QPoint(120, 100)
    for current in (
        QPoint(180, 120),
        QPoint(180, 80),
        QPoint(60, 120),
        QPoint(60, 80),
    ):
        selection = square_selection_rect(start, current, bounds)
        assert selection.width() == 61
        assert selection.height() == 61
        assert selection.contains(start)
        assert selection.contains(current)


def test_square_selection_expands_zero_axis_away_from_screen_edge() -> None:
    bounds = QRect(0, 0, 100, 100)
    horizontal = square_selection_rect(QPoint(20, 99), QPoint(70, 99), bounds)
    vertical = square_selection_rect(QPoint(99, 20), QPoint(99, 70), bounds)
    assert horizontal == QRect(20, 49, 51, 51)
    assert vertical == QRect(49, 20, 51, 51)


def test_square_selection_translates_instead_of_collapsing_on_edge_jitter() -> None:
    bounds = QRect(0, 0, 100, 100)
    selection = square_selection_rect(QPoint(20, 98), QPoint(70, 99), bounds)
    assert selection == QRect(20, 49, 51, 51)
    assert selection.contains(QPoint(20, 98))
    assert selection.contains(QPoint(70, 99))


def test_square_selection_clamps_nonzero_drags_at_each_corner() -> None:
    bounds = QRect(0, 0, 100, 100)
    cases = (
        (QPoint(1, 1), QPoint(41, 0)),
        (QPoint(98, 1), QPoint(58, 0)),
        (QPoint(1, 98), QPoint(41, 99)),
        (QPoint(98, 98), QPoint(58, 99)),
    )
    for start, current in cases:
        selection = square_selection_rect(start, current, bounds)
        assert selection.width() == selection.height()
        assert bounds.contains(selection)
        assert selection.contains(start)
        assert selection.contains(current)


def test_square_selection_preserves_negative_virtual_desktop_translation() -> None:
    virtual = QRect(-1920, -200, 3840, 1280)
    selection = square_selection_rect(
        QPoint(20, 30),
        QPoint(319, 279),
        QRect(QPoint(0, 0), virtual.size()),
    )
    region = region_from_selection(selection, virtual)
    assert region == CaptureRegion(top=-170, left=-1900, width=300, height=300)


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
    QTest.mousePress(selector, Qt.LeftButton, pos=QPoint(40, 50))
    QTest.mouseMove(selector, QPoint(200, 100))
    QTest.mouseRelease(selector, Qt.LeftButton, pos=QPoint(200, 100))
    assert selected[-1] == CaptureRegion(top=50, left=40, width=161, height=161)

    selector.show()
    QTest.keyClick(selector, Qt.Key_Escape)
    assert cancelled == [True]
    assert not selector.isVisible()


def test_region_selector_rejects_square_below_minimum_size(qapp: Any) -> None:
    del qapp
    selector = RegionSelector(minimum_size=50, geometry=QRect(0, 0, 400, 300))
    selected: list[CaptureRegion] = []
    selector.selected.connect(selected.append)
    selector.show()

    QTest.mousePress(selector, Qt.LeftButton, pos=QPoint(390, 290))
    QTest.mouseMove(selector, QPoint(399, 299))
    QTest.mouseRelease(selector, Qt.LeftButton, pos=QPoint(399, 299))

    assert selected == []
    assert selector.isVisible()
    assert selector.selection_rect().isEmpty()
    selector.close()


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
