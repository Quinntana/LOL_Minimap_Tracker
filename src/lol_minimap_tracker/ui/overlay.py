"""Transparent click-through tracking overlay."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

from PyQt5.QtCore import QPoint, QRect, Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPen, QRegion, QShowEvent
from PyQt5.QtWidgets import QApplication, QMainWindow

from ..config import CaptureRegion, TrackerConfig
from ..domain.interfaces import DisplayAffinityController, Image, OverlayInputController
from ..domain.models import (
    AffinityResult,
    AffinityStatus,
    ArrowDisplayMode,
    ChampionView,
    LastSeenMarkerStyle,
    TrackerSnapshot,
)
from .champion_portraits import ChampionPortraitRenderer
from .geometry import (
    arrow_display_length,
    arrow_range_color,
    marker_dot_offsets,
    marker_icon_offsets,
    normalized_map_distance,
    segment_intersects_rect,
)
from .role_icons import RoleIconRenderer


class TransparentOverlay(QMainWindow):
    def __init__(
        self,
        snapshot_provider: Callable[[], TrackerSnapshot],
        config: TrackerConfig,
        icons: RoleIconRenderer,
        affinity_controller: DisplayAffinityController,
        affinity_changed: Callable[[AffinityResult], None],
        capture_isolated: Callable[[], bool] | None = None,
        capture_region_provider: Callable[[], CaptureRegion] | None = None,
        portrait_provider: Callable[[], Mapping[str, Image]] | None = None,
        arrow_origin_provider: Callable[[], tuple[int, int] | None] | None = None,
        input_controller: OverlayInputController | None = None,
        input_changed: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__()
        self.snapshot_provider = snapshot_provider
        self.config = config
        self.icons = icons
        self.affinity_controller = affinity_controller
        self.affinity_changed = affinity_changed
        self.capture_isolated = capture_isolated or (lambda: False)
        self.capture_region_provider = capture_region_provider
        self.portrait_provider = portrait_provider or (lambda: {})
        self.arrow_origin_provider = arrow_origin_provider or (lambda: None)
        self.input_controller = input_controller
        self.input_changed = input_changed or (lambda _active: None)
        self.portrait_renderer = ChampionPortraitRenderer()
        self.portraits: Mapping[str, Image] = {}
        self.snapshot = TrackerSnapshot()
        self.capture_region = config.capture
        self.show_arrows = config.show_arrows
        self.arrow_display_mode = config.arrow_display_mode
        self.arrow_nearby_range_ratio = config.arrow_nearby_range_ratio
        self.show_last_seen = config.show_last_seen
        self.last_seen_marker_style = config.last_seen_marker_style
        self.affinity_result = AffinityResult(AffinityStatus.FAILED)
        self._affinity_applied = False
        self._input_style_applied = False
        self._input_style_attempts = 0
        self._input_failure_reported = False
        self._virtual_geometry = self._get_virtual_geometry()
        self._init_ui()

    @staticmethod
    def _get_virtual_geometry() -> QRect:
        screens = QApplication.screens()
        geometry = QRect(screens[0].geometry())
        for screen in screens[1:]:
            geometry = geometry.united(screen.geometry())
        return geometry

    def _init_ui(self) -> None:
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.NoFocus)
        self.setGeometry(self._virtual_geometry)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(self.config.update_interval_ms)

    def showEvent(self, event: QShowEvent | None) -> None:
        super().showEvent(event)
        if not self._input_style_applied and self.input_controller is not None:
            self._apply_input_style()
        if not self._affinity_applied:
            self.affinity_result = self.affinity_controller.apply(
                int(self.winId()), self.config.exclude_overlay_from_capture
            )
            self._affinity_applied = True
            self.affinity_changed(self.affinity_result)

    def _apply_input_style(self) -> None:
        if self._input_style_applied or self.input_controller is None:
            return
        self._input_style_attempts += 1
        if self.input_controller.apply(int(self.winId())):
            self._input_style_applied = True
            self.input_changed(True)
            return
        if self._input_style_attempts < 2:
            QTimer.singleShot(50, self._apply_input_style)
            return
        if not self._input_failure_reported:
            self._input_failure_reported = True
            self.input_changed(False)
        # Qt's transparent-input flags normally remain effective, but if the
        # native style cannot be verified the safe behavior is no overlay.
        self.hide()

    def update_overlay(self) -> None:
        if self.capture_region_provider is not None:
            region = self.capture_region_provider()
            if region != self.capture_region:
                self.set_capture_region(region)
        self.snapshot = self.snapshot_provider()
        self.portraits = dict(self.portrait_provider())
        self.update()

    def toggle_arrows(self) -> bool:
        self.show_arrows = not self.show_arrows
        self.update()
        return self.show_arrows

    def toggle_last_seen(self) -> bool:
        self.show_last_seen = not self.show_last_seen
        self.update()
        return self.show_last_seen

    def set_arrow_display_mode(self, mode: ArrowDisplayMode) -> None:
        self.arrow_display_mode = mode
        self.update()

    def set_last_seen_marker_style(self, style: LastSeenMarkerStyle) -> None:
        self.last_seen_marker_style = style
        self.update()

    def set_capture_region(self, region: CaptureRegion) -> None:
        geometry = self._get_virtual_geometry()
        if geometry != self._virtual_geometry:
            self._virtual_geometry = geometry
            self.setGeometry(geometry)
        self.capture_region = region
        self.update()

    def _map_rect_global(self) -> tuple[int, int, int, int]:
        region = self.capture_region
        return region.left, region.top, region.width, region.height

    def _map_rect_local(self) -> QRect:
        region = self.capture_region
        return QRect(
            region.left - self._virtual_geometry.left(),
            region.top - self._virtual_geometry.top(),
            region.width,
            region.height,
        )

    def _in_map_graphics_safe(self) -> bool:
        return self.capture_isolated() or self.affinity_result.status is AffinityStatus.ACTIVE

    def _arrow_origin_local(self) -> tuple[int, int] | None:
        provided = self.arrow_origin_provider()
        if provided is not None:
            return (
                provided[0] - self._virtual_geometry.left(),
                provided[1] - self._virtual_geometry.top(),
            )
        # A League-window capture can briefly lose its HWND/geometry while WGC
        # restarts.  Hiding arrows avoids making them jump to an unrelated
        # monitor center while the last detection snapshot is still visible.
        if self.capture_isolated():
            return None
        region = self.capture_region
        map_center = QPoint(
            region.left + region.width // 2,
            region.top + region.height // 2,
        )
        screen = next(
            (item for item in QApplication.screens() if item.geometry().contains(map_center)),
            None,
        )
        center = self._virtual_geometry.center() if screen is None else screen.geometry().center()
        return (
            center.x() - self._virtual_geometry.left(),
            center.y() - self._virtual_geometry.top(),
        )

    def _draw_arrows(self, painter: QPainter) -> None:
        if not self.show_arrows or self.snapshot.camera_center is None:
            return
        start = self._arrow_origin_local()
        if start is None:
            return
        map_rect = self._map_rect_local()
        map_tuple = (map_rect.x(), map_rect.y(), map_rect.width(), map_rect.height())
        graphics_safe = self._in_map_graphics_safe()
        if not graphics_safe:
            painter.save()
            painter.setClipRegion(QRegion(self.rect()).subtracted(QRegion(map_rect)))
        for champion in self.snapshot.champions:
            if champion.position is None:
                continue
            dx = champion.position[0] - self.snapshot.camera_center[0]
            dy = champion.position[1] - self.snapshot.camera_center[1]
            distance = math.hypot(dx, dy)
            if distance < 1.0:
                continue
            if (
                self.arrow_display_mode is ArrowDisplayMode.NEARBY
                and normalized_map_distance(
                    distance,
                    self.capture_region.width,
                    self.capture_region.height,
                )
                > self.arrow_nearby_range_ratio
            ):
                continue
            unit_x = dx / distance
            unit_y = dy / distance
            display_length = arrow_display_length(
                distance,
                self.capture_region.width,
                self.capture_region.height,
            )
            end = (
                round(start[0] + unit_x * display_length),
                round(start[1] + unit_y * display_length),
            )
            if not graphics_safe and segment_intersects_rect(start, end, map_tuple):
                continue
            red, green, blue = arrow_range_color(
                distance,
                self.capture_region.width,
                self.capture_region.height,
            )
            color = QColor(red, green, blue, 240)
            line_style = Qt.SolidLine if champion.is_current else Qt.DashLine
            pen = QPen(color)
            pen.setWidthF(1.4)
            pen.setStyle(line_style)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(start[0], start[1], end[0], end[1])
            if display_length >= 8:
                perpendicular_x = -unit_y
                perpendicular_y = unit_x
                head_length = min(7.0, max(4.0, display_length * 0.12))
                head_width = min(3.5, max(2.5, display_length * 0.06))
                base_x = end[0] - unit_x * head_length
                base_y = end[1] - unit_y * head_length
                painter.drawLine(
                    round(end[0]),
                    round(end[1]),
                    round(base_x + perpendicular_x * head_width),
                    round(base_y + perpendicular_y * head_width),
                )
                painter.drawLine(
                    round(end[0]),
                    round(end[1]),
                    round(base_x - perpendicular_x * head_width),
                    round(base_y - perpendicular_y * head_width),
                )
        if not graphics_safe:
            painter.restore()

    @staticmethod
    def _draw_dot_marker(painter: QPainter, champion: ChampionView, x: int, y: int) -> None:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(QPen(QColor(8, 8, 8, 230), 1))
        painter.setBrush(QColor(champion.identity.color))
        painter.drawEllipse(QRect(x - 2, y - 2, 5, 5))
        painter.restore()

    def _draw_role_marker(self, painter: QPainter, champion: ChampionView, x: int, y: int) -> None:
        halo = self.icons.render(
            champion.identity.role_icon,
            "#080A0E",
            20,
            0.78,
        )
        icon = self.icons.render(
            champion.identity.role_icon,
            champion.identity.color,
            16,
            0.88,
        )
        painter.drawPixmap(x - 10, y - 10, halo)
        painter.drawPixmap(x - 8, y - 8, icon)

    def _draw_portrait_marker(
        self, painter: QPainter, champion: ChampionView, x: int, y: int
    ) -> None:
        portrait = self.portraits.get(champion.identity.champion_name)
        if portrait is None:
            return
        pixmap = self.portrait_renderer.render(champion.identity.champion_name, portrait, 20)
        if pixmap.isNull():
            return
        painter.save()
        painter.setOpacity(0.68)
        painter.drawPixmap(x - 10, y - 10, pixmap)
        painter.restore()
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(8, 8, 8, 225), 3))
        painter.drawEllipse(QRect(x - 10, y - 10, 20, 20))
        missing_pen = QPen(QColor(248, 113, 113, 245))
        missing_pen.setWidthF(1.4)
        missing_pen.setStyle(Qt.DashLine)
        painter.setPen(missing_pen)
        painter.drawEllipse(QRect(x - 10, y - 10, 20, 20))

    def _draw_markers(self, painter: QPainter) -> None:
        if not self.show_last_seen:
            return
        if (
            self.last_seen_marker_style is not LastSeenMarkerStyle.DOT
            and not self._in_map_graphics_safe()
        ):
            return
        dot_offsets = (
            marker_dot_offsets(self.snapshot.champions)
            if self.last_seen_marker_style is LastSeenMarkerStyle.DOT
            else {}
        )
        icon_offsets = (
            marker_icon_offsets(self.snapshot.champions)
            if self.last_seen_marker_style is not LastSeenMarkerStyle.DOT
            else {}
        )
        region = self.capture_region
        offset_x = region.left - self._virtual_geometry.left()
        offset_y = region.top - self._virtual_geometry.top()
        for champion in self.snapshot.champions:
            if champion.is_current or champion.position is None:
                continue
            x = offset_x + champion.position[0]
            y = offset_y + champion.position[1]
            if self.last_seen_marker_style is LastSeenMarkerStyle.DOT:
                dot_x, dot_y = dot_offsets[champion.identity.champion_name]
                self._draw_dot_marker(painter, champion, x + dot_x, y + dot_y)
                continue
            icon_x, icon_y = icon_offsets[champion.identity.champion_name]
            x += icon_x
            y += icon_y
            if self.last_seen_marker_style is LastSeenMarkerStyle.ROLE:
                self._draw_role_marker(painter, champion, x, y)
            else:
                self._draw_portrait_marker(painter, champion, x, y)

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_arrows(painter)
        self._draw_markers(painter)
