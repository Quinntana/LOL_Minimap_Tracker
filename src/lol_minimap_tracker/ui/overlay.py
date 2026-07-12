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
    ChampionView,
    LastSeenMarkerStyle,
    TrackerSnapshot,
)
from .champion_portraits import ChampionPortraitRenderer
from .geometry import (
    arrow_range_style,
    marker_dot_offsets,
    marker_icon_offsets,
    segment_intersects_rect,
    status_origin,
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

    def _draw_status(self, painter: QPainter) -> None:
        screen = (
            self._virtual_geometry.left(),
            self._virtual_geometry.top(),
            self._virtual_geometry.width(),
            self._virtual_geometry.height(),
        )
        x_global, y_global = status_origin(
            screen, self._map_rect_global(), len(self.snapshot.champions)
        )
        x = x_global - self._virtual_geometry.left()
        y = y_global - self._virtual_geometry.top()
        for index, champion in enumerate(self.snapshot.champions):
            row_y = y + index * 22
            icon = self.icons.render(champion.identity.role_icon, champion.identity.color, 16)
            painter.drawPixmap(x, row_y + 2, icon)
            painter.setPen(QColor(champion.identity.color))
            name_x = x + 21
            painter.drawText(name_x, row_y + 16, champion.identity.champion_name)
            name_width = painter.fontMetrics().horizontalAdvance(champion.identity.champion_name)
            if champion.is_current:
                status = " - visible"
                status_color = QColor(235, 235, 235, 235)
            elif champion.position is not None and champion.seconds_since_seen is not None:
                status = f" - last seen {champion.seconds_since_seen:.1f}s ago"
                status_color = QColor(215, 215, 215, 230)
            else:
                status = " - not seen"
                status_color = QColor(180, 180, 180, 220)
            painter.setPen(status_color)
            painter.drawText(name_x + name_width, row_y + 16, status)

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
            end = start[0] + dx, start[1] + dy
            if not graphics_safe and segment_intersects_rect(start, end, map_tuple):
                continue
            if champion.is_current:
                (red, green, blue), range_label = arrow_range_style(
                    distance,
                    self.capture_region.width,
                    self.capture_region.height,
                )
                color = QColor(red, green, blue, 240)
                pen = QPen(color, 3)
                label = f"{champion.identity.champion_name} ({range_label})"
                label_color = QColor(245, 247, 250, 235)
            else:
                color = QColor(245, 190, 45, 155)
                pen = QPen(color, 2, Qt.DashLine)
                age = champion.seconds_since_seen or 0.0
                label = f"{champion.identity.champion_name} (last {age:.0f}s)"
                label_color = QColor(245, 220, 150, 190)
            painter.setPen(pen)
            painter.drawLine(start[0], start[1], end[0], end[1])
            if distance >= 5:
                unit_x = dx / distance
                unit_y = dy / distance
                perpendicular_x = -unit_y
                perpendicular_y = unit_x
                head_length = min(distance, 8.0, max(5.0, distance * 0.25))
                head_width = min(distance / 2, 5.0, max(3.0, distance * 0.16))
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
            painter.setPen(label_color)
            painter.drawText(end[0] + 5, end[1] - 5, label)
        if not graphics_safe:
            painter.restore()

    @staticmethod
    def _draw_missing_badge(painter: QPainter, x: int, y: int) -> None:
        badge_x = x + 8
        badge_y = y + 8
        painter.save()
        painter.setPen(QPen(QColor(8, 8, 8, 245), 2))
        painter.setBrush(QColor(20, 20, 20, 235))
        painter.drawEllipse(QRect(badge_x - 5, badge_y - 5, 11, 11))
        painter.setPen(QPen(QColor(235, 55, 55, 245), 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(badge_x - 2, badge_y - 2, badge_x + 2, badge_y + 2)
        painter.drawLine(badge_x + 2, badge_y - 2, badge_x - 2, badge_y + 2)
        painter.restore()

    def _draw_role_marker(self, painter: QPainter, champion: ChampionView, x: int, y: int) -> None:
        painter.save()
        painter.setPen(QPen(QColor(champion.identity.color), 1))
        painter.setBrush(QColor(8, 10, 14, 205))
        painter.drawEllipse(QRect(x - 11, y - 11, 22, 22))
        icon = self.icons.render(
            champion.identity.role_icon,
            champion.identity.color,
            18,
            0.9,
        )
        painter.drawPixmap(x - 9, y - 9, icon)
        painter.restore()
        self._draw_missing_badge(painter, x, y)

    def _draw_portrait_marker(
        self, painter: QPainter, champion: ChampionView, x: int, y: int
    ) -> None:
        portrait = self.portraits.get(champion.identity.champion_name)
        if portrait is None:
            self._draw_role_marker(painter, champion, x, y)
            return
        pixmap = self.portrait_renderer.render(champion.identity.champion_name, portrait, 24)
        if pixmap.isNull():
            self._draw_role_marker(painter, champion, x, y)
            return
        painter.save()
        painter.setOpacity(0.62)
        painter.drawPixmap(x - 12, y - 12, pixmap)
        painter.restore()
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(champion.identity.color), 1))
        painter.drawEllipse(QRect(x - 12, y - 12, 24, 24))
        self._draw_missing_badge(painter, x, y)

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
                painter.save()
                painter.setRenderHint(QPainter.Antialiasing, False)
                painter.setPen(QPen(QColor(8, 8, 8, 230), 1))
                painter.setBrush(QColor(champion.identity.color))
                painter.drawEllipse(QRect(x + dot_x - 2, y + dot_y - 2, 5, 5))
                painter.restore()
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
        if self._in_map_graphics_safe():
            painter.setPen(QPen(QColor(255, 0, 0, 180), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self._map_rect_local())
        self._draw_status(painter)
        self._draw_arrows(painter)
        self._draw_markers(painter)
