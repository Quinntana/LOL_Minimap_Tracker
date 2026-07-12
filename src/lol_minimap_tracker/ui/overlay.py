"""Transparent click-through tracking overlay."""

from __future__ import annotations

import math
from collections.abc import Callable

from PyQt5.QtCore import QRect, Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPen, QShowEvent
from PyQt5.QtWidgets import QApplication, QMainWindow

from ..config import CaptureRegion, TrackerConfig
from ..domain.interfaces import DisplayAffinityController
from ..domain.models import (
    AffinityResult,
    AffinityStatus,
    LastSeenMarkerStyle,
    TrackerSnapshot,
)
from .geometry import marker_dot_offsets, marker_layouts, segment_intersects_rect, status_origin
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
    ) -> None:
        super().__init__()
        self.snapshot_provider = snapshot_provider
        self.config = config
        self.icons = icons
        self.affinity_controller = affinity_controller
        self.affinity_changed = affinity_changed
        self.capture_isolated = capture_isolated or (lambda: False)
        self.capture_region_provider = capture_region_provider
        self.snapshot = TrackerSnapshot()
        self.capture_region = config.capture
        self.show_arrows = config.show_arrows
        self.show_last_seen = config.show_last_seen
        self.last_seen_marker_style = config.last_seen_marker_style
        self.affinity_result = AffinityResult(AffinityStatus.FAILED)
        self._affinity_applied = False
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
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(self._virtual_geometry)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(self.config.update_interval_ms)

    def showEvent(self, event: QShowEvent | None) -> None:
        super().showEvent(event)
        if not self._affinity_applied:
            self.affinity_result = self.affinity_controller.apply(
                int(self.winId()), self.config.exclude_overlay_from_capture
            )
            self._affinity_applied = True
            self.affinity_changed(self.affinity_result)

    def update_overlay(self) -> None:
        if self.capture_region_provider is not None:
            region = self.capture_region_provider()
            if region != self.capture_region:
                self.set_capture_region(region)
        self.snapshot = self.snapshot_provider()
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
        center = self.rect().center()
        map_rect = self._map_rect_local()
        map_tuple = (map_rect.x(), map_rect.y(), map_rect.width(), map_rect.height())
        for champion in self.snapshot.champions:
            if champion.position is None:
                continue
            dx = champion.position[0] - self.snapshot.camera_center[0]
            dy = champion.position[1] - self.snapshot.camera_center[1]
            end = center.x() + dx, center.y() + dy
            start = center.x(), center.y()
            if not self._in_map_graphics_safe() and segment_intersects_rect(start, end, map_tuple):
                continue
            color = QColor(255, 0, 0, 240) if champion.is_current else QColor(255, 255, 0, 225)
            painter.setPen(QPen(color, 2))
            painter.drawLine(start[0], start[1], end[0], end[1])
            painter.setPen(QColor(255, 255, 255, 235))
            painter.drawText(
                end[0] + 5,
                end[1] - 5,
                f"{champion.identity.champion_name} ({int(math.hypot(dx, dy))})",
            )

    def _draw_markers(self, painter: QPainter) -> None:
        if not self.show_last_seen:
            return
        if (
            self.last_seen_marker_style is LastSeenMarkerStyle.RING
            and not self._in_map_graphics_safe()
        ):
            return
        layouts = (
            marker_layouts(self.snapshot.champions)
            if self.last_seen_marker_style is LastSeenMarkerStyle.RING
            else {}
        )
        dot_offsets = (
            marker_dot_offsets(self.snapshot.champions)
            if self.last_seen_marker_style is LastSeenMarkerStyle.DOT
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
            layout = layouts[champion.identity.champion_name]
            radius = layout.radius
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(10, 10, 10, 210), 4))
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
            painter.setPen(QPen(QColor(champion.identity.color), 2))
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)

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
