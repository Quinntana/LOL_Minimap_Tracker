"""Interactive virtual-desktop minimap region selection."""

from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget

from ..config import CaptureRegion


def virtual_desktop_geometry() -> QRect:
    screens = QApplication.screens()
    if not screens:
        return QRect(0, 0, 1, 1)
    geometry = QRect(screens[0].geometry())
    for screen in screens[1:]:
        geometry = geometry.united(screen.geometry())
    return geometry


def region_from_selection(
    selection: QRect, virtual_geometry: QRect, minimum_size: int = 64
) -> CaptureRegion | None:
    normalized = selection.normalized().intersected(QRect(QPoint(0, 0), virtual_geometry.size()))
    if normalized.width() < minimum_size or normalized.height() < minimum_size:
        return None
    return CaptureRegion(
        top=virtual_geometry.top() + normalized.top(),
        left=virtual_geometry.left() + normalized.left(),
        width=normalized.width(),
        height=normalized.height(),
    )


def square_selection_rect(start: QPoint, current: QPoint, bounds: QRect) -> QRect:
    """Build a sign-aware, in-bounds square using inclusive QRect coordinates."""
    if bounds.isEmpty() or not bounds.contains(start):
        return QRect()

    dx = current.x() - start.x()
    dy = current.y() - start.y()
    left_capacity = start.x() - bounds.left()
    right_capacity = bounds.right() - start.x()
    top_capacity = start.y() - bounds.top()
    bottom_capacity = bounds.bottom() - start.y()

    x_direction = 1 if dx > 0 else -1 if dx < 0 else (1 if right_capacity >= left_capacity else -1)
    y_direction = 1 if dy > 0 else -1 if dy < 0 else (1 if bottom_capacity >= top_capacity else -1)
    side = min(max(abs(dx), abs(dy)), bounds.width() - 1, bounds.height() - 1)
    ideal_left = start.x() if x_direction > 0 else start.x() - side
    ideal_top = start.y() if y_direction > 0 else start.y() - side
    left = max(bounds.left(), min(ideal_left, bounds.right() - side))
    top = max(bounds.top(), min(ideal_top, bounds.bottom() - side))
    return QRect(left, top, side + 1, side + 1)


class RegionSelector(QWidget):
    selected = pyqtSignal(object)
    cancelled = pyqtSignal()

    def __init__(self, minimum_size: int = 64, geometry: QRect | None = None) -> None:
        super().__init__()
        self.minimum_size = minimum_size
        self._virtual_geometry = QRect(geometry or virtual_desktop_geometry())
        self._start: QPoint | None = None
        self._current: QPoint | None = None
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setGeometry(self._virtual_geometry)

    def begin(self) -> None:
        self._virtual_geometry = virtual_desktop_geometry()
        self.setGeometry(self._virtual_geometry)
        self._start = None
        self._current = None
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.ActiveWindowFocusReason)

    def selection_rect(self) -> QRect:
        if self._start is None or self._current is None:
            return QRect()
        return square_selection_rect(self._start, self._current, self.rect())

    def cancel(self) -> None:
        self.hide()
        self._start = None
        self._current = None
        self.cancelled.emit()

    def _bounded(self, point: QPoint) -> QPoint:
        return QPoint(
            max(0, min(point.x(), self.width() - 1)),
            max(0, min(point.y(), self.height() - 1)),
        )

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() != Qt.LeftButton:
            return
        self._start = self._bounded(event.pos())
        self._current = self._start
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if self._start is None or not event.buttons() & Qt.LeftButton:
            return
        self._current = self._bounded(event.pos())
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() != Qt.LeftButton or self._start is None:
            return
        self._current = self._bounded(event.pos())
        region = region_from_selection(
            self.selection_rect(), self._virtual_geometry, self.minimum_size
        )
        if region is None:
            self._start = None
            self._current = None
            self.update()
            return
        self.hide()
        self.selected.emit(region)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            return
        if event.key() == Qt.Key_Escape:
            self.cancel()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(8, 12, 20, 176))
        selection = self.selection_rect()
        if selection.isEmpty():
            return

        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.fillRect(selection, Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.setPen(QPen(QColor("#56B4E9"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(selection)

        label = f"{selection.width()} x {selection.height()}"
        metrics = painter.fontMetrics()
        label_rect = QRect(
            selection.left(),
            max(6, selection.top() - metrics.height() - 10),
            metrics.horizontalAdvance(label) + 16,
            metrics.height() + 8,
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(20, 24, 32, 230))
        painter.drawRect(label_rect)
        painter.setPen(QColor(245, 247, 250))
        painter.drawText(label_rect, Qt.AlignCenter, label)


class CalibrationController:
    def __init__(
        self,
        selector: RegionSelector,
        is_paused: Callable[[], bool],
        set_paused: Callable[[bool], object],
        hide_overlay: Callable[[], None],
        show_overlay: Callable[[], None],
        apply_region: Callable[[CaptureRegion], None],
        pause_changed: Callable[[bool], None] | None = None,
    ) -> None:
        self.selector = selector
        self.is_paused = is_paused
        self.set_paused = set_paused
        self.hide_overlay = hide_overlay
        self.show_overlay = show_overlay
        self.apply_region = apply_region
        self.pause_changed = pause_changed or (lambda _paused: None)
        self._restore_pause_state = False
        self.selector.selected.connect(self._selected)
        self.selector.cancelled.connect(self._finish)

    def start(self) -> None:
        if self.selector.isVisible():
            return
        self._restore_pause_state = self.is_paused()
        self.set_paused(True)
        self.pause_changed(True)
        self.hide_overlay()
        self.selector.begin()

    def _selected(self, value: object) -> None:
        if isinstance(value, CaptureRegion):
            self.apply_region(value)
        self._finish()

    def _finish(self) -> None:
        self.selector.hide()
        self.show_overlay()
        self.set_paused(self._restore_pause_state)
        self.pause_changed(self._restore_pause_state)
