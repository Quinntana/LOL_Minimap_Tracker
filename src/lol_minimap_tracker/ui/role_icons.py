"""Tinted role-icon rendering."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer


class RoleIconRenderer:
    def __init__(self, asset_dir: Path) -> None:
        self.asset_dir = asset_dir
        self._cache: dict[tuple[str, str, int, int], QPixmap] = {}

    def render(self, filename: str, color: str, size: int, opacity: float = 1.0) -> QPixmap:
        alpha = max(0, min(255, round(opacity * 255)))
        key = filename, color, size, alpha
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        renderer = QSvgRenderer(str(self.asset_dir / filename))
        painter = QPainter(image)
        renderer.render(painter, QRectF(0, 0, size, size))
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        tint = QColor(color)
        tint.setAlpha(alpha)
        painter.fillRect(image.rect(), tint)
        painter.end()
        pixmap = QPixmap.fromImage(image)
        self._cache[key] = pixmap
        return pixmap
