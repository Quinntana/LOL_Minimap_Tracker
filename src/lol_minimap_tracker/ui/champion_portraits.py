"""Cached Qt rendering for Data Dragon champion portraits."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QImage, QPainter, QPainterPath, QPixmap

from ..domain.interfaces import Image


class ChampionPortraitRenderer:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], QPixmap] = {}

    def render(self, champion_name: str, portrait: Image, size: int) -> QPixmap:
        key = champion_name, size
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if portrait.ndim != 3 or portrait.shape[0] < 1 or portrait.shape[1] < 1:
            return QPixmap()
        if portrait.shape[2] < 3 or size < 1:
            return QPixmap()

        bgr = np.ascontiguousarray(portrait[:, :, :3])
        height, width = bgr.shape[:2]
        source = QImage(
            bgr.data,
            width,
            height,
            int(bgr.strides[0]),
            QImage.Format_BGR888,
        ).copy()
        scaled = QPixmap.fromImage(source).scaled(
            size,
            size,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )

        canvas = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
        canvas.fill(Qt.transparent)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        clip = QPainterPath()
        clip.addEllipse(0, 0, size, size)
        painter.setClipPath(clip)
        painter.drawPixmap(QRect(0, 0, size, size), scaled)
        painter.end()

        result = QPixmap.fromImage(canvas)
        self._cache[key] = result
        return result
