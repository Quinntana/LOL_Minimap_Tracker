"""Reusable MSS screen capture adapter."""

from __future__ import annotations

from threading import RLock

import cv2
import mss
import numpy as np

from ..config import CaptureRegion
from ..domain.interfaces import Image


class MssFrameSource:
    def __init__(self, region: CaptureRegion) -> None:
        self._region = region
        self._lock = RLock()
        self._capture: mss.mss | None = None

    @property
    def region(self) -> CaptureRegion:
        with self._lock:
            return self._region

    def set_region(self, region: CaptureRegion) -> None:
        with self._lock:
            self._region = region

    def start(self) -> None:
        if self._capture is None:
            self._capture = mss.mss()

    def capture(self) -> Image:
        if self._capture is None:
            raise RuntimeError("Frame source has not been started")
        region = self.region
        frame = self._capture.grab(
            {
                "top": region.top,
                "left": region.left,
                "width": region.width,
                "height": region.height,
            }
        )
        return cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.close()
            self._capture = None
