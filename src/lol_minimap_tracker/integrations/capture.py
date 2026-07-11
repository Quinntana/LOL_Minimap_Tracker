"""Reusable MSS screen capture adapter."""

from __future__ import annotations

import cv2
import mss
import numpy as np

from ..config import CaptureRegion
from ..domain.interfaces import Image


class MssFrameSource:
    def __init__(self, region: CaptureRegion) -> None:
        self.region = region
        self._capture: mss.mss | None = None

    def start(self) -> None:
        if self._capture is None:
            self._capture = mss.mss()

    def capture(self) -> Image:
        if self._capture is None:
            raise RuntimeError("Frame source has not been started")
        frame = self._capture.grab(
            {
                "top": self.region.top,
                "left": self.region.left,
                "width": self.region.width,
                "height": self.region.height,
            }
        )
        return cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.close()
            self._capture = None
