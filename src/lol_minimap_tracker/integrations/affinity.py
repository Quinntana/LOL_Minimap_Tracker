"""Windows display-affinity adapter."""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from typing import Any

from ..domain.models import AffinityResult, AffinityStatus

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011


class WindowsDisplayAffinityController:
    def __init__(self, user32: Any | None = None, windows_build: int | None = None) -> None:
        self.user32 = user32
        self.windows_build = windows_build
        if self.user32 is None and os.name == "nt":
            self.user32 = ctypes.WinDLL("user32", use_last_error=True)

    def _build(self) -> int:
        if self.windows_build is not None:
            return self.windows_build
        if os.name != "nt":
            return 0
        return int(sys.getwindowsversion().build)

    def apply(self, window_handle: int, enabled: bool) -> AffinityResult:
        if not enabled:
            return AffinityResult(AffinityStatus.DISABLED)
        if self.user32 is None or self._build() < 19041:
            return AffinityResult(AffinityStatus.UNSUPPORTED)

        target = WDA_EXCLUDEFROMCAPTURE
        if not self.user32.SetWindowDisplayAffinity(wintypes.HWND(window_handle), target):
            return AffinityResult(AffinityStatus.FAILED, ctypes.get_last_error())

        affinity = wintypes.DWORD(WDA_NONE)
        if not self.user32.GetWindowDisplayAffinity(
            wintypes.HWND(window_handle), ctypes.byref(affinity)
        ):
            return AffinityResult(AffinityStatus.FAILED, ctypes.get_last_error())
        if affinity.value != target:
            return AffinityResult(AffinityStatus.FAILED)
        return AffinityResult(AffinityStatus.ACTIVE)
