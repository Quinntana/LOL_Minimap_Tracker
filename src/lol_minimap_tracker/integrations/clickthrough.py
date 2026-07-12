"""Native verification and fallback for a non-activating click-through overlay."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Any

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
REQUIRED_EXSTYLE = WS_EX_TRANSPARENT | WS_EX_NOACTIVATE

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
STYLE_REFRESH_FLAGS = SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED


class WindowsOverlayInputController:
    def __init__(self, user32: Any | None = None) -> None:
        self.user32 = user32
        self._pending_refresh: set[int] = set()
        if self.user32 is None and os.name == "nt":
            self.user32 = ctypes.WinDLL("user32", use_last_error=True)

    def apply(self, window_handle: int) -> bool:
        if self.user32 is None:
            return False
        hwnd = wintypes.HWND(window_handle)
        current = int(self.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)) & 0xFFFFFFFF
        target = current | REQUIRED_EXSTYLE
        if target != current:
            self.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, target)
            self._pending_refresh.add(window_handle)
        if window_handle in self._pending_refresh:
            if not self.user32.SetWindowPos(
                hwnd,
                wintypes.HWND(0),
                0,
                0,
                0,
                0,
                STYLE_REFRESH_FLAGS,
            ):
                return False
            self._pending_refresh.discard(window_handle)
        verified = int(self.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)) & 0xFFFFFFFF
        return verified & REQUIRED_EXSTYLE == REQUIRED_EXSTYLE
