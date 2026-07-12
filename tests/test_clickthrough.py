from __future__ import annotations

from typing import Any

from lol_minimap_tracker.integrations.clickthrough import (
    REQUIRED_EXSTYLE,
    STYLE_REFRESH_FLAGS,
    WS_EX_NOACTIVATE,
    WS_EX_TRANSPARENT,
    WindowsOverlayInputController,
)


class User32:
    def __init__(self, style: int = 0x00080080, position_ok: bool = True) -> None:
        self.style = style
        self.position_ok = position_ok
        self.set_styles: list[int] = []
        self.position_flags: list[int] = []

    def GetWindowLongW(self, *_args: Any) -> int:
        return self.style

    def SetWindowLongW(self, _hwnd: Any, _index: int, style: int) -> int:
        previous = self.style
        self.style = style
        self.set_styles.append(style)
        return previous

    def SetWindowPos(self, *_args: Any) -> bool:
        self.position_flags.append(int(_args[-1]))
        return self.position_ok


def test_native_clickthrough_adds_and_verifies_required_styles() -> None:
    user32 = User32()
    controller = WindowsOverlayInputController(user32)
    assert controller.apply(123)
    assert user32.style & WS_EX_TRANSPARENT
    assert user32.style & WS_EX_NOACTIVATE
    assert user32.style & 0x00080000
    assert user32.set_styles == [0x00080080 | REQUIRED_EXSTYLE]
    assert user32.position_flags == [STYLE_REFRESH_FLAGS]


def test_native_clickthrough_does_not_rewrite_verified_style() -> None:
    user32 = User32(REQUIRED_EXSTYLE | 0x00080000)
    assert WindowsOverlayInputController(user32).apply(123)
    assert not user32.set_styles
    assert not user32.position_flags


def test_native_clickthrough_reports_style_refresh_failure() -> None:
    user32 = User32(position_ok=False)
    controller = WindowsOverlayInputController(user32)
    assert not controller.apply(123)
    assert not controller.apply(123)
    assert user32.position_flags == [STYLE_REFRESH_FLAGS, STYLE_REFRESH_FLAGS]
