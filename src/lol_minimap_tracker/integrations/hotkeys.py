"""Failure-tolerant global hotkey adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress

import keyboard


class KeyboardHotkeyService:
    def start(
        self,
        bindings: Mapping[str, str],
        callback: Callable[[str], None],
    ) -> tuple[str, ...]:
        failures: list[str] = []
        for action, hotkey in bindings.items():
            try:
                keyboard.add_hotkey(hotkey, lambda name=action: callback(name))
            except Exception as exc:
                failures.append(f"{action} ({hotkey}): {exc}")
        return tuple(failures)

    def stop(self) -> None:
        with suppress(Exception):
            keyboard.unhook_all_hotkeys()
