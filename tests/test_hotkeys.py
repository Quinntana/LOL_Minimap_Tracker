from __future__ import annotations

from typing import Any

import lol_minimap_tracker.integrations.hotkeys as hotkey_module
from lol_minimap_tracker.integrations.hotkeys import KeyboardHotkeyService


def test_hotkey_failure_is_reported_and_cleanup_is_safe(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        hotkey_module.keyboard,
        "add_hotkey",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("blocked")),
    )
    monkeypatch.setattr(hotkey_module.keyboard, "unhook_all_hotkeys", lambda: None)
    service = KeyboardHotkeyService()
    failures = service.start({"quit": "ctrl+d"}, lambda _name: None)
    assert len(failures) == 1
    assert "blocked" in failures[0]
    service.stop()


def test_hotkey_success_dispatches_bound_action(monkeypatch: Any) -> None:
    registered: list[Any] = []
    monkeypatch.setattr(
        hotkey_module.keyboard,
        "add_hotkey",
        lambda _hotkey, callback: registered.append(callback),
    )
    monkeypatch.setattr(
        hotkey_module.keyboard,
        "unhook_all_hotkeys",
        lambda: (_ for _ in ()).throw(RuntimeError("already removed")),
    )
    dispatched: list[str] = []
    service = KeyboardHotkeyService()
    assert service.start({"pause": "ctrl+p"}, dispatched.append) == ()
    registered[0]()
    assert dispatched == ["pause"]
    service.stop()
