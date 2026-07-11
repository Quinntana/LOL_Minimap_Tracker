from __future__ import annotations

from typing import Any

from lol_minimap_tracker.domain.models import AffinityStatus
from lol_minimap_tracker.integrations.affinity import (
    WDA_EXCLUDEFROMCAPTURE,
    WindowsDisplayAffinityController,
)


class User32:
    def __init__(
        self, set_ok: bool = True, get_ok: bool = True, value: int = WDA_EXCLUDEFROMCAPTURE
    ) -> None:
        self.set_ok = set_ok
        self.get_ok = get_ok
        self.value = value

    def SetWindowDisplayAffinity(self, *_args: Any) -> bool:
        return self.set_ok

    def GetWindowDisplayAffinity(self, _handle: Any, pointer: Any) -> bool:
        pointer._obj.value = self.value
        return self.get_ok


def test_affinity_success_disabled_and_unsupported() -> None:
    assert (
        WindowsDisplayAffinityController(User32(), 22621).apply(1, True).status
        is AffinityStatus.ACTIVE
    )
    assert (
        WindowsDisplayAffinityController(User32(), 22621).apply(1, False).status
        is AffinityStatus.DISABLED
    )
    assert (
        WindowsDisplayAffinityController(User32(), 18362).apply(1, True).status
        is AffinityStatus.UNSUPPORTED
    )


def test_affinity_failure_and_wrong_value() -> None:
    assert (
        WindowsDisplayAffinityController(User32(set_ok=False), 22621).apply(1, True).status
        is AffinityStatus.FAILED
    )
    assert (
        WindowsDisplayAffinityController(User32(value=1), 22621).apply(1, True).status
        is AffinityStatus.FAILED
    )


def test_affinity_get_failure_and_non_windows_fallback() -> None:
    assert (
        WindowsDisplayAffinityController(User32(get_ok=False), 22621).apply(1, True).status
        is AffinityStatus.FAILED
    )
    assert (
        WindowsDisplayAffinityController(user32=None, windows_build=0).apply(1, True).status
        is AffinityStatus.UNSUPPORTED
    )
