"""Injectable boundaries used by the tracker engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .models import (
    AffinityResult,
    ChampionView,
    DetectionFrame,
    RosterMember,
    RosterResult,
)

Image: TypeAlias = NDArray[np.uint8]


class RosterProvider(Protocol):
    def poll(self) -> RosterResult: ...


class PortraitProvider(Protocol):
    def get_portraits(self, members: tuple[RosterMember, ...]) -> Mapping[str, Image]: ...


class FrameSource(Protocol):
    def start(self) -> None: ...

    def capture(self) -> Image: ...

    def close(self) -> None: ...


class ChampionDetector(Protocol):
    def process(self, image: Image, portraits: Mapping[str, Image]) -> DetectionFrame: ...


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def timestamp(self) -> str: ...


class TimelineSink(Protocol):
    def record(self, timestamp: str, champions: tuple[ChampionView, ...]) -> None: ...

    def flush(self) -> None: ...


class HotkeyService(Protocol):
    def start(
        self, bindings: Mapping[str, str], callback: Callable[[str], None]
    ) -> tuple[str, ...]: ...

    def stop(self) -> None: ...


class DisplayAffinityController(Protocol):
    def apply(self, window_handle: int, enabled: bool) -> AffinityResult: ...


class OverlayInputController(Protocol):
    def apply(self, window_handle: int) -> bool: ...
