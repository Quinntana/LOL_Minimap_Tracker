"""Desktop and League-window frame-source adapters."""

from __future__ import annotations

import importlib
import os
import sys
import time
from collections.abc import Callable
from threading import Condition, RLock
from typing import Protocol, cast

import cv2
import mss
import numpy as np

from ..config import CaptureRegion
from ..domain.interfaces import Image
from .league_window import LeagueWindowFinder, WindowGeometry


class CaptureUnavailableError(RuntimeError):
    """Raised when an isolated League-window frame is temporarily unavailable."""


class WindowFinder(Protocol):
    def find(self) -> WindowGeometry: ...

    def geometry(self, hwnd: int) -> WindowGeometry: ...


class CaptureControl(Protocol):
    def stop(self) -> None: ...

    def wait(self) -> None: ...


class WindowCaptureSession(Protocol):
    def event(self, handler: Callable[..., object]) -> object: ...

    def start_free_threaded(self) -> CaptureControl: ...


WindowCaptureFactory = Callable[[int], WindowCaptureSession]


class WindowCaptureFrame(Protocol):
    frame_buffer: Image


def _default_window_capture_factory(hwnd: int) -> WindowCaptureSession:
    """Import the optional native WGC package only when the backend is used."""
    try:
        module = importlib.import_module("windows_capture")
        capture_type = module.WindowsCapture
    except (ImportError, AttributeError) as exc:
        raise CaptureUnavailableError(
            "The windows-capture package is required for League window capture"
        ) from exc
    options: dict[str, object] = {"window_hwnd": hwnd}
    if os.name == "nt" and sys.getwindowsversion().build >= 19041:
        options["cursor_capture"] = False
    return cast(WindowCaptureSession, capture_type(**options))


class MssFrameSource:
    def __init__(self, region: CaptureRegion) -> None:
        self._region = region
        self._lock = RLock()
        self._capture: mss.mss | None = None

    @property
    def region(self) -> CaptureRegion:
        with self._lock:
            return self._region

    @property
    def screen_region(self) -> CaptureRegion:
        return self.region

    @property
    def isolates_overlay(self) -> bool:
        """MSS reads the composed desktop and therefore needs display affinity."""
        return False

    @property
    def game_client_center(self) -> tuple[int, int] | None:
        """Desktop fallback does not own League window geometry."""
        return None

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


class LeagueWindowFrameSource:
    """Capture only the League HWND and expose a synchronous minimap frame API.

    ``region`` and :meth:`set_region` accept physical screen coordinates for
    compatibility with the existing calibration UI.  Once a League window is
    found, the region is converted to client-relative coordinates so it follows
    the game window when it moves.
    """

    def __init__(
        self,
        region: CaptureRegion,
        *,
        finder: WindowFinder | None = None,
        capture_factory: WindowCaptureFactory | None = None,
        timeout_seconds: float = 1.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._pending_screen_region = region
        self._client_region: CaptureRegion | None = None
        self._screen_region = region
        self._finder = finder or LeagueWindowFinder()
        self._capture_factory = capture_factory or _default_window_capture_factory
        self._timeout_seconds = timeout_seconds
        self._condition = Condition(RLock())
        self._started = False
        self._session: WindowCaptureSession | None = None
        self._control: CaptureControl | None = None
        self._geometry: WindowGeometry | None = None
        self._latest: Image | None = None
        self._sequence = 0
        self._delivered_sequence = 0
        self._generation = 0
        self._region_generation = 0
        self._last_error: Exception | None = None

    @property
    def isolates_overlay(self) -> bool:
        """A WGC item for the League HWND cannot contain the separate overlay HWND."""
        return True

    @property
    def screen_region(self) -> CaptureRegion:
        with self._condition:
            return self._screen_region

    @property
    def game_client_center(self) -> tuple[int, int] | None:
        with self._condition:
            if self._geometry is None:
                return None
            return (
                self._geometry.client_left + self._geometry.client_width // 2,
                self._geometry.client_top + self._geometry.client_height // 2,
            )

    @property
    def region(self) -> CaptureRegion:
        """Return the current physical screen-space minimap region."""
        return self.screen_region

    @property
    def client_region(self) -> CaptureRegion | None:
        with self._condition:
            return self._client_region

    def set_region(self, region: CaptureRegion) -> None:
        """Set a physical screen-space region and convert it against current geometry."""
        with self._condition:
            self._region_generation += 1
            if self._geometry is None:
                self._pending_screen_region = region
                self._client_region = None
                self._screen_region = region
            else:
                self._client_region = self._geometry.screen_to_client(region)
                self._pending_screen_region = region
                self._screen_region = self._geometry.client_to_screen(self._client_region)
            self._latest = None
            self._delivered_sequence = self._sequence

    def set_client_region(self, region: CaptureRegion) -> None:
        """Set a client-relative minimap region for a migrated configuration."""
        with self._condition:
            self._region_generation += 1
            if self._geometry is not None:
                self._geometry.validate_client_region(region)
                self._screen_region = self._geometry.client_to_screen(region)
            self._client_region = region
            self._latest = None
            self._delivered_sequence = self._sequence

    def start(self) -> None:
        """Enable capture without requiring the game window to exist yet."""
        with self._condition:
            self._started = True

    def _receive_frame(self, generation: int, frame: object, control: object) -> None:
        try:
            with self._condition:
                if generation != self._generation or self._session is None:
                    return
                geometry = self._geometry
                client_region = self._client_region
                region_generation = self._region_generation
            if geometry is None or client_region is None:
                raise CaptureUnavailableError("League capture geometry is not initialized")

            geometry = self._finder.geometry(geometry.hwnd)
            frame_region = geometry.client_to_frame(client_region)
            buffer = np.asarray(cast(WindowCaptureFrame, frame).frame_buffer)
            if buffer.ndim != 3 or buffer.shape[2] < 3:
                raise CaptureUnavailableError("Windows Graphics Capture returned an invalid frame")
            right = frame_region.left + frame_region.width
            bottom = frame_region.top + frame_region.height
            if (
                frame_region.left < 0
                or frame_region.top < 0
                or right > buffer.shape[1]
                or bottom > buffer.shape[0]
            ):
                raise CaptureUnavailableError(
                    "Configured minimap region is outside the captured League window"
                )

            # windows-capture exposes callback-owned BGRA memory.  Copy before
            # returning from the callback so the detector never reads reused data.
            image = np.array(
                buffer[
                    frame_region.top : bottom,
                    frame_region.left : right,
                    :3,
                ],
                dtype=np.uint8,
                copy=True,
                order="C",
            )
            with self._condition:
                if (
                    generation != self._generation
                    or region_generation != self._region_generation
                    or self._session is None
                ):
                    return
                self._geometry = geometry
                self._screen_region = geometry.client_to_screen(client_region)
                self._latest = image
                self._sequence += 1
                self._last_error = None
                self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                if generation == self._generation:
                    self._last_error = exc
                    self._condition.notify_all()
            stop = getattr(control, "stop", None)
            if callable(stop):
                stop()

    def _session_closed(self, generation: int) -> None:
        with self._condition:
            if generation != self._generation:
                return
            self._last_error = CaptureUnavailableError("League capture session closed")
            self._condition.notify_all()

    def _ensure_session(self) -> None:
        with self._condition:
            if not self._started:
                raise RuntimeError("Frame source has not been started")
            if self._session is not None:
                return

        try:
            geometry = self._finder.find()
            with self._condition:
                if self._client_region is None:
                    self._client_region = geometry.screen_to_client(self._pending_screen_region)
                else:
                    geometry.validate_client_region(self._client_region)
                self._screen_region = geometry.client_to_screen(self._client_region)
                self._geometry = geometry
                self._latest = None
                self._last_error = None
                self._delivered_sequence = self._sequence
                self._generation += 1
                generation = self._generation

            session = self._capture_factory(geometry.hwnd)

            def on_frame_arrived(frame: object, control: object) -> None:
                self._receive_frame(generation, frame, control)

            def on_closed() -> None:
                self._session_closed(generation)

            with self._condition:
                if generation != self._generation or not self._started:
                    raise CaptureUnavailableError("League frame source stopped during startup")
                self._session = session

            session.event(on_frame_arrived)
            session.event(on_closed)
            control = session.start_free_threaded()
            with self._condition:
                if generation == self._generation and self._session is session:
                    self._control = control
                    return
            self._stop_control(control)
        except Exception as exc:
            self._discard_session()
            if isinstance(exc, CaptureUnavailableError):
                raise
            raise CaptureUnavailableError(f"Could not start League window capture: {exc}") from exc

    @staticmethod
    def _stop_control(control: CaptureControl | None) -> None:
        if control is None:
            return
        try:
            control.stop()
        finally:
            control.wait()

    def _discard_session(self) -> None:
        with self._condition:
            control = self._control
            self._control = None
            self._session = None
            self._geometry = None
            self._latest = None
            self._last_error = None
            self._generation += 1
            self._condition.notify_all()
        self._stop_control(control)

    def capture(self) -> Image:
        self._ensure_session()
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            with self._condition:
                if self._last_error is not None:
                    error = self._last_error
                    break
                if self._latest is not None and self._sequence > self._delivered_sequence:
                    self._delivered_sequence = self._sequence
                    return self._latest.copy()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = CaptureUnavailableError(
                        "Timed out waiting for a new League window frame"
                    )
                    break
                self._condition.wait(remaining)

        self._discard_session()
        if isinstance(error, CaptureUnavailableError):
            raise error
        raise CaptureUnavailableError(f"League window capture failed: {error}") from error

    def close(self) -> None:
        with self._condition:
            self._started = False
        self._discard_session()
