from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import lol_minimap_tracker.integrations.league_window as league_window_module
from lol_minimap_tracker.config import CaptureRegion
from lol_minimap_tracker.integrations.capture import (
    CaptureUnavailableError,
    LeagueWindowFrameSource,
)
from lol_minimap_tracker.integrations.league_window import (
    LeagueWindowFinder,
    WindowGeometry,
    WindowGeometryError,
    enable_process_dpi_awareness,
)


def geometry(
    *,
    hwnd: int = 44,
    frame_left: int = 90,
    frame_top: int = 180,
    client_left: int = 100,
    client_top: int = 200,
    client_width: int = 380,
    client_height: int = 270,
) -> WindowGeometry:
    return WindowGeometry(
        hwnd=hwnd,
        frame_left=frame_left,
        frame_top=frame_top,
        frame_width=400,
        frame_height=300,
        client_left=client_left,
        client_top=client_top,
        client_width=client_width,
        client_height=client_height,
    )


def test_window_geometry_converts_screen_client_and_frame_coordinates() -> None:
    target = geometry(
        frame_left=-1910,
        frame_top=100,
        client_left=-1900,
        client_top=120,
    )
    screen = CaptureRegion(top=140, left=-1880, width=30, height=40)

    client = target.screen_to_client(screen)

    assert client == CaptureRegion(top=20, left=20, width=30, height=40)
    assert target.client_to_screen(client) == screen
    assert target.client_to_frame(client) == CaptureRegion(top=40, left=30, width=30, height=40)


def test_window_geometry_rejects_regions_outside_client_or_frame() -> None:
    target = geometry()
    with pytest.raises(WindowGeometryError, match="client area"):
        target.screen_to_client(CaptureRegion(top=190, left=100, width=20, height=20))
    with pytest.raises(WindowGeometryError, match="client area"):
        target.client_to_screen(CaptureRegion(top=260, left=370, width=20, height=20))


class FakeFinder:
    def __init__(self, current: WindowGeometry) -> None:
        self.current = current
        self.find_calls = 0

    def find(self) -> WindowGeometry:
        self.find_calls += 1
        return self.current

    def geometry(self, hwnd: int) -> WindowGeometry:
        assert hwnd == self.current.hwnd
        return self.current


class HookFinder(FakeFinder):
    def __init__(self, current: WindowGeometry) -> None:
        super().__init__(current)
        self.hook: Any | None = None

    def geometry(self, hwnd: int) -> WindowGeometry:
        result = super().geometry(hwnd)
        if self.hook is not None:
            hook, self.hook = self.hook, None
            hook()
        return result


@dataclass
class FakeFrame:
    frame_buffer: np.ndarray[Any, Any]


class FakeInternalControl:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class FakeCaptureControl:
    def __init__(self) -> None:
        self.stop_calls = 0
        self.wait_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1

    def wait(self) -> None:
        self.wait_calls += 1


class FakeSession:
    def __init__(self, initial_frame: np.ndarray[Any, Any] | None = None) -> None:
        self.initial_frame = initial_frame
        self.handlers: dict[str, Any] = {}
        self.control = FakeCaptureControl()
        self.started = False

    def event(self, handler: Any) -> Any:
        self.handlers[handler.__name__] = handler
        return handler

    def start_free_threaded(self) -> FakeCaptureControl:
        self.started = True
        if self.initial_frame is not None:
            self.emit(self.initial_frame)
            # Emulate the native callback buffer becoming invalid immediately.
            self.initial_frame.fill(0)
        return self.control

    def emit(self, frame: np.ndarray[Any, Any]) -> FakeInternalControl:
        control = FakeInternalControl()
        self.handlers["on_frame_arrived"](FakeFrame(frame), control)
        return control

    def close_capture_item(self) -> None:
        self.handlers["on_closed"]()


class FakeFactory:
    def __init__(self, sessions: list[FakeSession]) -> None:
        self.sessions = sessions
        self.hwnds: list[int] = []

    def __call__(self, hwnd: int) -> FakeSession:
        self.hwnds.append(hwnd)
        return self.sessions[len(self.hwnds) - 1]


def colored_frame(
    value: tuple[int, int, int], *, left: int = 30, top: int = 40
) -> np.ndarray[Any, Any]:
    frame = np.zeros((300, 400, 4), dtype=np.uint8)
    # Client region (20,20,3,2) starts at frame coordinate (30,40).
    frame[top : top + 2, left : left + 3, :3] = value
    frame[:, :, 3] = 255
    return frame


def test_window_frame_source_copies_bgr_crop_and_tracks_moved_window() -> None:
    finder = FakeFinder(geometry())
    first_buffer = colored_frame((11, 22, 33))
    session = FakeSession(first_buffer)
    factory = FakeFactory([session])
    source = LeagueWindowFrameSource(
        CaptureRegion(top=220, left=120, width=3, height=2),
        finder=finder,
        capture_factory=factory,
        timeout_seconds=0.1,
    )

    assert source.isolates_overlay
    assert source.game_client_center is None
    source.start()
    first = source.capture()

    assert factory.hwnds == [44]
    assert first.shape == (2, 3, 3)
    assert np.all(first == np.array([11, 22, 33], dtype=np.uint8))
    assert np.all(first_buffer == 0)
    assert source.client_region == CaptureRegion(top=20, left=20, width=3, height=2)
    assert source.screen_region == CaptureRegion(top=220, left=120, width=3, height=2)
    assert source.game_client_center == (290, 335)

    finder.current = geometry(
        frame_left=290,
        frame_top=380,
        client_left=300,
        client_top=400,
    )
    session.emit(colored_frame((4, 5, 6)))
    second = source.capture()

    assert np.all(second == np.array([4, 5, 6], dtype=np.uint8))
    assert source.screen_region == CaptureRegion(top=420, left=320, width=3, height=2)
    assert source.game_client_center == (490, 535)
    source.close()
    assert session.control.stop_calls == 1
    assert session.control.wait_calls == 1


def test_window_frame_source_discards_crop_racing_with_calibration() -> None:
    finder = HookFinder(geometry())
    session = FakeSession(colored_frame((1, 2, 3)))
    source = LeagueWindowFrameSource(
        CaptureRegion(top=220, left=120, width=3, height=2),
        finder=finder,
        capture_factory=FakeFactory([session]),
        timeout_seconds=0.1,
    )
    source.start()
    source.capture()

    finder.hook = lambda: source.set_region(CaptureRegion(top=221, left=121, width=3, height=2))
    session.emit(colored_frame((4, 5, 6)))
    session.emit(colored_frame((7, 8, 9), left=31, top=41))

    current = source.capture()
    assert np.all(current == np.array([7, 8, 9], dtype=np.uint8))
    assert source.screen_region == CaptureRegion(top=221, left=121, width=3, height=2)
    source.close()


def test_window_frame_source_times_out_and_never_falls_back_to_desktop() -> None:
    session = FakeSession()
    source = LeagueWindowFrameSource(
        CaptureRegion(top=220, left=120, width=3, height=2),
        finder=FakeFinder(geometry()),
        capture_factory=FakeFactory([session]),
        timeout_seconds=0.01,
    )
    source.start()

    with pytest.raises(CaptureUnavailableError, match="Timed out"):
        source.capture()

    assert session.control.stop_calls == 1
    assert session.control.wait_calls == 1


def test_window_frame_source_restarts_when_the_last_frame_goes_stale() -> None:
    frozen = FakeSession(colored_frame((1, 2, 3)))
    recovered = FakeSession(colored_frame((7, 8, 9)))
    factory = FakeFactory([frozen, recovered])
    source = LeagueWindowFrameSource(
        CaptureRegion(top=220, left=120, width=3, height=2),
        finder=FakeFinder(geometry()),
        capture_factory=factory,
        timeout_seconds=0.01,
    )
    source.start()

    first = source.capture()
    assert np.all(first == np.array([1, 2, 3], dtype=np.uint8))
    with pytest.raises(CaptureUnavailableError, match="Timed out waiting for a new"):
        source.capture()

    assert frozen.control.stop_calls == 1
    assert frozen.control.wait_calls == 1
    next_frame = source.capture()
    assert np.all(next_frame == np.array([7, 8, 9], dtype=np.uint8))
    assert factory.hwnds == [44, 44]
    source.close()


def test_window_frame_source_recovers_after_capture_item_closes() -> None:
    first = FakeSession(colored_frame((1, 2, 3)))
    second = FakeSession(colored_frame((7, 8, 9)))
    factory = FakeFactory([first, second])
    source = LeagueWindowFrameSource(
        CaptureRegion(top=220, left=120, width=3, height=2),
        finder=FakeFinder(geometry()),
        capture_factory=factory,
        timeout_seconds=0.1,
    )
    source.start()
    source.capture()
    first.close_capture_item()

    with pytest.raises(CaptureUnavailableError, match="session closed"):
        source.capture()

    recovered = source.capture()
    assert np.all(recovered == np.array([7, 8, 9], dtype=np.uint8))
    assert factory.hwnds == [44, 44]
    source.close()


def test_window_frame_source_requires_start_and_accepts_client_region() -> None:
    source = LeagueWindowFrameSource(
        CaptureRegion(),
        finder=FakeFinder(geometry()),
        capture_factory=FakeFactory([FakeSession(colored_frame((1, 1, 1)))]),
    )
    source.set_client_region(CaptureRegion(top=20, left=20, width=3, height=2))
    with pytest.raises(RuntimeError, match="started"):
        source.capture()


def handle_value(handle: object) -> int:
    return int(getattr(handle, "value", handle) or 0)


class FakeUser32:
    def __init__(self, windows: dict[int, dict[str, Any]]) -> None:
        self.windows = windows

    def EnumWindows(self, callback: Any, lparam: int) -> int:
        for hwnd in self.windows:
            callback(hwnd, lparam)
        return 1

    def IsWindowVisible(self, hwnd: object) -> int:
        return int(self.windows[handle_value(hwnd)]["visible"])

    def IsIconic(self, hwnd: object) -> int:
        return int(self.windows[handle_value(hwnd)]["minimized"])

    def GetWindowThreadProcessId(self, hwnd: object, process_id: Any) -> int:
        process_id._obj.value = self.windows[handle_value(hwnd)]["pid"]
        return 1

    def GetClientRect(self, hwnd: object, rectangle: Any) -> int:
        item = self.windows[handle_value(hwnd)]
        rectangle._obj.left = 0
        rectangle._obj.top = 0
        rectangle._obj.right = item["client_width"]
        rectangle._obj.bottom = item["client_height"]
        return 1

    def ClientToScreen(self, hwnd: object, point: Any) -> int:
        item = self.windows[handle_value(hwnd)]
        point._obj.x += item["client_left"]
        point._obj.y += item["client_top"]
        return 1

    def GetWindowRect(self, hwnd: object, rectangle: Any) -> int:
        item = self.windows[handle_value(hwnd)]
        left, top, right, bottom = item["frame"]
        rectangle._obj.left = left
        rectangle._obj.top = top
        rectangle._obj.right = right
        rectangle._obj.bottom = bottom
        return 1


class FakeKernel32:
    def __init__(self, windows: dict[int, dict[str, Any]]) -> None:
        self.paths = {
            item["pid"]: f"C:\\Riot Games\\League of Legends\\Game\\{item['image']}"
            for item in windows.values()
        }
        self.closed: list[int] = []

    def OpenProcess(self, _access: int, _inherit: bool, process_id: int) -> int:
        return process_id

    def QueryFullProcessImageNameW(
        self, process: int, _flags: int, buffer: Any, length: Any
    ) -> int:
        value = self.paths[process]
        buffer.value = value
        length._obj.value = len(value)
        return 1

    def CloseHandle(self, process: int) -> int:
        self.closed.append(process)
        return 1


class FakeDwmApi:
    def __init__(self, windows: dict[int, dict[str, Any]]) -> None:
        self.windows = windows

    def DwmGetWindowAttribute(
        self, hwnd: object, _attribute: int, rectangle: Any, _size: int
    ) -> int:
        left, top, right, bottom = self.windows[handle_value(hwnd)]["frame"]
        rectangle._obj.left = left
        rectangle._obj.top = top
        rectangle._obj.right = right
        rectangle._obj.bottom = bottom
        return 0


def window(
    pid: int,
    image: str,
    width: int,
    height: int,
    *,
    visible: bool = True,
    minimized: bool = False,
) -> dict[str, Any]:
    return {
        "pid": pid,
        "image": image,
        "visible": visible,
        "minimized": minimized,
        "client_left": 100,
        "client_top": 200,
        "client_width": width,
        "client_height": height,
        "frame": (90, 180, 110 + width, 220 + height),
    }


def test_league_window_finder_uses_exact_process_and_largest_visible_window() -> None:
    windows = {
        1: window(101, "LeagueClient.exe", 1000, 800),
        2: window(102, "League of Legends.exe", 300, 200),
        3: window(103, "League of Legends.exe", 1280, 720),
        4: window(104, "League of Legends.exe", 1920, 1080, minimized=True),
    }
    kernel32 = FakeKernel32(windows)
    finder = LeagueWindowFinder(
        user32=FakeUser32(windows),
        kernel32=kernel32,
        dwmapi=FakeDwmApi(windows),
    )

    result = finder.find()

    assert result.hwnd == 3
    assert result.client_width == 1280
    assert result.client_height == 720
    assert kernel32.closed == [101, 102, 103]


def test_dpi_awareness_accepts_an_already_matching_context(monkeypatch: Any) -> None:
    class FakeDpiUser32:
        def SetProcessDpiAwarenessContext(self, _target: object) -> int:
            return 0

        def GetThreadDpiAwarenessContext(self) -> int:
            return -4

        def AreDpiAwarenessContextsEqual(self, _current: object, _target: object) -> int:
            return 1

    monkeypatch.setattr(
        league_window_module.ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: FakeDpiUser32(),
    )
    assert enable_process_dpi_awareness()
