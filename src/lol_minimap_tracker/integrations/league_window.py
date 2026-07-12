"""Win32 League game-window discovery and coordinate conversion."""

from __future__ import annotations

import ctypes
import ntpath
import os
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

from ..config import CaptureRegion

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
DWMWA_EXTENDED_FRAME_BOUNDS = 9
MAX_PROCESS_PATH = 32768
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4


class LeagueWindowNotFoundError(RuntimeError):
    """Raised when the League game window is not available for capture."""


class WindowGeometryError(RuntimeError):
    """Raised when Win32 cannot provide consistent window geometry."""


def enable_process_dpi_awareness() -> bool:
    """Use physical pixels for Win32, Qt, WGC, and calibration coordinates."""
    if os.name != "nt":
        return True
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        setter = user32.SetProcessDpiAwarenessContext
        _set_signature(setter, [wintypes.HANDLE], wintypes.BOOL)
        target = wintypes.HANDLE(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        if setter(target):
            return True
        getter = user32.GetThreadDpiAwarenessContext
        equals = user32.AreDpiAwarenessContextsEqual
        _set_signature(getter, [], wintypes.HANDLE)
        _set_signature(equals, [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL)
        return bool(equals(getter(), target))
    except (AttributeError, OSError):
        # Windows 10 1703+ exposes the preferred API. Older versions are not a
        # supported WGC target, so a missing entry point is harmless here.
        return False


@dataclass(frozen=True)
class WindowGeometry:
    """Physical-pixel geometry for one capturable top-level window."""

    hwnd: int
    frame_left: int
    frame_top: int
    frame_width: int
    frame_height: int
    client_left: int
    client_top: int
    client_width: int
    client_height: int

    @staticmethod
    def _validate_region(region: CaptureRegion) -> None:
        if region.width <= 0 or region.height <= 0:
            raise WindowGeometryError("Capture region must have a positive size")

    def screen_to_client(self, region: CaptureRegion) -> CaptureRegion:
        """Convert a screen-space region to coordinates relative to the client area."""
        self._validate_region(region)
        converted = CaptureRegion(
            top=region.top - self.client_top,
            left=region.left - self.client_left,
            width=region.width,
            height=region.height,
        )
        self.validate_client_region(converted)
        return converted

    def client_to_screen(self, region: CaptureRegion) -> CaptureRegion:
        """Convert a client-relative region to physical screen coordinates."""
        self.validate_client_region(region)
        return CaptureRegion(
            top=self.client_top + region.top,
            left=self.client_left + region.left,
            width=region.width,
            height=region.height,
        )

    def client_to_frame(self, region: CaptureRegion) -> CaptureRegion:
        """Convert a client-relative region to a Windows Graphics Capture frame crop."""
        screen = self.client_to_screen(region)
        converted = CaptureRegion(
            top=screen.top - self.frame_top,
            left=screen.left - self.frame_left,
            width=screen.width,
            height=screen.height,
        )
        if (
            converted.left < 0
            or converted.top < 0
            or converted.left + converted.width > self.frame_width
            or converted.top + converted.height > self.frame_height
        ):
            raise WindowGeometryError("Client capture region falls outside the window frame")
        return converted

    def validate_client_region(self, region: CaptureRegion) -> None:
        self._validate_region(region)
        if (
            region.left < 0
            or region.top < 0
            or region.left + region.width > self.client_width
            or region.top + region.height > self.client_height
        ):
            raise WindowGeometryError("Capture region falls outside the League client area")


def _handle_value(handle: object) -> int:
    if isinstance(handle, int):
        return handle
    value = getattr(handle, "value", None)
    return value if isinstance(value, int) else 0


def _set_signature(function: Any, argtypes: list[Any], restype: Any) -> None:
    """Set ctypes signatures while remaining friendly to Python test doubles."""
    try:
        function.argtypes = argtypes
        function.restype = restype
    except (AttributeError, TypeError):
        pass


class LeagueWindowFinder:
    """Find the visible top-level window owned by the League game executable."""

    def __init__(
        self,
        executable_name: str = "League of Legends.exe",
        *,
        user32: Any | None = None,
        kernel32: Any | None = None,
        dwmapi: Any | None = None,
    ) -> None:
        if user32 is None or kernel32 is None or dwmapi is None:
            if os.name != "nt":
                raise OSError("League window capture is available only on Windows")
            user32 = user32 or ctypes.WinDLL("user32", use_last_error=True)
            kernel32 = kernel32 or ctypes.WinDLL("kernel32", use_last_error=True)
            dwmapi = dwmapi or ctypes.WinDLL("dwmapi", use_last_error=True)
        self.executable_name = executable_name.casefold()
        self.user32 = user32
        self.kernel32 = kernel32
        self.dwmapi = dwmapi
        self._configure_functions()

    def _configure_functions(self) -> None:
        callback_type = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
        )
        _set_signature(self.user32.EnumWindows, [callback_type, wintypes.LPARAM], wintypes.BOOL)
        _set_signature(
            self.user32.GetWindowThreadProcessId,
            [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)],
            wintypes.DWORD,
        )
        _set_signature(self.user32.IsWindowVisible, [wintypes.HWND], wintypes.BOOL)
        _set_signature(self.user32.IsIconic, [wintypes.HWND], wintypes.BOOL)
        _set_signature(
            self.user32.GetClientRect,
            [wintypes.HWND, ctypes.POINTER(wintypes.RECT)],
            wintypes.BOOL,
        )
        _set_signature(
            self.user32.ClientToScreen,
            [wintypes.HWND, ctypes.POINTER(wintypes.POINT)],
            wintypes.BOOL,
        )
        _set_signature(
            self.user32.GetWindowRect,
            [wintypes.HWND, ctypes.POINTER(wintypes.RECT)],
            wintypes.BOOL,
        )
        _set_signature(
            self.kernel32.OpenProcess,
            [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD],
            wintypes.HANDLE,
        )
        _set_signature(
            self.kernel32.QueryFullProcessImageNameW,
            [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)],
            wintypes.BOOL,
        )
        _set_signature(self.kernel32.CloseHandle, [wintypes.HANDLE], wintypes.BOOL)
        _set_signature(
            self.dwmapi.DwmGetWindowAttribute,
            [wintypes.HWND, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD],
            ctypes.c_long,
        )

    def _process_image_name(self, hwnd: int) -> str | None:
        process_id = wintypes.DWORD()
        if not self.user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(process_id)):
            return None
        process = self.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, process_id.value
        )
        if not process:
            return None
        try:
            buffer = ctypes.create_unicode_buffer(MAX_PROCESS_PATH)
            length = wintypes.DWORD(len(buffer))
            if not self.kernel32.QueryFullProcessImageNameW(
                process, 0, buffer, ctypes.byref(length)
            ):
                return None
            return ntpath.basename(buffer.value[: length.value])
        finally:
            self.kernel32.CloseHandle(process)

    def geometry(self, hwnd: int) -> WindowGeometry:
        """Read current frame and client bounds for an HWND."""
        client = wintypes.RECT()
        if not self.user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(client)):
            raise WindowGeometryError("GetClientRect failed for the League window")
        client_origin = wintypes.POINT(0, 0)
        if not self.user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(client_origin)):
            raise WindowGeometryError("ClientToScreen failed for the League window")

        frame = wintypes.RECT()
        result = self.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(frame),
            ctypes.sizeof(frame),
        )
        if result != 0 and not self.user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(frame)):
            raise WindowGeometryError("Could not read League window frame bounds")

        geometry = WindowGeometry(
            hwnd=hwnd,
            frame_left=frame.left,
            frame_top=frame.top,
            frame_width=frame.right - frame.left,
            frame_height=frame.bottom - frame.top,
            client_left=client_origin.x,
            client_top=client_origin.y,
            client_width=client.right - client.left,
            client_height=client.bottom - client.top,
        )
        if (
            geometry.frame_width <= 0
            or geometry.frame_height <= 0
            or geometry.client_width <= 0
            or geometry.client_height <= 0
        ):
            raise WindowGeometryError("League window has empty frame or client bounds")
        return geometry

    def find(self) -> WindowGeometry:
        """Return the largest visible, non-minimized League game window."""
        handles: list[int] = []
        callback_type = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
        )

        def collect_handle(hwnd: wintypes.HWND, _lparam: wintypes.LPARAM) -> bool:
            handles.append(_handle_value(hwnd))
            return True

        collect = callback_type(collect_handle)

        if not self.user32.EnumWindows(collect, 0):
            raise LeagueWindowNotFoundError("Could not enumerate top-level windows")

        candidates: list[WindowGeometry] = []
        for hwnd in handles:
            if not self.user32.IsWindowVisible(wintypes.HWND(hwnd)):
                continue
            if self.user32.IsIconic(wintypes.HWND(hwnd)):
                continue
            image_name = self._process_image_name(hwnd)
            if image_name is None or image_name.casefold() != self.executable_name:
                continue
            try:
                candidates.append(self.geometry(hwnd))
            except WindowGeometryError:
                continue

        if not candidates:
            raise LeagueWindowNotFoundError(
                f"No visible {self.executable_name} game window was found"
            )
        return max(candidates, key=lambda item: item.client_width * item.client_height)
