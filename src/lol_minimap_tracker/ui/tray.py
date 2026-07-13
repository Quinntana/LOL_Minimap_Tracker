"""Notification-area status and controls."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QCursor, QIcon
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QMenu,
    QStyle,
    QSystemTrayIcon,
)

from ..config import CaptureRegion, TrackerConfig
from ..domain.models import (
    AffinityResult,
    AffinityStatus,
    LastSeenMarkerStyle,
    TrackerSnapshot,
)


class ActionBridge(QObject):
    requested = pyqtSignal(str)


class TrayController:
    def __init__(
        self,
        app: QApplication,
        dispatch: Callable[[str], None],
        config: TrackerConfig | None = None,
        icon: QIcon | None = None,
    ) -> None:
        config = config or TrackerConfig()
        style = app.style()
        assert style is not None
        tray_icon = (
            icon
            if icon is not None and not icon.isNull()
            else style.standardIcon(QStyle.SP_ComputerIcon)
        )
        self.tray = QSystemTrayIcon(tray_icon, app)
        self.tray.setToolTip("LoL Minimap Tracker - waiting")
        self.menu = QMenu()
        self.menu.setToolTipsVisible(True)

        self.status_action = self._add_status("Game: Waiting")
        self.analysis_action = self._add_status("Analysis: Waiting for live frames")
        self.affinity_action = self._add_status("Capture exclusion: pending")
        self.region_action = self._add_status("")
        self.update_region(config.capture)
        self.menu.addSeparator()

        self._add_command(self.menu, "Select minimap area...", "select_minimap", dispatch)
        self.menu.addSeparator()
        self.toggle_actions: dict[str, QAction] = {}
        self._add_toggle(
            self.menu, "Direction arrows", "toggle_arrows", config.show_arrows, dispatch
        )
        self._add_toggle(
            self.menu,
            "Missing-enemy markers",
            "toggle_last_seen",
            config.show_last_seen,
            dispatch,
        )
        self.marker_style_menu = cast(QMenu, self.menu.addMenu("Missing marker"))
        self.marker_style_menu.setToolTipsVisible(True)
        self.marker_style_actions: dict[LastSeenMarkerStyle, QAction] = {}
        self.marker_style_group = QActionGroup(self.marker_style_menu)
        self.marker_style_group.setExclusive(True)
        self._add_marker_style(
            "Champion portrait + X (default)",
            LastSeenMarkerStyle.PORTRAIT,
            "set_marker_style_portrait",
            "Faded champion portrait with a red X. Requires League-window capture "
            "or active capture exclusion.",
            dispatch,
        )
        self._add_marker_style(
            "Role icon",
            LastSeenMarkerStyle.ROLE,
            "set_marker_style_role",
            "Compact tinted role symbol. Requires League-window capture or active "
            "capture exclusion.",
            dispatch,
        )
        self._add_marker_style(
            "Minimal color dot (fallback)",
            LastSeenMarkerStyle.DOT,
            "set_marker_style_dot",
            "Five-pixel identity-color dot. Remains visible if capture exclusion is unavailable.",
            dispatch,
        )
        self.update_marker_style(config.last_seen_marker_style)
        self._add_toggle(self.menu, "Pause detection", "pause", False, dispatch)

        self.advanced_menu = self.menu.addMenu("Advanced")
        assert self.advanced_menu is not None
        self._add_toggle(
            self.advanced_menu,
            "Notifications",
            "toggle_notifications",
            config.show_notifications,
            dispatch,
        )
        self._add_toggle(
            self.advanced_menu,
            "Timeline recording",
            "toggle_timeline_logging",
            False,
            dispatch,
        )
        self.advanced_menu.addSeparator()
        self._add_command(self.advanced_menu, "Save timeline", "save_timeline", dispatch)
        self._add_command(self.advanced_menu, "Open configuration", "open_configuration", dispatch)
        self._add_command(self.advanced_menu, "Open data folder", "open_data_folder", dispatch)

        self.menu.addSeparator()
        self._add_command(self.menu, "Exit", "quit", dispatch)
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._activated)

    def _add_status(self, label: str) -> QAction:
        action = QAction(label, self.menu)
        action.setEnabled(False)
        self.menu.addAction(action)
        return action

    def _add_toggle(
        self,
        menu: QMenu,
        label: str,
        name: str,
        checked: bool,
        dispatch: Callable[[str], None],
    ) -> None:
        action = QAction(label, menu)
        action.setCheckable(True)
        action.setChecked(checked)
        action.triggered.connect(lambda _checked=False, key=name: dispatch(key))
        menu.addAction(action)
        self.toggle_actions[name] = action

    def _add_marker_style(
        self,
        label: str,
        style: LastSeenMarkerStyle,
        action_name: str,
        tooltip: str,
        dispatch: Callable[[str], None],
    ) -> None:
        action = QAction(label, self.marker_style_menu)
        action.setCheckable(True)
        action.setToolTip(tooltip)
        action.setStatusTip(tooltip)
        action.triggered.connect(lambda _checked=False, key=action_name: dispatch(key))
        self.marker_style_group.addAction(action)
        self.marker_style_menu.addAction(action)
        self.marker_style_actions[style] = action

    @staticmethod
    def _add_command(menu: QMenu, label: str, name: str, dispatch: Callable[[str], None]) -> None:
        action = QAction(label, menu)
        action.triggered.connect(lambda _checked=False, key=name: dispatch(key))
        menu.addAction(action)

    def _activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.Trigger:
            self.menu.popup(QCursor.pos())

    def show(self) -> None:
        self.tray.show()

    def notify(
        self,
        title: str,
        message: str,
        icon: QSystemTrayIcon.MessageIcon = QSystemTrayIcon.Information,
        *,
        force: bool = False,
    ) -> None:
        notifications = self.toggle_actions.get("toggle_notifications")
        if force or (notifications is not None and notifications.isChecked()):
            self.tray.showMessage(title, message, icon, 5000)

    def update_action(self, name: str, checked: bool) -> None:
        action = self.toggle_actions.get(name)
        if action is not None:
            action.setChecked(checked)

    def update_marker_style(self, style: LastSeenMarkerStyle) -> None:
        action = self.marker_style_actions.get(style)
        if action is None:
            return
        action.setChecked(True)
        labels = {
            LastSeenMarkerStyle.PORTRAIT: "Champion portrait",
            LastSeenMarkerStyle.ROLE: "Role icon",
            LastSeenMarkerStyle.DOT: "Minimal dot",
        }
        self.marker_style_menu.setTitle(f"Missing marker: {labels[style]}")

    def update_region(self, region: CaptureRegion) -> None:
        self.region_action.setText(
            f"Minimap: {region.width} x {region.height} at {region.left}, {region.top}"
        )

    def update_snapshot(self, snapshot: TrackerSnapshot) -> None:
        self.status_action.setText(f"Game: {snapshot.message}")
        self.analysis_action.setText(f"Analysis: {snapshot.health.message}")
        self.tray.setToolTip(
            f"LoL Minimap Tracker - {snapshot.mode.value} - {snapshot.health.status.value}"
        )
        self.update_action("pause", snapshot.mode.value == "paused")
        self.update_action("toggle_timeline_logging", snapshot.timeline_logging)

    def update_affinity(self, result: AffinityResult, capture_isolated: bool = False) -> None:
        if capture_isolated:
            self.affinity_action.setText("Overlay capture: isolated (League window)")
            self.affinity_action.setToolTip(
                "Only the League game window is captured, so overlay graphics are excluded."
            )
            return
        if result.status is AffinityStatus.ACTIVE:
            self.affinity_action.setText("Capture exclusion: active")
            self.affinity_action.setToolTip("Desktop capture excludes the overlay window.")
            return
        self.affinity_action.setText(
            f"Capture exclusion: {result.status.value} - dot fallback available"
        )
        self.affinity_action.setToolTip(
            "Portrait and role markers are hidden to prevent feedback. Select the minimal "
            "dot style to keep last-seen positions visible."
        )
