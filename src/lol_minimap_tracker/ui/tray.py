"""System tray controls and Qt hotkey bridge."""

from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QAction, QApplication, QMenu, QStyle, QSystemTrayIcon

from ..domain.models import AffinityResult, TrackerSnapshot


class ActionBridge(QObject):
    requested = pyqtSignal(str)


class TrayController:
    def __init__(self, app: QApplication, dispatch: Callable[[str], None]) -> None:
        style = app.style()
        assert style is not None
        self.tray = QSystemTrayIcon(style.standardIcon(QStyle.SP_ComputerIcon), app)
        self.tray.setToolTip("LoL Minimap Tracker - waiting")
        self.menu = QMenu()
        self.status_action = QAction("Status: waiting", self.menu)
        self.status_action.setEnabled(False)
        self.menu.addAction(self.status_action)
        self.analysis_action = QAction("Analysis: waiting for live frames", self.menu)
        self.analysis_action.setEnabled(False)
        self.menu.addAction(self.analysis_action)
        self.affinity_action = QAction("Capture exclusion: pending", self.menu)
        self.affinity_action.setEnabled(False)
        self.menu.addAction(self.affinity_action)
        self.menu.addSeparator()
        self.toggle_actions: dict[str, QAction] = {}
        self._add_toggle("Show direction arrows", "toggle_arrows", True, dispatch)
        self._add_toggle("Show last-seen markers", "toggle_last_seen", True, dispatch)
        self._add_toggle("Record timeline", "toggle_timeline_logging", False, dispatch)
        self._add_toggle("Pause tracking", "pause", False, dispatch)
        self.menu.addSeparator()
        self._add_command("Save timeline", "save_timeline", dispatch)
        self._add_command("Open data folder", "open_data_folder", dispatch)
        self.menu.addSeparator()
        self._add_command("Quit", "quit", dispatch)
        self.tray.setContextMenu(self.menu)

    def _add_toggle(
        self,
        label: str,
        name: str,
        checked: bool,
        dispatch: Callable[[str], None],
    ) -> None:
        action = QAction(label, self.menu)
        action.setCheckable(True)
        action.setChecked(checked)
        action.triggered.connect(lambda _checked=False, key=name: dispatch(key))
        self.menu.addAction(action)
        self.toggle_actions[name] = action

    def _add_command(self, label: str, name: str, dispatch: Callable[[str], None]) -> None:
        action = QAction(label, self.menu)
        action.triggered.connect(lambda _checked=False, key=name: dispatch(key))
        self.menu.addAction(action)

    def show(self) -> None:
        self.tray.show()

    def update_action(self, name: str, checked: bool) -> None:
        action = self.toggle_actions.get(name)
        if action is not None:
            action.setChecked(checked)

    def update_snapshot(self, snapshot: TrackerSnapshot) -> None:
        self.status_action.setText(f"Status: {snapshot.message}")
        self.analysis_action.setText(f"Analysis: {snapshot.health.message}")
        self.tray.setToolTip(
            f"LoL Minimap Tracker - {snapshot.mode.value} - {snapshot.health.status.value}"
        )
        self.update_action("pause", snapshot.mode.value == "paused")
        self.update_action("toggle_timeline_logging", snapshot.timeline_logging)

    def update_affinity(self, result: AffinityResult) -> None:
        self.affinity_action.setText(f"Capture exclusion: {result.status.value}")
