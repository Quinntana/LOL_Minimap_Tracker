"""Application composition root."""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import asdict, replace
from typing import Any

from PyQt5.QtCore import QLockFile, Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon

from .config import CaptureRegion, TrackerConfig, ensure_config, load_config, save_config
from .domain.models import AffinityResult
from .integrations.affinity import WindowsDisplayAffinityController
from .integrations.capture import MssFrameSource
from .integrations.clock import SystemClock
from .integrations.data_dragon import DataDragonClient
from .integrations.hotkeys import KeyboardHotkeyService
from .integrations.live_client import LiveClientClient
from .integrations.timeline import CsvTimelineSink
from .logging_setup import configure_logging
from .paths import AppPaths
from .tracking.detector import OpenCvChampionDetector
from .tracking.engine import TrackerEngine
from .ui.calibration import CalibrationController, RegionSelector
from .ui.overlay import TransparentOverlay
from .ui.role_icons import RoleIconRenderer
from .ui.tray import ActionBridge, TrayController


def run() -> int:
    paths = AppPaths.discover()
    paths.ensure_runtime_directories()
    logger = configure_logging(paths.log_dir)
    config_created = ensure_config(paths.config_path, logger)
    config = load_config(paths.config_path, logger)
    config_state = config
    logger.setLevel(getattr(logging, config.log_level, logging.INFO))

    QApplication.setAttribute(Qt.AA_DisableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName("LoL Minimap Tracker")
    app.setQuitOnLastWindowClosed(False)

    instance_lock = QLockFile(str(paths.lock_path))
    instance_lock.setStaleLockTime(0)
    if not instance_lock.tryLock(100):
        logger.error("Another tracker instance is already running")
        return 2

    live_client = LiveClientClient(
        config.local_api_timeout_seconds,
        logger.getChild("live_client"),
    )
    data_dragon = DataDragonClient(
        paths.cache_dir,
        logger.getChild("data_dragon"),
    )
    frame_source = MssFrameSource(config.capture)
    engine = TrackerEngine(
        config=config,
        roster_provider=live_client,
        portrait_provider=data_dragon,
        frame_source=frame_source,
        detector=OpenCvChampionDetector(config, logger.getChild("detector")),
        timeline_sink=CsvTimelineSink(paths.timeline_path),
        clock=SystemClock(),
        logger=logger.getChild("engine"),
    )

    tray: TrayController | None = None
    pending_affinity: AffinityResult | None = None

    def affinity_changed(result: AffinityResult) -> None:
        nonlocal pending_affinity
        pending_affinity = result
        logger.info("Capture exclusion: %s", result.status.value)
        if result.error_code is not None:
            logger.error("Display affinity failed with Win32 error %s", result.error_code)
        if tray is not None:
            tray.update_affinity(result)

    overlay = TransparentOverlay(
        snapshot_provider=engine.get_snapshot,
        config=config,
        icons=RoleIconRenderer(paths.role_asset_dir),
        affinity_controller=WindowsDisplayAffinityController(),
        affinity_changed=affinity_changed,
    )
    selector = RegionSelector()

    def persist(next_config: TrackerConfig) -> bool:
        nonlocal config_state
        config_state = next_config
        saved = save_config(paths.config_write_path, next_config, logger)
        if not saved and tray is not None:
            tray.notify(
                "Configuration not saved",
                "The current session was updated, but the settings file is not writable.",
                QSystemTrayIcon.Warning,
            )
        return saved

    def toggle_arrows() -> bool:
        state = overlay.toggle_arrows()
        persist(replace(config_state, show_arrows=state))
        return state

    def toggle_last_seen() -> bool:
        state = overlay.toggle_last_seen()
        persist(replace(config_state, show_last_seen=state))
        return state

    def toggle_notifications() -> bool:
        state = not config_state.show_notifications
        persist(replace(config_state, show_notifications=state))
        return state

    def open_configuration() -> None:
        if not paths.config_write_path.exists() and not save_config(
            paths.config_write_path, config_state, logger
        ):
            if tray is not None:
                tray.notify(
                    "Configuration unavailable",
                    "The configuration file could not be created.",
                    QSystemTrayIcon.Warning,
                )
            return
        os.startfile(paths.config_write_path)

    def apply_region(value: object) -> None:
        if not isinstance(value, CaptureRegion):
            logger.error("Calibration returned an invalid capture region: %r", value)
            return
        frame_source.set_region(value)
        overlay.set_capture_region(value)
        persist(replace(config_state, capture=value))
        if tray is not None:
            tray.update_region(value)
            tray.notify(
                "Minimap area updated",
                f"Capture set to {value.width} x {value.height} at {value.left}, {value.top}.",
            )

    def update_pause_action(paused: bool) -> None:
        if tray is not None:
            tray.update_action("pause", paused)

    calibration = CalibrationController(
        selector=selector,
        is_paused=engine.is_paused,
        set_paused=engine.set_paused,
        hide_overlay=overlay.hide,
        show_overlay=overlay.show,
        apply_region=apply_region,
        pause_changed=update_pause_action,
    )

    actions: dict[str, Any] = {
        "save_timeline": engine.flush_timeline,
        "quit": app.quit,
        "toggle_arrows": toggle_arrows,
        "pause": engine.toggle_pause,
        "toggle_last_seen": toggle_last_seen,
        "toggle_notifications": toggle_notifications,
        "toggle_timeline_logging": engine.toggle_timeline_logging,
        "open_configuration": open_configuration,
        "open_data_folder": lambda: os.startfile(paths.user_data_dir),
        "select_minimap": calibration.start,
    }

    def dispatch(action_name: str) -> None:
        action = actions.get(action_name)
        if action is None:
            logger.warning("Unknown action requested: %s", action_name)
            return
        try:
            state = action()
            if tray is not None and isinstance(state, bool):
                tray.update_action(action_name, state)
        except Exception:
            logger.exception("Action failed: %s", action_name)

    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = TrayController(
            app,
            dispatch,
            config,
            QIcon(str(paths.role_asset_dir / "position-middle.svg")),
        )
        if pending_affinity is not None:
            tray.update_affinity(pending_affinity)
        tray.show()
        if config_created:

            def notify_first_run() -> None:
                if tray is not None:
                    tray.notify(
                        "Minimap setup",
                        "A default configuration was created. "
                        "Select the minimap area to calibrate.",
                    )

            QTimer.singleShot(750, notify_first_run)
    else:
        logger.warning("System tray is unavailable; configured hotkeys remain active")

    bridge = ActionBridge()
    bridge.requested.connect(dispatch)
    hotkeys = KeyboardHotkeyService()
    if config.enable_global_hotkeys:
        failures = hotkeys.start(
            asdict(config.hotkeys),
            lambda name: bridge.requested.emit(name),
        )
        for failure in failures:
            logger.error("Hotkey registration failed: %s", failure)

    tracker_thread = threading.Thread(
        target=engine.run,
        name="champion-tracker",
        daemon=True,
    )
    tracker_thread.start()

    status_timer = QTimer()
    if tray is not None:
        status_timer.timeout.connect(lambda: tray.update_snapshot(engine.get_snapshot()))
        status_timer.start(500)

    cleanup_started = False

    def cleanup() -> None:
        nonlocal cleanup_started
        if cleanup_started:
            return
        cleanup_started = True
        selector.close()
        engine.stop()
        hotkeys.stop()
        tracker_thread.join(timeout=2.0)
        try:
            engine.flush_timeline()
        except OSError:
            logger.exception("Could not flush timeline during shutdown")
        instance_lock.unlock()

    app.aboutToQuit.connect(cleanup)
    overlay.show()
    logger.info("Application started")
    exit_code = app.exec_()
    cleanup()
    return int(exit_code)
