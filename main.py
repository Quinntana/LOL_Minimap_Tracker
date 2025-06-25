"""Main application for the League of Legends champion tracker."""
import sys
import os
import threading
import keyboard
from PyQt5.QtWidgets import QApplication
from config import load_config
from champion_tracker import ChampionTracker
from overlay import TransparentOverlay
from utils import setup_logger

def main():
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    os.chdir(base_path)

    logger = setup_logger("main")
    config = load_config()
    tracker = ChampionTracker(config)

    app = QApplication(sys.argv)
    overlay = TransparentOverlay(tracker.get_latest_positions,
                                tracker.get_rectangle_center, config)

    actions = {
        "save_timeline": tracker.save_timeline_to_csv,
        "quit": app.quit,
        "toggle_arrows": overlay.toggle_arrows,
        "pause": lambda: logger.info("Pause functionality not implemented yet"),
        "toggle_last_seen": overlay.toggle_last_seen,
        "toggle_timeline_logging": tracker.toggle_timeline_logging
    }

    hotkey_mappings = config.get("hotkeys", {})
    for action_name, hotkey in hotkey_mappings.items():
        if action_name in actions:
            try:
                keyboard.add_hotkey(hotkey, actions[action_name])
                logger.info(f"Set hotkey {hotkey} for {action_name}")
            except Exception as e:
                logger.error(f"Failed to set hotkey {hotkey} for {action_name}: {e}")
        else:
            logger.warning(f"Unknown action {action_name} in hotkey mappings")

    def run_tracker():
        tracker.capture_and_process()

    tracker_thread = threading.Thread(target=run_tracker, daemon=True)
    tracker_thread.start()

    overlay.show()
    logger.info("Application started")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
