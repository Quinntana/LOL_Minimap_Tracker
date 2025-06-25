"""Tracks champion positions and states over time."""
import time
import csv
import numpy as np  
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from api_client import LeagueApiClient
from image_processor import ImageProcessor
from utils import setup_logger

class ChampionTracker:
    """Manages champion tracking and timeline recording."""
    
    def __init__(self, config: Dict):
        """Initialize tracker with configuration."""
        self.logger = setup_logger("champion_tracker")
        self.config = config
        self.api_client = LeagueApiClient()
        self.image_processor = ImageProcessor(config)
        self.champion_images: Dict[str, np.ndarray] = {}
        self.opponent_champions: List[str] = []
        self.latest_positions: Dict[str, Dict] = {}
        self.rectangle_center: Tuple[int, int] = (0, 0)
        self.timeline_positions: List[Dict] = []
        self.last_save_time = 0
        self.last_detected_times: Dict[str, float] = {}
        self.current_detection_timeout = config.get("detection_timeout_seconds", 5.0)
        self.timeline_logging_enabled = False

    def fetch_opponent_champions(self):
        """Fetch and load opponent champion data."""
        self.opponent_champions = self.api_client.get_opponent_champions()
        self.champion_images = {champ: self.api_client.get_champion_image(champ) 
                               for champ in self.opponent_champions}

    def update_positions(self, champions: List[Tuple[str, int, int, float]]):
        current_time = time.time()
        detected_champs = set()
        for name, x, y, _ in champions:
            self.latest_positions[name] = {"X": x, "Y": y}
            self.last_detected_times[name] = current_time
            detected_champs.add(name)

    def save_to_timeline(self):
        """Save current positions to timeline if logging is enabled."""
        if not self.timeline_logging_enabled:  # Check the flag
            return
        current_time = time.time()
        if current_time - self.last_save_time < 1:
            return
        for champ, pos in self.latest_positions.items():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.timeline_positions.append({
                "timestamp": timestamp,
                "champion": champ,
                "X": pos["X"],
                "Y": pos["Y"]
            })
        self.last_save_time = current_time

    def toggle_timeline_logging(self):
        """Toggle timeline logging on or off."""
        self.timeline_logging_enabled = not self.timeline_logging_enabled
        status = "enabled" if self.timeline_logging_enabled else "disabled"
        self.logger.info(f"Timeline logging {status}")

    def save_timeline_to_csv(self, file_path: str = "data/timeline.csv"):
        """Save timeline data to CSV."""
        if not self.timeline_positions:
            self.logger.info("No timeline data to save.")
            return
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        file_exists = Path(file_path).is_file()
        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "champion", "X", "Y"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.timeline_positions)
        self.logger.info(f"Timeline saved to {file_path}")
        self.timeline_positions.clear()

    def get_latest_positions(self) -> List[Dict]:
        """Get current champion positions and status."""
        current_time = time.time()
        return [
            {
                "Champion": champ,
                "Position": self.latest_positions.get(champ),
                "IsCurrent": current_time - self.last_detected_times.get(champ, 0) < self.current_detection_timeout
            }
            for champ in self.opponent_champions
        ]

    def get_rectangle_center(self) -> Tuple[int, int]:
        """Get the current rectangle center."""
        return self.rectangle_center

    def capture_and_process(self):
        """Main loop to capture and process screen data."""
        self.fetch_opponent_champions()
        while True:
            img = self.image_processor.capture_screen()
            self.rectangle_center = self.image_processor.find_rectangle_center(img)
            champions = self.image_processor.detect_champions(img, self.champion_images)
            self.update_positions(champions)
            self.save_to_timeline()