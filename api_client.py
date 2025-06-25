"""Client for interacting with League of Legends APIs and caching responses."""
import json
import time
import requests
import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from threading import Lock
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from utils import setup_logger, LRUCache

class LeagueApiClient:
    """Client for interacting with League of Legends APIs."""

    # Local client API
    LOCAL_API_BASE_URL = "https://127.0.0.1:2999"
    LOCAL_API_ENDPOINTS = {
        "player_list": "/liveclientdata/playerlist",
        "active_player": "/liveclientdata/activeplayer"
    }

    # Data Dragon API
    DATA_DRAGON_BASE_URL = "https://ddragon.leagueoflegends.com"
    DATA_DRAGON_ENDPOINTS = {
        "versions": "/api/versions.json",
        "champions": "/cdn/{version}/data/en_US/championFull.json",
        "champion_image": "/cdn/{version}/img/champion/{champion_key}.png"
    }

    # Request parameters
    REQUEST_TIMEOUT = 5
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(self):
        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(sys.executable)
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))

        self.data_dir = Path(base_path) / "data"

        """Initialize the League API client."""
        self.logger = setup_logger("league_api_client")
        self.cache = LRUCache(capacity=100)
        self.version: Optional[str] = None
        self.version_lock = Lock()
        self.champion_data: Optional[Dict[str, Any]] = None

        # Define paths using pathlib
        self.data_dir = Path("data").absolute()
        self.champion_images_dir = self.data_dir / "champion_images"
        self.version_file = self.data_dir / "version.txt"

        # Create data directories
        self.champion_images_dir.mkdir(parents=True, exist_ok=True)

        # Disable insecure request warnings
        requests.packages.urllib3.disable_warnings(
            requests.packages.urllib3.exceptions.InsecureRequestWarning
        )

    def _request_with_retry(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        verify: bool = True,
        cache_key: Optional[str] = None
    ) -> Optional[requests.Response]:
        """
        Make HTTP request with retries and caching.
        """
        if cache_key and method == "GET":
            cached = self.cache.get(cache_key)
            if cached:
                self.logger.debug(f"Cache hit for {cache_key}")
                return cached

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=self.REQUEST_TIMEOUT,
                    verify=verify
                )
                response.raise_for_status()

                if cache_key and method == "GET":
                    self.cache.put(cache_key, response)

                return response
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))

        self.logger.error(f"All {self.MAX_RETRIES} request attempts failed for {url}")
        return None

    def _handle_version_change(self, new_version: str) -> None:
        """Handle version changes and clean up cached data."""
        with self.version_lock:
            existing_version = None
            if self.version_file.exists():
                try:
                    existing_version = self.version_file.read_text().strip()
                except Exception as e:
                    self.logger.error(f"Error reading version file: {e}")

            if existing_version == new_version:
                return

            self.logger.info(f"Detected version change ({existing_version or 'none'} → {new_version}). Cleaning cache...")
            for filename in self.champion_images_dir.iterdir():
                if filename.is_file():
                    try:
                        filename.unlink()
                        self.logger.debug(f"Deleted cached image: {filename}")
                    except Exception as e:
                        self.logger.error(f"Error deleting {filename}: {e}")

            try:
                self.version_file.write_text(new_version)
                self.logger.info(f"Updated version to {new_version}")
            except Exception as e:
                self.logger.error(f"Failed to update version file: {e}")

    def get_latest_version(self) -> Optional[str]:
        """
        Get the latest game version from Data Dragon API.
        """
        if self.version:
            return self.version

        url = f"{self.DATA_DRAGON_BASE_URL}{self.DATA_DRAGON_ENDPOINTS['versions']}"
        response = self._request_with_retry(url, cache_key="latest_version")

        if not response:
            return None

        try:
            versions = response.json()
            new_version = versions[0]
            self._handle_version_change(new_version)
            self.version = new_version
            return self.version
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            self.logger.error(f"Failed to parse version data: {e}")
            return None

    def get_champion_data(self) -> Optional[Dict[str, Any]]:
        """
        Get champion data from Data Dragon API.
        """
        if self.champion_data:
            return self.champion_data

        version = self.get_latest_version()
        if not version:
            return None

        url = f"{self.DATA_DRAGON_BASE_URL}{self.DATA_DRAGON_ENDPOINTS['champions'].format(version=version)}"
        response = self._request_with_retry(url, cache_key=f"champion_data_{version}")

        if not response:
            return None

        try:
            self.champion_data = response.json()
            return self.champion_data
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse champion data: {e}")
            return None

    def normalize_champion_name(self, champion_name: str) -> Optional[str]:
        """
        Convert champion display name to internal key.
        """
        champion_data = self.get_champion_data()
        if not champion_data:
            return None

        if champion_name in champion_data.get("data", {}):
            return champion_name

        for key, data in champion_data.get("data", {}).items():
            if data.get("name", "").lower() == champion_name.lower():
                return key

        self.logger.warning(f"Could not normalize champion name: {champion_name}")
        return None

    def get_champion_image(self, champion_name: str) -> Optional[np.ndarray]:
        """
        Get champion image from Data Dragon API or local cache.
        """
        cache_path = self.champion_images_dir / f"{champion_name}.png"
        if cache_path.exists():
            try:
                img = cv2.imread(str(cache_path))
                if img is not None:
                    return img
            except Exception as e:
                self.logger.warning(f"Error loading cached image for {champion_name}: {e}")

        version = self.get_latest_version()
        if not version:
            return None

        champion_key = self.normalize_champion_name(champion_name)
        if not champion_key:
            return None

        url = f"{self.DATA_DRAGON_BASE_URL}{self.DATA_DRAGON_ENDPOINTS['champion_image'].format(version=version, champion_key=champion_key)}"
        response = self._request_with_retry(url)

        if not response:
            return None

        try:
            pil_img = Image.open(BytesIO(response.content))
            img_array = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(cache_path), img_bgr)

            return img_bgr
        except Exception as e:
            self.logger.error(f"Failed to process image for {champion_name}: {e}")
            return None

    def get_player_list(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get player list from local League client API.
        """
        url = f"{self.LOCAL_API_BASE_URL}{self.LOCAL_API_ENDPOINTS['player_list']}"
        response = self._request_with_retry(url, verify=False)

        if not response:
            return None

        try:
            return response.json()
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse player list: {e}")
            return None

    def get_active_player(self) -> Optional[Dict[str, Any]]:
        """
        Get active player data from local League client API.
        """
        url = f"{self.LOCAL_API_BASE_URL}{self.LOCAL_API_ENDPOINTS['active_player']}"
        response = self._request_with_retry(url, verify=False)

        if not response:
            return None

        try:
            return response.json()
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse active player data: {e}")
            return None

    def get_opponent_champions(self) -> List[str]:
        """
        Get list of opponent champion names.
        """
        players = self.get_player_list()
        active_player = self.get_active_player()

        if not players or not active_player or "riotId" not in active_player:
            return []

        my_team = next(
            (p.get("team") for p in players if p.get("riotId") == active_player["riotId"]),
            None
        )

        if not my_team:
            return []

        opponent_champions = [
            p["championName"] for p in players
            if p.get("team") != my_team and "championName" in p
        ]

        return opponent_champions

    def load_champion_images(self, champion_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Load images for specified champions.
        """
        results = {}
        for name in champion_names:
            img = self.get_champion_image(name)
            if img is not None:
                results[name] = img

        return results
