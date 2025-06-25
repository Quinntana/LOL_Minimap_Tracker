"""Configuration management for the League of Legends champion tracker."""
import json
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG = {
    "top": 813,
    "left": 1655,
    "width": 252,
    "height": 252,
    "ssim_threshold": 0.3,
    "circle_radius_min": 12,
    "circle_radius_max": 40,
    "update_interval_ms": 100,
    "detection_timeout_seconds": 4.0,
    "log_level": "INFO",
    "hotkeys": {
        "save_timeline": "ctrl+s",
        "quit": "ctrl+d",
        "toggle_arrows": "ctrl+a",
        "pause": "ctrl+p",  # Placeholder for future functionality
        "toggle_last_seen" : "ctrl+l", 
        "toggle_timeline_logging": "ctrl+t"
    }
}

def load_config(file_path: str = "config.txt") -> Dict[str, Any]:
    """
    Load configuration from a file. If the file doesn't exist or has errors,
    default values are used.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration values
    """
    config = DEFAULT_CONFIG.copy()
    
    try:
        # Try to load from text file (current format)
        if Path(file_path).exists() and file_path.endswith(".txt"):
            with open(file_path, "r") as file:
                for line in file:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        try:
                            # Try to convert to appropriate type
                            if value.isdigit():
                                config[key] = int(value)
                            elif value.replace(".", "", 1).isdigit():
                                config[key] = float(value)
                            else:
                                config[key] = value
                        except ValueError:
                            print(f"Warning: Could not parse value for {key}, using default")
        
        # Try to load from JSON file (for more complex configurations)
        elif Path(file_path).exists() and file_path.endswith(".json"):
            with open(file_path, "r") as file:
                loaded_config = json.load(file)
                config.update(loaded_config)
                
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error loading config from {file_path}: {e}")
        print("Using default configuration values")
    
    return config

def save_config(config: Dict[str, Any], file_path: str = "config.json") -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Dictionary containing configuration values
        file_path: Path to save the configuration
    """
    try:
        with open(file_path, "w") as file:
            json.dump(config, file, indent=4)
        print(f"Configuration saved to {file_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")

def ensure_directories() -> None:
    """
    Ensure that necessary directories exist.
    """
    directories = ["logs", "data", "data/champion_images"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)