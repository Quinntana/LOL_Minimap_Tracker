# League of Legends Champion Tracker

A real-time champion tracking tool that creates an overlay to visualize enemy positions on the minimap during League of Legends games.
Based on "https://github.com/realr4an/LeagueOfLegendsMinimapTracker"

## Features

- **Real-time Champion Detection**: Automatically identifies enemy champions on the minimap
- **Position Tracking**: Shows the current and last known positions of enemy champions
- **Transparent Overlay**: Non-intrusive interface that stays on top of the game window
- **Timeline Recording**: Log champion movements over time for post-game analysis
- **Configurable Settings**: Easily adjust detection parameters through config files

## Demo

![image](https://github.com/user-attachments/assets/112e207c-fa5f-4b39-966e-a466d71a251d)

## Installation

### Prerequisites

- Python 3.8 or higher
- League of Legends client

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/lol-champion-tracker.git
   cd lol-champion-tracker
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Configuration

The application uses a config file (`config.json` or `config.txt`) to customize its behavior. You can modify the following settings:

Default minimap size is 33 (1920 : 1080)

```json
{
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
        "pause": "ctrl+p",
        "toggle_last_seen": "ctrl+l",
        "toggle_timeline_logging": "ctrl+t"
    }
}
```

### Key Settings

- **Map Position**: Adjust `top`, `left`, `width`, and `height` to match your minimap's screen position
- **Detection Sensitivity**: Fine-tune `ssim_threshold` for champion recognition accuracy
- **Circle Detection**: Adjust `circle_radius_min` and `circle_radius_max` for different minimap sizes
- **Timeouts**: Modify how long champions remain displayed after disappearing

## Usage

1. Start the application before or during a League of Legends game
2. The overlay will automatically appear and track enemy champion positions
3. Use the following hotkeys to control the application:

| Hotkey | Action |
|--------|--------|
| Ctrl+S | Save timeline data to CSV |
| Ctrl+D | Quit application |
| Ctrl+A | Toggle position arrows |
| Ctrl+L | Toggle last seen indicators |
| Ctrl+T | Toggle timeline logging |
| Ctrl+P | Pause (placeholder) |

## How It Works

1. **Screen Capture**: The application captures the minimap region of your screen
2. **Image Processing**: OpenCV is used to detect champion indicators (red circles)
3. **Champion Recognition**: SSIM (Structural Similarity Index) matches detected objects to known champion images
4. **Overlay Display**: PyQt5 creates a transparent overlay showing champion positions
5. **Data Logging**: Champion positions are stored in a timeline for later analysis

## Project Structure

- `main.py`: Entry point and application initialization
- `champion_tracker.py`: Core tracking logic and position management
- `image_processor.py`: Screen capture and image analysis
- `overlay.py`: PyQt5 transparent overlay window
- `api_client.py`: League of Legends API integration
- `config.py`: Configuration loading and management
- `utils.py`: Utility functions and helper classes

## Requirements

- PyQt5
- opencv-python
- numpy
- requests
- pillow
- scikit-image
- mss
- keyboard

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is not affiliated with Riot Games and doesn't interact with the League of Legends game client in any way that violates Riot's Terms of Service. It only uses screen capture for image processing and doesn't modify game files or memory.
