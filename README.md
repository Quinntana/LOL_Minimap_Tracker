# League of Legends Minimap Tracker

A private Windows research project that detects enemy markers in a captured League
of Legends minimap region and displays a click-through overlay.

## Research and policy notice

This project is not affiliated with, endorsed by, or approved by Riot Games. Riot's
current game-integrity policy restricts products that expose game-session-specific
information or create a competitive advantage. Review the
[League of Legends developer policy](https://developer.riotgames.com/docs/lol)
before using or distributing this research software.

The tracker reads pixels already rendered on the local display and the documented
Live Client Data API. It does not modify game files or process memory.

## Overlay identity

Every enemy receives a stable, colorblind-conscious color for the match. The same
color is used for the champion name outside the minimap and the hollow last-seen
ring. The ring contains a faint role icon for Top, Jungle, Mid, Bot, or Support.
Names and portraits are never drawn over the minimap.

Direction arrows keep their original state colors:

- Red: currently detected.
- Yellow: last-seen position.

## Real-time analysis safeguards

The live detector uses two independent confidence stages. OpenCV first rejects
portrait matches that do not clear the SSIM threshold or are too close to the
second-best portrait, and keeps at most one observation per champion per frame.
The tracker then requires spatially consistent observations across consecutive
frames before publishing a position. Large coordinate jumps require an additional
confirming frame, reducing false last-seen markers without changing the original
BGR mask or red/yellow arrow colors.

The tray reports rolling frame rate, processing time, stale frames, Live Client
interruptions, and capture/detection failures. Repeated MSS capture failures cause
a user-level capture restart with backoff; the executable never requests elevation.
The confidence, movement, health, and recovery values are configurable in
`config.json`:

- `ssim_margin`: required separation from the second-best portrait score.
- `confirmation_frames`: normal consecutive-frame requirement.
- `confirmation_position_tolerance_pixels`: maximum movement within a confirmation run.
- `jump_confirmation_frames`: consecutive frames required for an implausible jump.
- `max_position_jump_pixels` and `max_position_speed_pixels_per_second`: movement gate.
- `health_stale_after_seconds`: delayed-frame warning threshold.
- `capture_recovery_failure_count` and `capture_recovery_backoff_seconds`: MSS recovery.

On Windows 10 version 2004 and newer, the overlay requests
`WDA_EXCLUDEFROMCAPTURE`, which keeps it visible on the monitor while supported
capture APIs omit it. This is a best-effort Windows feature, not a security
guarantee. If affinity cannot be verified, graphics inside the minimap are disabled.
See [Microsoft's display-affinity documentation](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowdisplayaffinity).

## Development

Python 3.11 or newer is supported. The known-good Windows build uses Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\ruff format --check src tests tools main.py
.\.venv\Scripts\ruff check src tests tools main.py
.\.venv\Scripts\mypy src
.\.venv\Scripts\pytest
```

The tests use fake Riot responses, synthetic images, mocked Win32 calls, and Qt's
offscreen platform. League of Legends and administrator access are not required.

## Windows build

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_exe.ps1
```

The build installs `requirements.lock`, validates the source, creates
`dist/LoLMinimapTracker.exe`, copies `config.json`, inspects the embedded archive,
and prints a SHA-256 hash. It never launches the executable.

The executable is unsigned, so Windows SmartScreen or endpoint-security software
may restrict it. The project does not request elevation and has no installer.

## Configuration and data

Edit `config.json` beside `main.py` or the portable executable. Legacy JSON stored
in `config.txt` remains readable but is deprecated.

Runtime data is written to `%LOCALAPPDATA%\LoLMinimapTracker`:

- `logs/tracker.log`: rotating diagnostics.
- `cache/ddragon`: versioned metadata and champion portraits.
- `timeline.csv`: explicitly recorded movement data.
- `tracker.lock`: user-level single-instance lock.

Global hotkeys are enabled by default and never request administrator access. The
system-tray menu remains available if registration is blocked.

| Hotkey | Action |
| --- | --- |
| `Ctrl+S` | Save buffered timeline data |
| `Ctrl+D` | Quit |
| `Ctrl+A` | Toggle direction arrows |
| `Ctrl+L` | Toggle last-seen markers |
| `Ctrl+T` | Toggle timeline recording |
| `Ctrl+P` | Pause or resume tracking |

## External contracts

The application uses Riot's documented Live Client endpoints:

- `https://127.0.0.1:2999/liveclientdata/activeplayername`
- `https://127.0.0.1:2999/liveclientdata/playerlist`

Run `python tools/verify_ddragon.py` to verify the current Data Dragon version,
champion metadata, and one portrait without modifying local data.

Role SVGs are vendored from
[CommunityDragon](https://communitydragon.org/documentation/assets), which operates
under Riot Games' Legal Jibber Jabber policy. Riot Games owns the underlying assets
and does not endorse or sponsor CommunityDragon or this project. Source URLs and
SHA-256 hashes are recorded in `src/lol_minimap_tracker/assets/roles/manifest.json`.

## Optional timeline analysis

```powershell
python -m pip install -e ".[plot]"
python -m lol_minimap_tracker.analysis data\timeline.csv --map data\Minimap.png
```
