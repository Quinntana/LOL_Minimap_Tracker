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
color is used for the champion name outside the minimap and the selected last-seen
marker. Missing enemies with a known prior position use a compact faded champion
portrait with a centered red X by default. A standalone tinted role icon and a minimal
color dot fallback are available as isolated manual alternatives.

Direction arrows encode camera-relative range without relying on a number:

- Red, amber, or green carries the distance warning for both live and last-seen positions.
- Solid means currently detected; dashed means the arrow uses a last-seen position.

## Real-time analysis safeguards

The live detector uses two independent confidence stages. OpenCV first rejects
portrait matches that do not clear the SSIM threshold or are too close to the
second-best portrait, and keeps at most one observation per champion per frame.
The tracker then requires spatially consistent observations across consecutive
frames before publishing a position. Large coordinate jumps require an additional
confirming frame, reducing false last-seen markers. Red marker candidates are
isolated in HSV after the captured BGRA frame is converted to BGR; portrait matching
uses the icon center so the red rim and minimap background do not dominate SSIM.

The tray reports rolling frame rate, processing time, portrait count, detected
circles, accepted matches, stale frames, Live Client interruptions, and
capture/detection failures. Repeated capture failures cause a user-level capture
restart with backoff; the executable never requests elevation.
The confidence, movement, health, and recovery values are configurable in
`config.json`:

- `capture_backend`: `league_window` (default) or explicit `desktop_mss` fallback.
- `capture_region_space`: `screen` for legacy/global coordinates or `client` for a
  region that follows the League window.
- `league_process_name`: exact game executable used for window discovery.
- `window_capture_timeout_seconds`: maximum wait for a fresh game-window frame.
- `last_seen_marker_style`: persisted manual choice of `portrait`, `role`, or `dot`.
- `ssim_margin`: required separation from the second-best portrait score.
- `confirmation_frames`: normal consecutive-frame requirement.
- `confirmation_position_tolerance_pixels`: maximum movement within a confirmation run.
- `jump_confirmation_frames`: consecutive frames required for an implausible jump.
- `max_position_jump_pixels` and `max_position_speed_pixels_per_second`: movement gate.
- `health_stale_after_seconds`: delayed-frame warning threshold.
- `capture_recovery_failure_count` and `capture_recovery_backoff_seconds`: MSS recovery.
- `cooldown_tracker_enabled`: opt-in private clickable base-cooldown panel; disabled
  by default.

## Private cooldown research panel

The development-only cooldown panel is a separate compact, interactive window; the
minimap overlay remains fully click-through. It displays each enemy champion,
ultimate, and two summoner-spell icons. Left-click starts or restarts a timer and
right-click clears it. Enemy level selects the ordinary ultimate rank at levels
6/11/16, with explicit handling for supported nonstandard rank layouts. Summoner
spells use their patch-static base cooldown.

These timers are deliberately approximate. They ignore ability and summoner haste,
runes, items, resets, refunds, charges, and mode-specific modifiers. Unknown,
zero-cooldown, non-inferable, and charge-based entries such as Smite are disabled
instead of guessed. Manual transitions are recorded in `cooldown-events.csv` for
research playback. The panel is disabled by default and intended only for the private
development scope documented in `docs/cooldown-tracker-investigation.md`.

The default `league_window` backend uses Windows Graphics Capture to target the
visible window owned by `League of Legends.exe`. Only that game window is captured,
so the separate overlay window cannot feed back into detection. Window capture
requires Windows 10 version 1903 or newer. The older composed-desktop MSS backend
remains available only through the explicit `desktop_mss` setting.

For desktop capture, the overlay requests `WDA_EXCLUDEFROMCAPTURE` on Windows 10
version 2004 and newer. This is a best-effort Windows feature, not a security
guarantee; if affinity cannot be verified, larger graphics inside the minimap are
disabled.
Portrait and role markers remain disabled in that state, but the user can explicitly
select the five-pixel identity-color dot fallback. The tracker never changes marker
style automatically. Dot colors stay outside the detector's red hue bands and the
dot is well below the normal circle-radius threshold, minimizing recapture feedback.
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

On first run, the portable executable creates an editable configuration at
`%LOCALAPPDATA%\LoLMinimapTracker\config.json`. A `config.json` placed beside the
executable takes precedence, preserving fully portable setups. Legacy JSON stored
in `config.txt` remains readable but is deprecated and is never rewritten.

Use `Select minimap area...` from the notification-area menu to calibrate without
restarting. Drag around the full minimap and release; the selector stays square in
every drag direction. Detection pauses while the selection surface is open, then
returns to its previous state. The selected global coordinates apply immediately to
capture and rendering and are saved atomically.

The primary menu is focused on live play:

- `Direction arrows`: red-to-green distance color for every arrow; solid means currently
  detected and dashed means last seen.
- `Missing-enemy markers`: shows or hides last confirmed missing-enemy positions.
- `Missing marker`: manually selects champion portrait + X (default), role icon, or
  minimal identity-color dot. Rich modes require isolated capture; dot is the
  desktop-capture fallback.
- `Pause detection`: stops new capture and analysis until resumed.
- `Select minimap area...`: recalibrates the capture rectangle across all displays.

Timeline recording, configuration access, and the data folder are under `Advanced`.
Left-clicking the notification-area icon opens the same menu as right-clicking it.
The overlay uses native top-level input transparency and never accepts focus, so all
map markers remain click-through.
Arrow, marker visibility/style, notification, and capture-region preferences persist
between runs.

Runtime data is written to `%LOCALAPPDATA%\LoLMinimapTracker`:

- `logs/tracker.log`: rotating diagnostics.
- `cache/ddragon`: versioned metadata and champion portraits.
- `timeline.csv`: explicitly recorded movement data.
- `cooldown-events.csv`: manually clicked cooldown transitions for research playback.
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
