"""League of Legends minimap tracker research package."""

from .config import CaptureRegion, HotkeyConfig, TrackerConfig
from .domain.models import (
    AnalysisStatus,
    ChampionObservation,
    ChampionView,
    DetectionDiagnostics,
    EnemyIdentity,
    Role,
    RosterResult,
    RosterStatus,
    RuntimeHealth,
    TrackerMode,
    TrackerSnapshot,
)

__all__ = [
    "AnalysisStatus",
    "CaptureRegion",
    "ChampionObservation",
    "ChampionView",
    "DetectionDiagnostics",
    "EnemyIdentity",
    "HotkeyConfig",
    "Role",
    "RosterResult",
    "RosterStatus",
    "RuntimeHealth",
    "TrackerConfig",
    "TrackerMode",
    "TrackerSnapshot",
]
