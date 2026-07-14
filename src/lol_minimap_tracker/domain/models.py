"""Immutable application models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    TOP = "TOP"
    JUNGLE = "JUNGLE"
    MIDDLE = "MIDDLE"
    BOTTOM = "BOTTOM"
    UTILITY = "UTILITY"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_api(cls, value: object) -> Role:
        aliases = {
            "TOP": cls.TOP,
            "JUNGLE": cls.JUNGLE,
            "MIDDLE": cls.MIDDLE,
            "MID": cls.MIDDLE,
            "BOTTOM": cls.BOTTOM,
            "BOT": cls.BOTTOM,
            "UTILITY": cls.UTILITY,
            "SUPPORT": cls.UTILITY,
        }
        return aliases.get(str(value or "").upper(), cls.UNKNOWN)


class RosterStatus(StrEnum):
    ACTIVE = "active"
    UNAVAILABLE = "unavailable"
    INVALID_RESPONSE = "invalid_response"


class TrackerMode(StrEnum):
    WAITING = "waiting"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"


class AffinityStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class LastSeenMarkerStyle(StrEnum):
    PORTRAIT = "portrait"
    ROLE = "role"
    DOT = "dot"


class ArrowDisplayMode(StrEnum):
    NEARBY = "nearby"
    ALL = "all"


class AnalysisStatus(StrEnum):
    WARMING_UP = "warming_up"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STALLED = "stalled"


@dataclass(frozen=True)
class SummonerSpellRef:
    identifier: str | None
    display_name: str


@dataclass(frozen=True)
class RosterMember:
    champion_name: str
    role: Role
    participant_id: str = ""
    champion_id: str | None = None
    level: int | None = None
    summoner_spells: tuple[SummonerSpellRef, ...] = ()


@dataclass(frozen=True)
class RosterState:
    generation: int = 0
    members: tuple[RosterMember, ...] = ()


@dataclass(frozen=True)
class RosterResult:
    status: RosterStatus
    members: tuple[RosterMember, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class EnemyIdentity:
    champion_name: str
    role: Role
    color: str
    role_icon: str


@dataclass(frozen=True)
class ChampionObservation:
    champion_name: str
    x: int
    y: int
    score: float


@dataclass(frozen=True)
class DetectionDiagnostics:
    portraits: int = 0
    circles: int = 0
    accepted: int = 0
    below_threshold: int = 0
    ambiguous: int = 0
    duplicate: int = 0
    best_score: float = 0.0
    best_margin: float = 0.0


@dataclass(frozen=True)
class DetectionFrame:
    observations: tuple[ChampionObservation, ...]
    camera_center: tuple[int, int] | None
    diagnostics: DetectionDiagnostics = DetectionDiagnostics()


@dataclass(frozen=True)
class RuntimeHealth:
    status: AnalysisStatus = AnalysisStatus.WARMING_UP
    frames_per_second: float = 0.0
    processing_ms: float = 0.0
    frame_age_seconds: float | None = None
    api_age_seconds: float | None = None
    api_failures: int = 0
    consecutive_failures: int = 0
    ambiguous_rejections: int = 0
    duplicate_rejections: int = 0
    motion_deferrals: int = 0
    pending_confirmations: int = 0
    portraits: int = 0
    detected_circles: int = 0
    accepted_matches: int = 0
    below_threshold: int = 0
    best_match_score: float = 0.0
    best_match_margin: float = 0.0
    last_error: str | None = None
    message: str = "Waiting for live frames"


@dataclass(frozen=True)
class ChampionView:
    identity: EnemyIdentity
    position: tuple[int, int] | None
    is_current: bool
    seconds_since_seen: float | None


@dataclass(frozen=True)
class TrackerSnapshot:
    mode: TrackerMode = TrackerMode.WAITING
    roster_status: RosterStatus = RosterStatus.UNAVAILABLE
    champions: tuple[ChampionView, ...] = ()
    camera_center: tuple[int, int] | None = None
    timeline_logging: bool = False
    message: str = "Waiting for a game"
    health: RuntimeHealth = RuntimeHealth()


@dataclass(frozen=True)
class AffinityResult:
    status: AffinityStatus
    error_code: int | None = None


PortraitMap = Mapping[str, object]
