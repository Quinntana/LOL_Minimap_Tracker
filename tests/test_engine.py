from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from lol_minimap_tracker.config import TrackerConfig
from lol_minimap_tracker.domain.models import (
    AnalysisStatus,
    ChampionObservation,
    DetectionDiagnostics,
    DetectionFrame,
    Role,
    RosterMember,
    RosterResult,
    RosterStatus,
    TrackerMode,
)
from lol_minimap_tracker.tracking.engine import FrameProcessingError, TrackerEngine


class Clock:
    now = 0.0

    def monotonic(self) -> float:
        return self.now

    def timestamp(self) -> str:
        return "2026-07-11 12:00:00"


class Rosters:
    def __init__(self, results: list[RosterResult]) -> None:
        self.results = results

    def poll(self) -> RosterResult:
        return self.results.pop(0)


class Portraits:
    def get_portraits(self, members: tuple[RosterMember, ...]) -> dict[str, np.ndarray[Any, Any]]:
        return {member.champion_name: np.zeros((10, 10, 3), dtype=np.uint8) for member in members}


class Frames:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True

    def capture(self) -> np.ndarray[Any, Any]:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def close(self) -> None:
        self.started = False


class Detector:
    def __init__(self, frame: DetectionFrame) -> None:
        self.frame = frame

    def process(self, *_args: Any) -> DetectionFrame:
        return self.frame


class Timeline:
    def __init__(self) -> None:
        self.records: list[Any] = []
        self.flushed = False

    def record(self, timestamp: str, champions: tuple[Any, ...]) -> None:
        self.records.append((timestamp, champions))

    def flush(self) -> None:
        self.flushed = True


def active(name: str = "Aatrox") -> RosterResult:
    return RosterResult(
        RosterStatus.ACTIVE,
        (RosterMember(name, Role.TOP),),
    )


def make_engine(rosters: list[RosterResult]) -> tuple[TrackerEngine, Clock, Timeline]:
    clock = Clock()
    timeline = Timeline()
    engine = TrackerEngine(
        TrackerConfig(),
        Rosters(rosters),
        Portraits(),
        Frames(),
        Detector(DetectionFrame((ChampionObservation("Aatrox", 12, 34, 0.9),), (50, 50))),
        timeline,
        clock,
        logging.getLogger("test"),
    )
    return engine, clock, timeline


def test_roster_survives_two_failures_and_expires_on_third() -> None:
    unavailable = RosterResult(RosterStatus.UNAVAILABLE, error="no game")
    engine, _, _ = make_engine([active(), unavailable, unavailable, unavailable])
    engine.poll_roster()
    assert engine.get_snapshot().mode is TrackerMode.ACTIVE
    engine.poll_roster()
    engine.poll_roster()
    assert len(engine.get_snapshot().champions) == 1
    engine.poll_roster()
    assert engine.get_snapshot().mode is TrackerMode.WAITING
    assert not engine.get_snapshot().champions


def test_observation_becomes_last_seen_and_timeline_records() -> None:
    engine, clock, timeline = make_engine([active()])
    engine.poll_roster()
    assert engine.toggle_timeline_logging()
    clock.now = 1.0
    engine.process_frame()
    assert engine.get_snapshot().champions[0].position is None
    clock.now = 1.1
    engine.process_frame()
    current = engine.get_snapshot().champions[0]
    assert current.is_current
    assert current.position == (12, 34)
    assert len(timeline.records) == 1
    clock.now = 6.0
    stale = engine.get_snapshot().champions[0]
    assert not stale.is_current
    assert stale.seconds_since_seen == 4.9


def test_pause_and_flush_actions() -> None:
    engine, _, timeline = make_engine([active()])
    engine.poll_roster()
    assert engine.toggle_pause()
    assert engine.get_snapshot().mode is TrackerMode.PAUSED
    assert not engine.toggle_pause()
    assert engine.set_paused(True)
    assert engine.is_paused()
    assert not engine.set_paused(False)
    assert not engine.is_paused()
    engine.flush_timeline()
    assert timeline.flushed


def test_worker_lifecycle_starts_processes_and_closes() -> None:
    roster = Rosters([active()])
    frames = Frames()
    clock = Clock()
    timeline = Timeline()
    engine: TrackerEngine

    class StoppingDetector(Detector):
        def process(self, *_args: Any) -> DetectionFrame:
            engine.stop()
            return self.frame

    engine = TrackerEngine(
        TrackerConfig(),
        roster,
        Portraits(),
        frames,
        StoppingDetector(DetectionFrame((ChampionObservation("Aatrox", 1, 2, 0.9),), None)),
        timeline,
        clock,
        logging.getLogger("test"),
    )
    engine.run()
    assert not frames.started
    assert engine.get_snapshot().mode is TrackerMode.STOPPING


def test_missing_portrait_is_retried_without_roster_change() -> None:
    class FlakyPortraits(Portraits):
        def __init__(self) -> None:
            self.calls = 0

        def get_portraits(
            self, members: tuple[RosterMember, ...]
        ) -> dict[str, np.ndarray[Any, Any]]:
            self.calls += 1
            if self.calls == 1:
                return {}
            return super().get_portraits(members)

    portraits = FlakyPortraits()
    engine = TrackerEngine(
        TrackerConfig(),
        Rosters([active(), active()]),
        portraits,
        Frames(),
        Detector(DetectionFrame((), None)),
        Timeline(),
        Clock(),
        logging.getLogger("test"),
    )
    engine.poll_roster()
    engine.poll_roster()
    assert portraits.calls == 2


def test_new_roster_clears_old_positions_and_identity() -> None:
    engine, _, _ = make_engine(
        [
            active("Aatrox"),
            RosterResult(
                RosterStatus.ACTIVE,
                (RosterMember("Nami", Role.UTILITY),),
            ),
        ]
    )
    engine.poll_roster()
    engine.process_frame()
    engine.process_frame()
    assert engine.get_snapshot().champions[0].position == (12, 34)
    engine.poll_roster()
    replacement = engine.get_snapshot().champions[0]
    assert replacement.identity.champion_name == "Nami"
    assert replacement.identity.role is Role.UTILITY
    assert replacement.position is None


def test_unknown_detection_is_ignored() -> None:
    engine, _, _ = make_engine([active()])
    engine.detector = Detector(
        DetectionFrame((ChampionObservation("Unknown", 9, 9, 0.99),), (10, 10))
    )
    engine.poll_roster()
    engine.process_frame()
    champion = engine.get_snapshot().champions[0]
    assert champion.position is None
    assert engine.get_snapshot().camera_center == (10, 10)


def test_invalid_roster_response_uses_same_grace_period() -> None:
    invalid = RosterResult(RosterStatus.INVALID_RESPONSE, error="schema drift")
    engine, _, _ = make_engine([active(), invalid, invalid, invalid])
    engine.poll_roster()
    engine.poll_roster()
    engine.poll_roster()
    assert engine.get_snapshot().mode is TrackerMode.ACTIVE
    assert engine.get_snapshot().roster_status is RosterStatus.INVALID_RESPONSE
    engine.poll_roster()
    assert engine.get_snapshot().mode is TrackerMode.WAITING


def test_live_client_interruption_is_visible_in_runtime_health() -> None:
    unavailable = RosterResult(RosterStatus.UNAVAILABLE, error="endpoint unavailable")
    engine, clock, _ = make_engine([active(), unavailable])
    engine.poll_roster()
    engine.process_frame()
    clock.now = 0.1
    engine.process_frame()
    clock.now = 0.2
    engine.poll_roster()

    health = engine.get_snapshot().health
    assert health.status is AnalysisStatus.DEGRADED
    assert health.api_failures == 1
    assert health.api_age_seconds == 0.2
    assert health.message == "endpoint unavailable"


def test_stopping_while_waiting_closes_frame_source() -> None:
    frames = Frames()
    clock = Clock()
    engine: TrackerEngine

    class StopOnPoll(Rosters):
        def poll(self) -> RosterResult:
            engine.stop()
            return RosterResult(RosterStatus.UNAVAILABLE)

    engine = TrackerEngine(
        TrackerConfig(),
        StopOnPoll([]),
        Portraits(),
        frames,
        Detector(DetectionFrame((), None)),
        Timeline(),
        clock,
        logging.getLogger("test"),
    )
    engine.run()
    assert not frames.started
    assert engine.get_snapshot().mode is TrackerMode.STOPPING


def test_single_frame_false_positive_and_interrupted_confirmation_are_rejected() -> None:
    engine, clock, _ = make_engine([active()])
    detector = Detector(DetectionFrame((ChampionObservation("Aatrox", 12, 34, 0.9),), None))
    engine.detector = detector
    engine.poll_roster()

    engine.process_frame()
    assert engine.get_snapshot().champions[0].position is None
    assert engine.get_snapshot().health.pending_confirmations == 1

    clock.now = 0.1
    detector.frame = DetectionFrame((), None)
    engine.process_frame()
    assert engine.get_snapshot().health.pending_confirmations == 0

    clock.now = 0.2
    detector.frame = DetectionFrame((ChampionObservation("Aatrox", 12, 34, 0.9),), None)
    engine.process_frame()
    assert engine.get_snapshot().champions[0].position is None


def test_large_position_jump_needs_stricter_confirmation() -> None:
    engine, clock, _ = make_engine([active()])
    detector = Detector(DetectionFrame((ChampionObservation("Aatrox", 10, 10, 0.9),), None))
    engine.detector = detector
    engine.poll_roster()
    engine.process_frame()
    clock.now = 0.1
    engine.process_frame()
    assert engine.get_snapshot().champions[0].position == (10, 10)

    detector.frame = DetectionFrame((ChampionObservation("Aatrox", 100, 100, 0.9),), None)
    clock.now = 0.2
    engine.process_frame()
    clock.now = 0.3
    engine.process_frame()
    deferred = engine.get_snapshot()
    assert deferred.champions[0].position == (10, 10)
    assert deferred.health.motion_deferrals == 2

    clock.now = 0.4
    engine.process_frame()
    assert engine.get_snapshot().champions[0].position == (100, 100)


def test_runtime_health_reports_detector_diagnostics_and_stale_frames() -> None:
    engine, clock, _ = make_engine([active()])
    engine.detector = Detector(
        DetectionFrame(
            (),
            None,
            DetectionDiagnostics(
                portraits=5,
                circles=3,
                accepted=0,
                below_threshold=1,
                ambiguous=2,
                duplicate=1,
                best_score=0.27,
                best_margin=0.01,
            ),
        )
    )
    engine.poll_roster()
    engine.process_frame()
    clock.now = 0.1
    engine.process_frame()
    healthy = engine.get_snapshot().health
    assert healthy.status is AnalysisStatus.HEALTHY
    assert healthy.frames_per_second == 10.0
    assert healthy.ambiguous_rejections == 4
    assert healthy.duplicate_rejections == 2
    assert healthy.portraits == 5
    assert healthy.detected_circles == 3
    assert healthy.accepted_matches == 0
    assert healthy.below_threshold == 1
    assert healthy.best_match_score == 0.27
    assert "0/3 matched" in healthy.message

    clock.now = 1.2
    assert engine.get_snapshot().health.status is AnalysisStatus.DEGRADED
    clock.now = 3.2
    assert engine.get_snapshot().health.status is AnalysisStatus.STALLED


def test_capture_startup_failures_are_visible_before_the_first_frame() -> None:
    engine, _, _ = make_engine([active()])

    class UnavailableFrames(Frames):
        def capture(self) -> np.ndarray[Any, Any]:
            raise OSError("League game window not found")

    engine.frame_source = UnavailableFrames()
    engine.poll_roster()

    with pytest.raises(FrameProcessingError):
        engine.process_frame()
    degraded = engine.get_snapshot().health
    assert degraded.status is AnalysisStatus.DEGRADED
    assert "League game window not found" in degraded.message

    for _ in range(2):
        with pytest.raises(FrameProcessingError):
            engine.process_frame()
    stalled = engine.get_snapshot().health
    assert stalled.status is AnalysisStatus.STALLED
    assert "League game window not found" in stalled.message


def test_repeated_capture_failures_restart_source_and_recover() -> None:
    class RecoveringFrames(Frames):
        def __init__(self) -> None:
            super().__init__()
            self.start_calls = 0
            self.capture_calls = 0

        def start(self) -> None:
            super().start()
            self.start_calls += 1

        def capture(self) -> np.ndarray[Any, Any]:
            self.capture_calls += 1
            if self.capture_calls <= 3:
                raise OSError("capture unavailable")
            return super().capture()

    frames = RecoveringFrames()
    clock = Clock()
    engine: TrackerEngine

    class StopAfterRecovery(Detector):
        def process(self, *_args: Any) -> DetectionFrame:
            engine.stop()
            return self.frame

    engine = TrackerEngine(
        TrackerConfig(capture_recovery_backoff_seconds=0.0),
        Rosters([active()]),
        Portraits(),
        frames,
        StopAfterRecovery(DetectionFrame((), None)),
        Timeline(),
        clock,
        logging.getLogger("test"),
    )
    engine.run()
    assert frames.capture_calls == 4
    assert frames.start_calls == 2
    assert not frames.started
