"""Threaded tracker state machine with real-time confidence filtering."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from math import hypot
from threading import Event, RLock

from ..config import TrackerConfig
from ..domain.identity import assign_identities
from ..domain.interfaces import (
    ChampionDetector,
    Clock,
    FrameSource,
    Image,
    PortraitProvider,
    RosterProvider,
    TimelineSink,
)
from ..domain.models import (
    AnalysisStatus,
    ChampionObservation,
    ChampionView,
    DetectionFrame,
    EnemyIdentity,
    RosterMember,
    RosterStatus,
    RuntimeHealth,
    TrackerMode,
    TrackerSnapshot,
)


@dataclass
class _PendingObservation:
    position: tuple[int, int]
    count: int
    last_frame: int
    required_frames: int


class FrameProcessingError(RuntimeError):
    def __init__(self, phase: str, cause: Exception) -> None:
        super().__init__(f"{phase} failed: {cause}")
        self.phase = phase
        self.cause = cause


class TrackerEngine:
    def __init__(
        self,
        config: TrackerConfig,
        roster_provider: RosterProvider,
        portrait_provider: PortraitProvider,
        frame_source: FrameSource,
        detector: ChampionDetector,
        timeline_sink: TimelineSink,
        clock: Clock,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.roster_provider = roster_provider
        self.portrait_provider = portrait_provider
        self.frame_source = frame_source
        self.detector = detector
        self.timeline_sink = timeline_sink
        self.clock = clock
        self.logger = logger
        self._lock = RLock()
        self._stop = Event()
        self._paused = False
        self._timeline_enabled = False
        self._roster_status = RosterStatus.UNAVAILABLE
        self._identities: tuple[EnemyIdentity, ...] = ()
        self._roster_signature: tuple[tuple[str, str], ...] = ()
        self._portraits: Mapping[str, Image] = {}
        self._positions: dict[str, tuple[int, int]] = {}
        self._last_seen: dict[str, float] = {}
        self._pending: dict[str, _PendingObservation] = {}
        self._camera_center: tuple[int, int] | None = None
        self._missing_polls = 0
        self._last_timeline_record = 0.0
        self._message = "Waiting for a game"
        self._frame_sequence = 0
        self._last_frame_completed: float | None = None
        self._last_api_success: float | None = None
        self._api_failures = 0
        self._api_error: str | None = None
        self._frames_per_second = 0.0
        self._processing_ms = 0.0
        self._consecutive_failures = 0
        self._capture_failures = 0
        self._last_error: str | None = None
        self._ambiguous_rejections = 0
        self._duplicate_rejections = 0
        self._motion_deferrals = 0
        self._next_capture_recovery = 0.0

    @staticmethod
    def _signature(members: tuple[RosterMember, ...]) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted((member.champion_name.casefold(), member.role.value) for member in members)
        )

    def _clear_match(self) -> None:
        self._identities = ()
        self._roster_signature = ()
        self._portraits = {}
        self._positions.clear()
        self._last_seen.clear()
        self._pending.clear()
        self._camera_center = None
        self._message = "Waiting for a game"
        self._frame_sequence = 0
        self._last_frame_completed = None
        self._frames_per_second = 0.0
        self._processing_ms = 0.0
        self._consecutive_failures = 0
        self._capture_failures = 0
        self._last_error = None
        self._ambiguous_rejections = 0
        self._duplicate_rejections = 0
        self._motion_deferrals = 0

    def poll_roster(self) -> None:
        polled_at = self.clock.monotonic()
        result = self.roster_provider.poll()
        if result.status is RosterStatus.ACTIVE:
            signature = self._signature(result.members)
            if signature != self._roster_signature:
                identities = assign_identities(result.members)
                portraits = self.portrait_provider.get_portraits(result.members)
                with self._lock:
                    self._clear_match()
                    self._roster_signature = signature
                    self._identities = identities
                    self._portraits = portraits
                    self._message = "Tracking " + ", ".join(
                        identity.champion_name for identity in identities
                    )
                self.logger.info(self._message)
            else:
                with self._lock:
                    missing = tuple(
                        member
                        for member in result.members
                        if member.champion_name not in self._portraits
                    )
                if missing:
                    recovered = self.portrait_provider.get_portraits(missing)
                    with self._lock:
                        self._portraits = {**self._portraits, **recovered}
            with self._lock:
                self._roster_status = RosterStatus.ACTIVE
                self._missing_polls = 0
                self._last_api_success = polled_at
                self._api_failures = 0
                self._api_error = None
            return

        with self._lock:
            self._roster_status = result.status
            self._api_failures += 1
            self._api_error = result.error or f"Live Client {result.status.value}"
            if not self._identities:
                self._message = "Waiting for a game"
                return
            self._missing_polls += 1
            if self._missing_polls >= self.config.roster_missing_grace_polls:
                self.logger.info(
                    "Live Client unavailable for %s polls; clearing match", self._missing_polls
                )
                self._clear_match()
            else:
                self._message = (
                    f"Live Client unavailable ({self._missing_polls}/"
                    f"{self.config.roster_missing_grace_polls})"
                )

    def _mode(self) -> TrackerMode:
        if self._stop.is_set():
            return TrackerMode.STOPPING
        if self._paused:
            return TrackerMode.PAUSED
        if self._identities:
            return TrackerMode.ACTIVE
        return TrackerMode.WAITING

    def _runtime_health(self, now: float) -> RuntimeHealth:
        frame_age = (
            max(0.0, now - self._last_frame_completed)
            if self._last_frame_completed is not None
            else None
        )
        api_age = (
            max(0.0, now - self._last_api_success) if self._last_api_success is not None else None
        )
        if not self._identities or self._last_frame_completed is None:
            status = AnalysisStatus.WARMING_UP
            message = "Waiting for live frames"
        elif (
            self._consecutive_failures >= self.config.capture_recovery_failure_count
            or frame_age is not None
            and frame_age >= self.config.health_stale_after_seconds * 3
        ):
            status = AnalysisStatus.STALLED
            message = self._last_error or "Live analysis stalled"
        elif (
            self._consecutive_failures > 0
            or self._roster_status is not RosterStatus.ACTIVE
            or frame_age is not None
            and frame_age >= self.config.health_stale_after_seconds
            or self._processing_ms > self.config.update_interval_ms * 1.5
        ):
            status = AnalysisStatus.DEGRADED
            message = self._last_error or self._api_error or "Live analysis delayed"
        else:
            status = AnalysisStatus.HEALTHY
            message = f"{self._frames_per_second:.1f} FPS, {self._processing_ms:.1f} ms"
        return RuntimeHealth(
            status=status,
            frames_per_second=self._frames_per_second,
            processing_ms=self._processing_ms,
            frame_age_seconds=frame_age,
            api_age_seconds=api_age,
            api_failures=self._api_failures,
            consecutive_failures=self._consecutive_failures,
            ambiguous_rejections=self._ambiguous_rejections,
            duplicate_rejections=self._duplicate_rejections,
            motion_deferrals=self._motion_deferrals,
            pending_confirmations=len(self._pending),
            last_error=self._last_error,
            message=message,
        )

    def get_snapshot(self) -> TrackerSnapshot:
        now = self.clock.monotonic()
        with self._lock:
            champions: list[ChampionView] = []
            for identity in self._identities:
                position = self._positions.get(identity.champion_name)
                seen_at = self._last_seen.get(identity.champion_name)
                age = max(0.0, now - seen_at) if seen_at is not None else None
                champions.append(
                    ChampionView(
                        identity=identity,
                        position=position,
                        is_current=(
                            age is not None and age < self.config.detection_timeout_seconds
                        ),
                        seconds_since_seen=age,
                    )
                )
            return TrackerSnapshot(
                mode=self._mode(),
                roster_status=self._roster_status,
                champions=tuple(champions),
                camera_center=self._camera_center,
                timeline_logging=self._timeline_enabled,
                message=self._message,
                health=self._runtime_health(now),
            )

    def _required_confirmation_frames(
        self, champion_name: str, position: tuple[int, int], now: float
    ) -> int:
        previous = self._positions.get(champion_name)
        seen_at = self._last_seen.get(champion_name)
        if previous is None or seen_at is None:
            return self.config.confirmation_frames
        elapsed = max(0.0, now - seen_at)
        allowed_distance = (
            self.config.max_position_jump_pixels
            + self.config.max_position_speed_pixels_per_second * elapsed
        )
        distance = hypot(position[0] - previous[0], position[1] - previous[1])
        if distance > allowed_distance:
            return max(self.config.confirmation_frames, self.config.jump_confirmation_frames)
        return self.config.confirmation_frames

    def _apply_detection(self, detection: DetectionFrame, now: float) -> None:
        with self._lock:
            self._frame_sequence += 1
            frame_number = self._frame_sequence
            self._camera_center = detection.camera_center
            known = {identity.champion_name for identity in self._identities}
            observations: dict[str, ChampionObservation] = {}
            for observation in detection.observations:
                if observation.champion_name not in known:
                    continue
                current = observations.get(observation.champion_name)
                if current is None or observation.score > current.score:
                    observations[observation.champion_name] = observation

            absent = set(self._pending) - set(observations)
            for champion_name in absent:
                del self._pending[champion_name]

            for champion_name, observation in observations.items():
                position = (observation.x, observation.y)
                required = self._required_confirmation_frames(champion_name, position, now)
                pending = self._pending.get(champion_name)
                consistent = (
                    pending is not None
                    and pending.last_frame == frame_number - 1
                    and pending.required_frames == required
                    and hypot(
                        position[0] - pending.position[0],
                        position[1] - pending.position[1],
                    )
                    <= self.config.confirmation_position_tolerance_pixels
                )
                count = pending.count + 1 if consistent and pending is not None else 1
                if required > self.config.confirmation_frames and count < required:
                    self._motion_deferrals += 1
                if count >= required:
                    self._positions[champion_name] = position
                    self._last_seen[champion_name] = now
                    self._pending.pop(champion_name, None)
                else:
                    self._pending[champion_name] = _PendingObservation(
                        position=position,
                        count=count,
                        last_frame=frame_number,
                        required_frames=required,
                    )

    def _record_frame_success(
        self, started: float, completed: float, detection: DetectionFrame
    ) -> None:
        with self._lock:
            if self._last_frame_completed is not None:
                interval = completed - self._last_frame_completed
                if interval > 0:
                    instant_fps = 1.0 / interval
                    self._frames_per_second = (
                        instant_fps
                        if self._frames_per_second == 0.0
                        else self._frames_per_second * 0.8 + instant_fps * 0.2
                    )
            self._last_frame_completed = completed
            self._processing_ms = max(0.0, (completed - started) * 1000.0)
            self._consecutive_failures = 0
            self._capture_failures = 0
            self._last_error = None
            self._ambiguous_rejections += detection.diagnostics.ambiguous
            self._duplicate_rejections += detection.diagnostics.duplicate

    def _record_failure(self, phase: str, error: Exception) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if phase == "capture":
                self._capture_failures += 1
            self._last_error = f"{phase.capitalize()} failure: {error}"

    def _record_api_failure(self, error: Exception) -> None:
        with self._lock:
            self._api_failures += 1
            self._api_error = f"Live Client failure: {error}"

    def process_frame(self) -> None:
        started = self.clock.monotonic()
        try:
            frame = self.frame_source.capture()
        except Exception as exc:
            self._record_failure("capture", exc)
            raise FrameProcessingError("capture", exc) from exc
        with self._lock:
            portraits = dict(self._portraits)
        try:
            detection = self.detector.process(frame, portraits)
        except Exception as exc:
            self._record_failure("detection", exc)
            raise FrameProcessingError("detection", exc) from exc

        completed = self.clock.monotonic()
        self._apply_detection(detection, completed)
        self._record_frame_success(started, completed, detection)

        if self._timeline_enabled and completed - self._last_timeline_record >= 1.0:
            self.timeline_sink.record(self.clock.timestamp(), self.get_snapshot().champions)
            self._last_timeline_record = completed

    def toggle_pause(self) -> bool:
        with self._lock:
            self._paused = not self._paused
            return self._paused

    def toggle_timeline_logging(self) -> bool:
        with self._lock:
            self._timeline_enabled = not self._timeline_enabled
            return self._timeline_enabled

    def flush_timeline(self) -> None:
        self.timeline_sink.flush()

    def stop(self) -> None:
        self._stop.set()

    def _recover_capture_if_needed(self, now: float) -> None:
        with self._lock:
            should_recover = (
                self._capture_failures >= self.config.capture_recovery_failure_count
                and now >= self._next_capture_recovery
            )
            if should_recover:
                self._next_capture_recovery = now + self.config.capture_recovery_backoff_seconds
        if not should_recover:
            return
        self.logger.warning("Restarting screen capture after repeated failures")
        try:
            self.frame_source.close()
            self.frame_source.start()
        except Exception as exc:
            self._record_failure("capture recovery", exc)
            self.logger.error("Screen capture recovery failed: %s", exc)
            return
        with self._lock:
            self._capture_failures = 0
        self.logger.info("Screen capture restarted")

    def _start_frame_source(self) -> bool:
        while not self._stop.is_set():
            try:
                self.frame_source.start()
                return True
            except Exception as exc:
                self._record_failure("capture", exc)
                self.logger.error("Screen capture startup failed: %s", exc)
                self._stop.wait(self.config.capture_recovery_backoff_seconds)
        return False

    def run(self) -> None:
        next_roster_poll = 0.0
        if not self._start_frame_source():
            return
        try:
            while not self._stop.is_set():
                started = self.clock.monotonic()
                if started >= next_roster_poll:
                    try:
                        self.poll_roster()
                    except Exception as exc:
                        self.logger.exception("Roster polling failed")
                        self._record_api_failure(exc)
                    next_roster_poll = started + self.config.roster_refresh_interval_seconds

                with self._lock:
                    can_process = bool(self._identities) and not self._paused
                if can_process:
                    try:
                        self.process_frame()
                    except FrameProcessingError as exc:
                        self.logger.warning("%s", exc)
                        if exc.phase == "capture":
                            self._recover_capture_if_needed(self.clock.monotonic())
                    except Exception as exc:
                        self._record_failure("processing", exc)
                        self.logger.exception("Minimap processing failed")

                elapsed = self.clock.monotonic() - started
                wait_seconds = max(0.0, self.config.update_interval_ms / 1000.0 - elapsed)
                if not can_process:
                    wait_seconds = min(0.5, self.config.roster_refresh_interval_seconds)
                self._stop.wait(wait_seconds)
        finally:
            try:
                self.frame_source.close()
            except Exception:
                self.logger.exception("Could not close screen capture")
