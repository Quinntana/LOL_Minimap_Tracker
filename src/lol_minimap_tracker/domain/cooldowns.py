"""Pure models and monotonic timers for manually tracked enemy cooldowns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Protocol

from .interfaces import Clock


class CooldownSlot(StrEnum):
    """A manually clickable slot in an enemy cooldown row."""

    ULTIMATE = "ultimate"
    SPELL_ONE = "spell_one"
    SPELL_TWO = "spell_two"


class CooldownAction(StrEnum):
    """Auditable state transitions emitted by :class:`CooldownTimerStore`."""

    START = "start"
    RESTART = "restart"
    CANCEL = "cancel"
    READY = "ready"
    RESET = "reset"


@dataclass(frozen=True)
class CooldownKey:
    """Stable identity for one participant's clickable cooldown slot."""

    participant_id: str
    slot: CooldownSlot


@dataclass(frozen=True)
class CooldownDefinition:
    """Patch-scoped static cooldown data for an ability or summoner spell."""

    identifier: str
    display_name: str
    icon_path: Path | None
    cooldowns: tuple[float, ...]
    max_rank: int
    unsupported_reason: str | None

    @classmethod
    def placeholder(
        cls,
        identifier: str,
        display_name: str,
        reason: str = "Cooldown data unavailable",
        icon_path: Path | None = None,
    ) -> CooldownDefinition:
        """Create a visibly unsupported definition for incomplete static data."""

        return cls(identifier, display_name, icon_path, (), 0, reason)

    def duration_for_level(self, level: int) -> float | None:
        """Return a conservative base cooldown inferred only from enemy level.

        Three-rank ultimates unlock at levels 6/11/16. Four-rank definitions
        use levels 1/6/11/16 (transform ultimates and similar exceptions), and
        one-rank definitions unlock at level 1. Any other rank layout is not
        inferable from level alone and is deliberately unsupported.
        """

        if self.unsupported_reason is not None:
            return None
        if isinstance(level, bool) or not 1 <= level <= 18:
            return None
        if isinstance(self.max_rank, bool) or self.max_rank not in {1, 3, 4}:
            return None
        if len(self.cooldowns) != self.max_rank:
            return None
        if any(
            isinstance(cooldown, bool) or not math.isfinite(cooldown) or cooldown <= 0.0
            for cooldown in self.cooldowns
        ):
            return None

        rank_index: int
        if self.max_rank == 1:
            rank_index = 0
        elif self.max_rank == 3:
            if level < 6:
                return None
            rank_index = 0 if level < 11 else 1 if level < 16 else 2
        else:
            rank_index = 0 if level < 6 else 1 if level < 11 else 2 if level < 16 else 3
        return float(self.cooldowns[rank_index])


@dataclass(frozen=True)
class EnemyCooldownLoadout:
    """Static cooldown definitions displayed for one enemy participant."""

    participant_id: str
    champion_name: str
    champion_icon_path: Path | None
    ultimate: CooldownDefinition
    summoner_spells: tuple[CooldownDefinition, CooldownDefinition]

    def __post_init__(self) -> None:
        if len(self.summoner_spells) != 2:
            raise ValueError("summoner_spells must contain exactly two definitions")

    def definition_for(self, slot: CooldownSlot) -> CooldownDefinition:
        """Return the definition belonging to a clickable slot."""

        if slot is CooldownSlot.ULTIMATE:
            return self.ultimate
        if slot is CooldownSlot.SPELL_ONE:
            return self.summoner_spells[0]
        return self.summoner_spells[1]


@dataclass(frozen=True)
class CooldownEvent:
    """One immutable transition suitable for research event logging."""

    timestamp: str
    session_id: str
    key: CooldownKey
    champion_name: str
    identifier: str
    action: CooldownAction
    duration: float
    level: int
    remaining: float


@dataclass(frozen=True)
class CooldownSnapshot:
    """Current immutable presentation state for one timer."""

    key: CooldownKey
    champion_name: str
    identifier: str
    display_name: str
    icon_path: Path | None
    duration: float
    level: int
    started_at: float
    ready_at: float
    remaining: float
    is_ready: bool


class CooldownEventRecorder(Protocol):
    """Minimal event persistence boundary used by the pure timer store."""

    def record(self, event: CooldownEvent) -> None: ...


@dataclass
class _TimerState:
    key: CooldownKey
    champion_name: str
    definition: CooldownDefinition
    duration: float
    level: int
    started_at: float
    ready_at: float
    ready_emitted: bool = False


class CooldownTimerStore:
    """Own monotonic manual timers and emit every user-visible transition."""

    def __init__(
        self,
        clock: Clock,
        event_recorder: CooldownEventRecorder | None = None,
        *,
        session_id: str = "",
    ) -> None:
        self._clock = clock
        self._event_recorder = event_recorder
        self._session_id = session_id
        self._timers: dict[CooldownKey, _TimerState] = {}
        self._lock = Lock()

    @property
    def session_id(self) -> str:
        with self._lock:
            return self._session_id

    def __len__(self) -> int:
        with self._lock:
            return len(self._timers)

    def start(
        self,
        key: CooldownKey,
        champion_name: str,
        definition: CooldownDefinition,
        level: int,
    ) -> CooldownSnapshot | None:
        """Start or restart a timer, snapshotting level and base duration."""

        duration = definition.duration_for_level(level)
        if duration is None:
            return None

        events: list[CooldownEvent] = []
        with self._lock:
            now = self._clock.monotonic()
            previous = self._timers.get(key)
            if previous is not None:
                ready_event = self._mark_ready(previous, now)
                if ready_event is not None:
                    events.append(ready_event)
            action = CooldownAction.RESTART if previous is not None else CooldownAction.START
            state = _TimerState(
                key=key,
                champion_name=champion_name,
                definition=definition,
                duration=duration,
                level=level,
                started_at=now,
                ready_at=now + duration,
            )
            self._timers[key] = state
            events.append(self._event(state, action, duration))
            snapshot = self._snapshot(state, now)
        self._record(events)
        return snapshot

    def clear(self, key: CooldownKey) -> bool:
        """Cancel and remove one timer, returning whether it existed."""

        events: list[CooldownEvent] = []
        with self._lock:
            now = self._clock.monotonic()
            state = self._timers.pop(key, None)
            if state is None:
                return False
            ready_event = self._mark_ready(state, now)
            if ready_event is not None:
                events.append(ready_event)
            events.append(self._event(state, CooldownAction.CANCEL, self._remaining(state, now)))
        self._record(events)
        return True

    def clear_all(self) -> int:
        """Cancel and remove every timer, returning the number removed."""

        return self._discard_all(CooldownAction.CANCEL, new_session_id=None)

    def reset_session(self, session_id: str) -> int:
        """Reset all timers and change the stable match/session identifier."""

        return self._discard_all(CooldownAction.RESET, new_session_id=session_id)

    def snapshot(self, key: CooldownKey) -> CooldownSnapshot | None:
        """Return one current timer state and emit its ready transition once."""

        events: list[CooldownEvent] = []
        with self._lock:
            state = self._timers.get(key)
            if state is None:
                return None
            now = self._clock.monotonic()
            ready_event = self._mark_ready(state, now)
            if ready_event is not None:
                events.append(ready_event)
            snapshot = self._snapshot(state, now)
        self._record(events)
        return snapshot

    def snapshots(self) -> tuple[CooldownSnapshot, ...]:
        """Return all current states and emit newly reached ready transitions."""

        events: list[CooldownEvent] = []
        with self._lock:
            now = self._clock.monotonic()
            states = sorted(
                self._timers.values(),
                key=lambda state: (state.key.participant_id, state.key.slot.value),
            )
            snapshots: list[CooldownSnapshot] = []
            for state in states:
                ready_event = self._mark_ready(state, now)
                if ready_event is not None:
                    events.append(ready_event)
                snapshots.append(self._snapshot(state, now))
        self._record(events)
        return tuple(snapshots)

    def remaining(self, key: CooldownKey) -> float | None:
        """Return the non-negative remaining duration for one timer."""

        snapshot = self.snapshot(key)
        return None if snapshot is None else snapshot.remaining

    def _discard_all(
        self,
        action: CooldownAction,
        *,
        new_session_id: str | None,
    ) -> int:
        events: list[CooldownEvent] = []
        with self._lock:
            now = self._clock.monotonic()
            states = tuple(self._timers.values())
            self._timers.clear()
            for state in states:
                ready_event = self._mark_ready(state, now)
                if ready_event is not None:
                    events.append(ready_event)
                events.append(self._event(state, action, self._remaining(state, now)))
            if new_session_id is not None:
                self._session_id = new_session_id
        self._record(events)
        return len(states)

    def _mark_ready(self, state: _TimerState, now: float) -> CooldownEvent | None:
        if state.ready_emitted or now < state.ready_at:
            return None
        state.ready_emitted = True
        return self._event(state, CooldownAction.READY, 0.0)

    def _event(
        self,
        state: _TimerState,
        action: CooldownAction,
        remaining: float,
    ) -> CooldownEvent:
        return CooldownEvent(
            timestamp=self._clock.timestamp(),
            session_id=self._session_id,
            key=state.key,
            champion_name=state.champion_name,
            identifier=state.definition.identifier,
            action=action,
            duration=state.duration,
            level=state.level,
            remaining=remaining,
        )

    @staticmethod
    def _remaining(state: _TimerState, now: float) -> float:
        return max(0.0, state.ready_at - now)

    @classmethod
    def _snapshot(cls, state: _TimerState, now: float) -> CooldownSnapshot:
        remaining = cls._remaining(state, now)
        return CooldownSnapshot(
            key=state.key,
            champion_name=state.champion_name,
            identifier=state.definition.identifier,
            display_name=state.definition.display_name,
            icon_path=state.definition.icon_path,
            duration=state.duration,
            level=state.level,
            started_at=state.started_at,
            ready_at=state.ready_at,
            remaining=remaining,
            is_ready=remaining == 0.0,
        )

    def _record(self, events: list[CooldownEvent]) -> None:
        if self._event_recorder is None:
            return
        for event in events:
            self._event_recorder.record(event)
