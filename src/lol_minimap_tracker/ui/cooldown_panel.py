"""Separate interactive panel for private manual enemy cooldown research."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from queue import Empty, Queue, SimpleQueue
from threading import Event, Lock, Thread
from typing import Protocol

from PyQt5.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap, QShowEvent
from PyQt5.QtWidgets import (
    QAbstractButton,
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..domain.cooldowns import (
    CooldownDefinition,
    CooldownKey,
    CooldownSlot,
    CooldownSnapshot,
    CooldownTimerStore,
    EnemyCooldownLoadout,
)
from ..domain.interfaces import DisplayAffinityController
from ..domain.models import AffinityStatus, Role, RosterMember, RosterState


class CooldownCatalog(Protocol):
    def get_loadouts(
        self, members: tuple[RosterMember, ...]
    ) -> tuple[EnemyCooldownLoadout, ...]: ...


_CatalogTask = tuple[int, tuple[RosterMember, ...]]
_CatalogResult = tuple[int, tuple[EnemyCooldownLoadout, ...] | None, Exception | None]


class _CooldownCatalogWorker:
    """Latest-task metadata loader that never delays interpreter shutdown."""

    def __init__(self, catalog: CooldownCatalog) -> None:
        self._catalog = catalog
        self._tasks: Queue[_CatalogTask | None] = Queue()
        self._results: SimpleQueue[_CatalogResult] = SimpleQueue()
        self._stop = Event()
        self._idle = Event()
        self._idle.set()
        self._state_lock = Lock()
        self._started = False
        self._thread = Thread(target=self._run, name="cooldown-data", daemon=True)

    def submit(self, generation: int, members: tuple[RosterMember, ...]) -> None:
        with self._state_lock:
            if self._stop.is_set():
                return
            if not self._started:
                self._thread.start()
                self._started = True
            self._idle.clear()
            self._tasks.put((generation, members))

    def poll(self) -> tuple[_CatalogResult, ...]:
        results: list[_CatalogResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except Empty:
                return tuple(results)

    def wait_until_idle(self, timeout: float) -> bool:
        """Test/diagnostic hook; normal UI refreshes never block on metadata."""

        return self._idle.wait(timeout)

    def shutdown(self) -> None:
        self._stop.set()
        cancel = getattr(self._catalog, "cancel", None)
        if callable(cancel):
            cancel()
        self._tasks.put(None)
        self._idle.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            task = self._tasks.get()
            if task is None:
                return
            while True:
                try:
                    newer = self._tasks.get_nowait()
                except Empty:
                    break
                if newer is None:
                    return
                task = newer
            generation, members = task
            try:
                loadouts = self._catalog.get_loadouts(members)
                error: Exception | None = None
            except Exception as exc:
                loadouts = None
                error = exc
            if not self._stop.is_set():
                self._results.put((generation, loadouts, error))
            with self._state_lock:
                if self._tasks.empty():
                    self._idle.set()


ROLE_ORDER = {
    Role.TOP: 0,
    Role.JUNGLE: 1,
    Role.MIDDLE: 2,
    Role.BOTTOM: 3,
    Role.UTILITY: 4,
    Role.UNKNOWN: 5,
}


def format_remaining(seconds: float) -> str:
    """Round upward so the display never reports ready before the timer is ready."""

    rounded = max(0, math.ceil(seconds))
    if rounded < 60:
        return str(rounded)
    minutes, remainder = divmod(rounded, 60)
    return f"{minutes}:{remainder:02d}"


class CooldownIconButton(QAbstractButton):
    """One icon with explicit left-start and right-clear mouse semantics."""

    start_requested = pyqtSignal(object)
    clear_requested = pyqtSignal(object)

    def __init__(self, slot: CooldownSlot, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slot = slot
        self.key: CooldownKey | None = None
        self.definition: CooldownDefinition | None = None
        self.timer_snapshot: CooldownSnapshot | None = None
        self.level: int | None = None
        self._pixmap = QPixmap()
        self._icon_path: Path | None = None
        self._icon_loaded = False
        self._can_start = False
        self.setFixedSize(34, 34)
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.PointingHandCursor)

    def set_view(
        self,
        key: CooldownKey | None,
        definition: CooldownDefinition | None,
        timer_snapshot: CooldownSnapshot | None,
        level: int | None,
    ) -> None:
        self.key = key
        self.definition = definition
        self.timer_snapshot = timer_snapshot
        self.level = level
        icon_path = definition.icon_path if definition else None
        if not self._icon_loaded or icon_path != self._icon_path:
            self._pixmap = self._load_icon(icon_path)
            self._icon_path = icon_path
            self._icon_loaded = True
        duration = definition.duration_for_level(level) if definition and level else None
        self._can_start = key is not None and duration is not None
        self.setCursor(Qt.PointingHandCursor if self._can_start else Qt.ArrowCursor)
        self.setToolTip(self._tooltip(duration))
        accessible_name = definition.display_name if definition is not None else self.slot.value
        self.setAccessibleName(accessible_name)
        self.setAccessibleDescription(self.toolTip())
        self.update()

    @staticmethod
    def _load_icon(path: Path | None) -> QPixmap:
        if path is None or not path.exists():
            return QPixmap()
        return QPixmap(str(path))

    def _tooltip(self, duration: float | None) -> str:
        definition = self.definition
        if definition is None:
            return "Waiting for patch cooldown data."
        if definition.unsupported_reason:
            return f"{definition.display_name}: {definition.unsupported_reason}"
        if duration is None:
            if self.level is None:
                return f"{definition.display_name}: waiting for enemy level."
            return f"{definition.display_name}: not learned at level {self.level}."
        instruction = "Left-click start/restart; right-click clear."
        limitation = "Base cooldown only; ability haste, runes and items are ignored."
        return f"{definition.display_name}: {duration:g}s. {instruction} {limitation}"

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        key = self.key
        inside = self.rect().contains(event.pos())
        can_start = self._can_start
        button = event.button()
        # Let QAbstractButton clear its pressed state before callbacks refresh
        # this widget.  Skipping the base release handler leaves isDown() stuck.
        super().mouseReleaseEvent(event)
        self.setDown(False)
        if key is None or not inside:
            return
        if button == Qt.RightButton:
            self.clear_requested.emit(key)
            event.accept()
            return
        if button == Qt.LeftButton and can_start:
            self.start_requested.emit(key)
            event.accept()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(93, 105, 128, 210), 1))
        painter.setBrush(QColor(17, 24, 39, 245))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)

        snapshot = self.timer_snapshot
        if not self._pixmap.isNull():
            icon = self._pixmap.scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(
                (self.width() - icon.width()) // 2, (self.height() - icon.height()) // 2, icon
            )
        elif self._can_start or snapshot is not None:
            labels = {
                CooldownSlot.ULTIMATE: "R",
                CooldownSlot.SPELL_ONE: "D",
                CooldownSlot.SPELL_TWO: "F",
            }
            painter.setPen(QColor(190, 199, 215, 230))
            painter.drawText(self.rect(), Qt.AlignCenter, labels[self.slot])
        if snapshot is not None and not snapshot.is_ready:
            painter.setBrush(QColor(3, 7, 18, 175))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 4, 4)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(self.rect(), Qt.AlignCenter, format_remaining(snapshot.remaining))
        elif snapshot is not None and snapshot.is_ready:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(74, 222, 128), 2))
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 5, 5)
        elif not self._can_start:
            painter.setBrush(QColor(3, 7, 18, 150))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 4, 4)
            painter.setPen(QColor(188, 196, 211, 235))
            painter.drawText(self.rect(), Qt.AlignCenter, "—")


class CooldownEnemyRow(QWidget):
    start_requested = pyqtSignal(object)
    clear_requested = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.member: RosterMember | None = None
        self._champion_icon_path: Path | None = None
        self._champion_fallback = ""
        self._champion_icon_loaded = False
        self.setFixedHeight(38)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(4)
        self.champion_icon = QLabel()
        self.champion_icon.setFixedSize(30, 30)
        self.champion_icon.setAlignment(Qt.AlignCenter)
        self.champion_icon.setStyleSheet(
            "background: #111827; border: 1px solid #4b5563; border-radius: 4px; color: #d1d5db;"
        )
        layout.addWidget(self.champion_icon)
        self.buttons = {
            slot: CooldownIconButton(slot, self)
            for slot in (
                CooldownSlot.ULTIMATE,
                CooldownSlot.SPELL_ONE,
                CooldownSlot.SPELL_TWO,
            )
        }
        for button in self.buttons.values():
            button.start_requested.connect(self.start_requested.emit)
            button.clear_requested.connect(self.clear_requested.emit)
            layout.addWidget(button)
        layout.addStretch(1)

    def set_view(
        self,
        member: RosterMember | None,
        loadout: EnemyCooldownLoadout | None,
        timers: Mapping[CooldownKey, CooldownSnapshot],
    ) -> None:
        self.member = member
        if member is None:
            self.champion_icon.clear()
            self.champion_icon.setText("?")
            self.champion_icon.setToolTip("Waiting for enemy roster")
            self.champion_icon.setAccessibleName("Waiting for enemy roster")
            for button in self.buttons.values():
                button.set_view(None, None, None, None)
            return

        level_text = "?" if member.level is None else str(member.level)
        identity_text = f"{member.champion_name} — level {level_text}"
        self.champion_icon.setToolTip(identity_text)
        self.champion_icon.setAccessibleName(identity_text)
        fallback = "".join(character for character in member.champion_name if character.isalnum())[
            :2
        ].upper()
        self._set_champion_icon(loadout.champion_icon_path if loadout else None, fallback or "?")
        participant_id = member.participant_id or member.champion_name.casefold()
        for slot, button in self.buttons.items():
            key = CooldownKey(participant_id, slot)
            definition = loadout.definition_for(slot) if loadout is not None else None
            button.set_view(key, definition, timers.get(key), member.level)

    def _set_champion_icon(self, path: Path | None, fallback: str) -> None:
        if (
            self._champion_icon_loaded
            and path == self._champion_icon_path
            and fallback == self._champion_fallback
        ):
            return
        self._champion_icon_path = path
        self._champion_fallback = fallback
        self._champion_icon_loaded = True
        pixmap = QPixmap(str(path)) if path is not None and path.exists() else QPixmap()
        if pixmap.isNull():
            self.champion_icon.setPixmap(QPixmap())
            self.champion_icon.setText(fallback)
            return
        self.champion_icon.setText("")
        self.champion_icon.setPixmap(
            pixmap.scaled(28, 28, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        )


class CooldownPanel(QWidget):
    """Five-row interactive timer panel; intentionally not click-through."""

    visibility_requested = pyqtSignal(bool)
    lock_requested = pyqtSignal(bool)
    position_changed = pyqtSignal(object)
    safety_blocked = pyqtSignal(str)

    def __init__(
        self,
        roster_provider: Callable[[], RosterState],
        catalog: CooldownCatalog,
        timer_store: CooldownTimerStore,
        affinity_controller: DisplayAffinityController,
        logger: logging.Logger,
        *,
        capture_isolated: Callable[[], bool] | None = None,
        exclude_from_capture: bool = True,
        enabled: bool = False,
        locked: bool = False,
        left: int | None = None,
        top: int | None = None,
        session_prefix: str = "",
    ) -> None:
        super().__init__()
        self.roster_provider = roster_provider
        self.catalog = catalog
        self.timer_store = timer_store
        self.affinity_controller = affinity_controller
        self.logger = logger
        self.capture_isolated = capture_isolated or (lambda: False)
        self.exclude_from_capture = exclude_from_capture
        self._enabled = enabled
        self._locked = locked
        self._session_prefix = session_prefix.strip()
        self._affinity_applied = False
        self._shutting_down = False
        self._drag_offset: QPoint | None = None
        self._roster_generation: int | None = None
        self._confirmed_roster_identity: tuple[tuple[str, str], ...] = ()
        self._last_confirmed_members: tuple[RosterMember, ...] = ()
        self._members: tuple[RosterMember, ...] = ()
        self._loadout_signature: tuple[tuple[str, ...], ...] = ()
        self._catalog_revision = 0
        self._loadouts: dict[str, EnemyCooldownLoadout] = {}
        self._worker = _CooldownCatalogWorker(catalog)
        self._catalog_request_pending = False
        self._catalog_retry_failures = 0
        self._catalog_retry_at = 0.0
        self._init_ui()
        self._place_initial(left, top)
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.refresh)
        if enabled:
            self.timer.start()

    def _init_ui(self) -> None:
        self.setWindowFlags(
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.NoFocus)
        self.setFixedSize(160, 218)
        self.setStyleSheet("background: #0b1220;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QWidget(self)
        header.setFixedHeight(28)
        header.setObjectName("cooldownHeader")
        header.setStyleSheet(
            "#cooldownHeader { background: #172033; border-bottom: 1px solid #334155; }"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(5, 3, 3, 3)
        header_layout.addStretch(1)
        self.lock_button = QPushButton("Lock" if not self._locked else "Unlock")
        self.lock_button.setFixedSize(48, 22)
        self.lock_button.setFocusPolicy(Qt.NoFocus)
        control_style = (
            "QPushButton { color: #e5e7eb; background: #263247; border: 1px solid #475569; "
            "border-radius: 3px; font-size: 9px; } "
            "QPushButton:hover { background: #334155; }"
        )
        self.lock_button.setStyleSheet(control_style)
        self.lock_button.clicked.connect(self._toggle_lock_requested)
        header_layout.addWidget(self.lock_button)
        self.hide_button = QPushButton("Hide")
        self.hide_button.setFixedSize(38, 22)
        self.hide_button.setFocusPolicy(Qt.NoFocus)
        self.hide_button.setStyleSheet(control_style)
        self.hide_button.clicked.connect(lambda: self.visibility_requested.emit(False))
        header_layout.addWidget(self.hide_button)
        outer.addWidget(header)
        self.header = header

        self.rows = tuple(CooldownEnemyRow(self) for _index in range(5))
        for row in self.rows:
            row.start_requested.connect(self._start)
            row.clear_requested.connect(self._clear)
            outer.addWidget(row)

    def _place_initial(self, left: int | None, top: int | None) -> None:
        screen_object = self.screen() or QApplication.primaryScreen()
        if screen_object is None:
            self.move(QPoint(left or 0, top or 0))
            return
        screen = screen_object.availableGeometry()
        target = QPoint(screen.left() + 20, screen.top() + 180)
        if left is not None and top is not None:
            target = QPoint(left, top)
        self.move(self._clamped(target))

    def _clamped(self, target: QPoint) -> QPoint:
        screen_object = self.screen() or QApplication.primaryScreen()
        if screen_object is None:
            return target
        screens = screen_object.virtualSiblings()
        available = next(
            (
                screen.availableGeometry()
                for screen in screens
                if screen.availableGeometry().contains(target)
            ),
            screen_object.availableGeometry(),
        )
        return QPoint(
            min(
                max(target.x(), available.left()),
                max(available.left(), available.right() - self.width() + 1),
            ),
            min(
                max(target.y(), available.top()),
                max(available.top(), available.bottom() - self.height() + 1),
            ),
        )

    def showEvent(self, event: QShowEvent | None) -> None:
        super().showEvent(event)
        if not self._affinity_applied:
            result = self.affinity_controller.apply(int(self.winId()), self.exclude_from_capture)
            if not self.capture_isolated() and result.status is not AffinityStatus.ACTIVE:
                self.logger.error("Cooldown panel capture exclusion unavailable: %s", result.status)
                self._enabled = False
                self.hide()
                self.timer.stop()
                self.safety_blocked.emit(result.status.value)
                return
            self._affinity_applied = True
        self.refresh()

    def set_panel_visible(self, visible: bool) -> bool:
        self._enabled = visible
        if visible:
            self.timer.start()
            self.show()
            self.raise_()
            self.refresh()
        else:
            self.timer.stop()
            self.hide()
        return self._enabled

    def set_locked(self, locked: bool) -> bool:
        self._locked = locked
        self.lock_button.setText("Unlock" if locked else "Lock")
        return self._locked

    def _toggle_lock_requested(self) -> None:
        self.lock_requested.emit(not self._locked)

    def refresh(self) -> None:
        if not self._enabled or self._shutting_down:
            return
        state = self.roster_provider()
        ordered_members = self._ordered(state.members)
        roster_identity = self._roster_identity(ordered_members)
        loadout_signature = self._metadata_signature(ordered_members)
        generation_changed = state.generation != self._roster_generation
        same_confirmed_roster = bool(ordered_members) and (
            roster_identity == self._confirmed_roster_identity
        )
        if generation_changed:
            self._roster_generation = state.generation
            # A temporarily empty roster is how the engine represents a Live
            # Client outage after its grace period. Keep the manual timers so a
            # short local-API interruption cannot erase the user's clicks. An
            # actually different confirmed roster establishes a new session.
            if ordered_members and not same_confirmed_roster:
                session_id = (
                    f"{self._session_prefix}-{state.generation}"
                    if self._session_prefix
                    else str(state.generation)
                )
                self.timer_store.reset_session(session_id)
        if loadout_signature != self._loadout_signature:
            if same_confirmed_roster and generation_changed:
                self._clear_changed_spell_timers(self._last_confirmed_members, ordered_members)
            elif not generation_changed:
                self._clear_changed_spell_timers(self._members, ordered_members)
            self._loadout_signature = loadout_signature
            self._catalog_revision += 1
            self._loadouts.clear()
            self._catalog_request_pending = False
            self._catalog_retry_failures = 0
            self._catalog_retry_at = 0.0
            self._queue_loadouts(self._catalog_revision, ordered_members)
        if ordered_members:
            self._confirmed_roster_identity = roster_identity
            self._last_confirmed_members = ordered_members
        self._members = ordered_members
        self._collect_loadouts()
        if (
            self._members
            and not self._loadouts
            and not self._catalog_request_pending
            and time.monotonic() >= self._catalog_retry_at
        ):
            self._queue_loadouts(self._catalog_revision, self._members)
        timers = {snapshot.key: snapshot for snapshot in self.timer_store.snapshots()}
        for index, row in enumerate(self.rows):
            member = self._members[index] if index < len(self._members) else None
            key = (member.participant_id or member.champion_name.casefold()) if member else ""
            row.set_view(member, self._loadouts.get(key), timers)

    @staticmethod
    def _ordered(members: tuple[RosterMember, ...]) -> tuple[RosterMember, ...]:
        return tuple(
            sorted(
                members,
                key=lambda member: (
                    ROLE_ORDER[member.role],
                    member.champion_name.casefold(),
                    member.participant_id,
                ),
            )[:5]
        )

    @staticmethod
    def _roster_identity(
        members: tuple[RosterMember, ...],
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (
                member.participant_id or member.champion_name.casefold(),
                (member.champion_id or member.champion_name).casefold(),
            )
            for member in members
        )

    @staticmethod
    def _metadata_signature(members: tuple[RosterMember, ...]) -> tuple[tuple[str, ...], ...]:
        result: list[tuple[str, ...]] = []
        for member in members:
            spells = tuple(
                (spell.identifier or spell.display_name).strip().casefold()
                for spell in member.summoner_spells[:2]
            )
            result.append(
                (
                    member.participant_id or member.champion_name.casefold(),
                    (member.champion_id or member.champion_name).casefold(),
                    *spells,
                )
            )
        return tuple(result)

    def _clear_changed_spell_timers(
        self,
        previous: tuple[RosterMember, ...],
        current: tuple[RosterMember, ...],
    ) -> None:
        previous_by_id = {
            member.participant_id or member.champion_name.casefold(): member for member in previous
        }
        slots = (CooldownSlot.SPELL_ONE, CooldownSlot.SPELL_TWO)
        for member in current:
            participant_id = member.participant_id or member.champion_name.casefold()
            old_member = previous_by_id.get(participant_id)
            if old_member is None:
                continue
            for index, slot in enumerate(slots):
                old_reference = (
                    old_member.summoner_spells[index]
                    if index < len(old_member.summoner_spells)
                    else None
                )
                new_reference = (
                    member.summoner_spells[index] if index < len(member.summoner_spells) else None
                )
                old_identity = (
                    (old_reference.identifier or old_reference.display_name).strip().casefold()
                    if old_reference is not None
                    else ""
                )
                new_identity = (
                    (new_reference.identifier or new_reference.display_name).strip().casefold()
                    if new_reference is not None
                    else ""
                )
                if old_identity and new_identity and old_identity != new_identity:
                    self.timer_store.clear(CooldownKey(participant_id, slot))

    def _queue_loadouts(self, revision: int, members: tuple[RosterMember, ...]) -> None:
        if members and not self._catalog_request_pending:
            self._catalog_request_pending = True
            self._worker.submit(revision, members)

    def _collect_loadouts(self) -> None:
        for revision, loadouts, error in self._worker.poll():
            if revision != self._catalog_revision:
                continue
            self._catalog_request_pending = False
            if error is not None:
                self._catalog_retry_failures += 1
                delay = min(30.0, float(2 ** min(self._catalog_retry_failures, 5)))
                self._catalog_retry_at = time.monotonic() + delay
                self.logger.warning(
                    "Cooldown metadata loading failed; retrying in %.0fs: %s", delay, error
                )
                continue
            if loadouts is None:
                continue
            self._loadouts = {loadout.participant_id: loadout for loadout in loadouts}
            self._catalog_retry_failures = 0
            self._catalog_retry_at = 0.0

    def _member_for(self, participant_id: str) -> RosterMember | None:
        return next(
            (
                member
                for member in self._members
                if (member.participant_id or member.champion_name.casefold()) == participant_id
            ),
            None,
        )

    def _start(self, value: object) -> None:
        if not isinstance(value, CooldownKey):
            return
        member = self._member_for(value.participant_id)
        loadout = self._loadouts.get(value.participant_id)
        if member is None or member.level is None or loadout is None:
            return
        self.timer_store.start(
            value,
            member.champion_name,
            loadout.definition_for(value.slot),
            member.level,
        )
        self.refresh()

    def _clear(self, value: object) -> None:
        if isinstance(value, CooldownKey):
            self.timer_store.clear(value)
            self.refresh()

    def clear_timers(self) -> int:
        removed = self.timer_store.clear_all()
        self.refresh()
        return removed

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if (
            event is not None
            and event.button() == Qt.LeftButton
            and not self._locked
            and event.pos().y() < self.header.height()
            and self.childAt(event.pos()) not in {self.lock_button, self.hide_button}
        ):
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(self._clamped(event.globalPos() - self._drag_offset))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.LeftButton and self._drag_offset is not None:
            self._drag_offset = None
            self.move(self._clamped(self.pos()))
            self.position_changed.emit((self.x(), self.y()))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self.timer.stop()
        self.hide()
        self._worker.shutdown()
