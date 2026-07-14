"""Thread-safe CSV persistence for manual cooldown research events."""

from __future__ import annotations

import csv
from pathlib import Path
from threading import Lock

from ..domain.cooldowns import CooldownEvent


class CsvCooldownEventSink:
    """Buffer cooldown transitions and append them to a stable CSV schema."""

    FIELDNAMES = (
        "timestamp",
        "session",
        "participant",
        "champion",
        "slot",
        "identifier",
        "action",
        "duration",
        "level",
        "remaining",
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows: list[dict[str, object]] = []
        self._lock = Lock()

    def record(self, event: CooldownEvent) -> None:
        with self._lock:
            self._rows.append(
                {
                    "timestamp": event.timestamp,
                    "session": event.session_id,
                    "participant": event.key.participant_id,
                    "champion": event.champion_name,
                    "slot": event.key.slot.value,
                    "identifier": event.identifier,
                    "action": event.action.value,
                    "duration": event.duration,
                    "level": event.level,
                    "remaining": event.remaining,
                }
            )

    def flush(self) -> None:
        with self._lock:
            if not self._rows:
                return
            rows = list(self._rows)
            self._rows.clear()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            has_content = self.path.exists() and self.path.stat().st_size > 0
            with self.path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
                if not has_content:
                    writer.writeheader()
                writer.writerows(rows)
        except OSError:
            with self._lock:
                self._rows[0:0] = rows
            raise
