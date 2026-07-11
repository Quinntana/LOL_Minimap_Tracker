"""Thread-safe CSV timeline persistence."""

from __future__ import annotations

import csv
from pathlib import Path
from threading import Lock

from ..domain.models import ChampionView


class CsvTimelineSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows: list[dict[str, object]] = []
        self._lock = Lock()

    def record(self, timestamp: str, champions: tuple[ChampionView, ...]) -> None:
        with self._lock:
            for champion in champions:
                if not champion.is_current or champion.position is None:
                    continue
                self._rows.append(
                    {
                        "timestamp": timestamp,
                        "champion": champion.identity.champion_name,
                        "role": champion.identity.role.value,
                        "X": champion.position[0],
                        "Y": champion.position[1],
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
                writer = csv.DictWriter(
                    file,
                    fieldnames=["timestamp", "champion", "role", "X", "Y"],
                )
                if not has_content:
                    writer.writeheader()
                writer.writerows(rows)
        except OSError:
            with self._lock:
                self._rows[0:0] = rows
            raise
