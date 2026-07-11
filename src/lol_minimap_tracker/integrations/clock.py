"""Production clock adapter."""

from __future__ import annotations

import time
from datetime import datetime


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
