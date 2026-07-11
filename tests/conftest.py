from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp() -> Iterator[object]:
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
