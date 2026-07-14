from __future__ import annotations

import logging

import pytest

from lol_minimap_tracker.app import _flush_research_outputs


def test_research_flush_attempts_both_outputs() -> None:
    calls: list[str] = []

    _flush_research_outputs(
        lambda: calls.append("timeline"),
        lambda: calls.append("cooldowns"),
        logging.getLogger("test"),
    )

    assert calls == ["timeline", "cooldowns"]


def test_research_flush_failure_does_not_suppress_the_other_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []

    def fail_timeline() -> None:
        calls.append("timeline")
        raise OSError("disk unavailable")

    with caplog.at_level(logging.ERROR, logger="test.research_flush"):
        _flush_research_outputs(
            fail_timeline,
            lambda: calls.append("cooldowns"),
            logging.getLogger("test.research_flush"),
        )

    assert calls == ["timeline", "cooldowns"]
    assert "Could not flush timeline research events" in caplog.text
