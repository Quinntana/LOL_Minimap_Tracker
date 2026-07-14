from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from lol_minimap_tracker.app import _flush_research_outputs, _LegacyCaptureRegionMigrator
from lol_minimap_tracker.config import DEFAULT_CONFIG, CaptureRegion, TrackerConfig


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


def test_legacy_capture_migration_waits_for_confirmation_and_persists_once() -> None:
    screen_region = CaptureRegion(top=900, left=1700, width=260, height=260)
    client_region = CaptureRegion(top=700, left=1500, width=260, height=260)
    current = replace(
        DEFAULT_CONFIG,
        capture=screen_region,
        capture_region_space="screen",
    )
    persisted: list[TrackerConfig] = []
    migrator = _LegacyCaptureRegionMigrator(enabled=True)

    assert (
        migrator.persist_if_ready(current, None, lambda value: persisted.append(value) or True)
        is None
    )
    assert migrator.pending
    assert persisted == []

    assert (
        migrator.persist_if_ready(
            current,
            client_region,
            lambda value: persisted.append(value) or True,
        )
        is True
    )
    assert not migrator.pending
    assert persisted == [replace(current, capture=client_region, capture_region_space="client")]

    assert (
        migrator.persist_if_ready(
            current,
            client_region,
            lambda value: persisted.append(value) or True,
        )
        is None
    )
    assert len(persisted) == 1


def test_failed_legacy_capture_save_is_not_retried_in_a_write_loop() -> None:
    current = replace(DEFAULT_CONFIG, capture_region_space="screen")
    client_region = CaptureRegion(top=700, left=1500, width=260, height=260)
    attempts: list[TrackerConfig] = []
    migrator = _LegacyCaptureRegionMigrator(enabled=True)

    assert (
        migrator.persist_if_ready(
            current,
            client_region,
            lambda value: attempts.append(value) or False,
        )
        is False
    )
    assert not migrator.pending
    assert migrator.persist_if_ready(current, client_region, lambda _value: True) is None
    assert len(attempts) == 1


def test_client_space_configuration_disables_legacy_migration() -> None:
    current = replace(DEFAULT_CONFIG, capture_region_space="client")
    migrator = _LegacyCaptureRegionMigrator(enabled=True)

    assert migrator.persist_if_ready(current, current.capture, lambda _value: True) is None
    assert not migrator.pending
