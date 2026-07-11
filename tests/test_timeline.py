from pathlib import Path
from typing import Any

from lol_minimap_tracker.domain.models import ChampionView, EnemyIdentity, Role
from lol_minimap_tracker.integrations.timeline import CsvTimelineSink


def test_timeline_writes_only_current_positions(tmp_path: Path) -> None:
    identity = EnemyIdentity("Aatrox", Role.TOP, "#E69F00", "position-top.svg")
    sink = CsvTimelineSink(tmp_path / "timeline.csv")
    sink.record(
        "2026-07-11 12:00:00",
        (
            ChampionView(identity, (10, 20), True, 0.0),
            ChampionView(identity, (30, 40), False, 5.0),
        ),
    )
    sink.flush()
    content = (tmp_path / "timeline.csv").read_text(encoding="utf-8")
    assert "timestamp,champion,role,X,Y" in content
    assert "Aatrox,TOP,10,20" in content
    assert "30,40" not in content


def test_empty_flush_does_not_create_file(tmp_path: Path) -> None:
    path = tmp_path / "timeline.csv"
    CsvTimelineSink(path).flush()
    assert not path.exists()


def test_failed_flush_restores_buffer_for_retry(tmp_path: Path, monkeypatch: Any) -> None:
    identity = EnemyIdentity("Aatrox", Role.TOP, "#E69F00", "position-top.svg")
    path = tmp_path / "timeline.csv"
    sink = CsvTimelineSink(path)
    sink.record(
        "2026-07-11 12:00:00",
        (ChampionView(identity, (10, 20), True, 0.0),),
    )
    original_open = Path.open

    def fail_open(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "open", fail_open)
    try:
        sink.flush()
    except OSError as exc:
        assert "disk unavailable" in str(exc)
    else:
        raise AssertionError("flush should expose persistence failure")
    monkeypatch.setattr(Path, "open", original_open)
    sink.flush()
    assert "Aatrox,TOP,10,20" in path.read_text(encoding="utf-8")
