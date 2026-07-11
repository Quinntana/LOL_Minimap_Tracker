from __future__ import annotations

from typing import Any

from PyQt5.QtGui import QImage

from lol_minimap_tracker.config import CaptureRegion, TrackerConfig
from lol_minimap_tracker.domain.models import (
    AffinityResult,
    AffinityStatus,
    ChampionView,
    EnemyIdentity,
    Role,
    RosterStatus,
    TrackerMode,
    TrackerSnapshot,
)
from lol_minimap_tracker.paths import AppPaths
from lol_minimap_tracker.ui.overlay import TransparentOverlay
from lol_minimap_tracker.ui.role_icons import RoleIconRenderer
from lol_minimap_tracker.ui.tray import TrayController


class Affinity:
    def apply(self, _handle: int, _enabled: bool) -> AffinityResult:
        return AffinityResult(AffinityStatus.ACTIVE)


def snapshot() -> TrackerSnapshot:
    top = EnemyIdentity("Aatrox", Role.TOP, "#E69F00", "position-top.svg")
    jungle = EnemyIdentity("Nidalee", Role.JUNGLE, "#009E73", "position-jungle.svg")
    return TrackerSnapshot(
        mode=TrackerMode.ACTIVE,
        roster_status=RosterStatus.ACTIVE,
        champions=(
            ChampionView(top, (30, 40), True, 0.1),
            ChampionView(jungle, (70, 80), False, 6.0),
        ),
        camera_center=(50, 50),
        timeline_logging=True,
        message="Tracking Aatrox, Nidalee",
    )


def test_role_icons_render_and_cache(qapp: object) -> None:
    del qapp
    renderer = RoleIconRenderer(AppPaths.discover().role_asset_dir)
    first = renderer.render("position-top.svg", "#E69F00", 16, 0.35)
    second = renderer.render("position-top.svg", "#E69F00", 16, 0.35)
    assert not first.isNull()
    assert first.cacheKey() == second.cacheKey()
    assert not renderer.render("position-unknown.svg", "#B8B8B8", 18).isNull()


def test_overlay_renders_safe_and_fallback_modes(qapp: Any) -> None:
    current = snapshot()
    changed: list[AffinityResult] = []
    overlay = TransparentOverlay(
        lambda: current,
        TrackerConfig(capture=CaptureRegion(200, 300, 100, 100)),
        RoleIconRenderer(AppPaths.discover().role_asset_dir),
        Affinity(),
        changed.append,
    )
    overlay.snapshot = current
    overlay.show()
    qapp.processEvents()
    assert changed[-1].status is AffinityStatus.ACTIVE
    overlay.affinity_result = AffinityResult(AffinityStatus.ACTIVE)
    image = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(0)
    overlay.render(image)
    assert not image.isNull()
    assert not overlay.toggle_arrows()
    assert not overlay.toggle_last_seen()
    overlay.update_overlay()

    overlay.affinity_result = AffinityResult(AffinityStatus.FAILED)
    fallback = QImage(overlay.size(), QImage.Format_ARGB32_Premultiplied)
    fallback.fill(0)
    overlay.render(fallback)
    overlay.close()


def test_tray_updates_and_dispatches(qapp: Any) -> None:
    dispatched: list[str] = []
    tray = TrayController(qapp, dispatched.append)
    tray.toggle_actions["toggle_arrows"].trigger()
    assert dispatched == ["toggle_arrows"]
    tray.update_action("toggle_arrows", True)
    assert tray.toggle_actions["toggle_arrows"].isChecked()
    tray.update_snapshot(snapshot())
    assert "Tracking Aatrox" in tray.status_action.text()
    assert "waiting for live frames" in tray.analysis_action.text().lower()
    tray.update_affinity(AffinityResult(AffinityStatus.ACTIVE))
    assert tray.affinity_action.text() == "Capture exclusion: active"
    save_action = next(action for action in tray.menu.actions() if action.text() == "Save timeline")
    save_action.trigger()
    assert dispatched[-1] == "save_timeline"
