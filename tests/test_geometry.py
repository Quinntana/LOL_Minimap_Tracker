from lol_minimap_tracker.domain.models import (
    ChampionView,
    EnemyIdentity,
    Role,
)
from lol_minimap_tracker.ui.geometry import (
    marker_layouts,
    rectangles_intersect,
    segment_intersects_rect,
    status_origin,
)


def view(name: str, position: tuple[int, int], age: float) -> ChampionView:
    return ChampionView(
        EnemyIdentity(name, Role.TOP, "#E69F00", "position-top.svg"),
        position,
        False,
        age,
    )


def test_status_panel_avoids_minimap_on_negative_virtual_desktop() -> None:
    screen = (-1920, 0, 3840, 1080)
    minimap = (1655, 813, 252, 252)
    x, y = status_origin(screen, minimap, 5)
    assert not rectangles_intersect((x, y, 260, 110), minimap)


def test_overlapping_markers_are_concentric_oldest_first() -> None:
    layouts = marker_layouts((view("Old", (50, 50), 12), view("New", (54, 53), 5)))
    assert layouts["Old"].radius == 10
    assert layouts["New"].radius == 13
    assert layouts["Old"].icon_opacity > layouts["New"].icon_opacity


def test_segment_intersection_is_conservative() -> None:
    assert segment_intersects_rect((0, 0), (100, 100), (40, 40, 20, 20))
    assert not segment_intersects_rect((0, 0), (10, 10), (40, 40, 20, 20))


def test_non_overlapping_markers_keep_base_radius() -> None:
    layouts = marker_layouts((view("Top", (10, 10), 8), view("Bot", (100, 100), 2)))
    assert layouts["Top"].radius == 10
    assert layouts["Bot"].radius == 10


def test_status_origin_returns_least_overlap_when_map_fills_screen() -> None:
    screen = (0, 0, 300, 200)
    minimap = (0, 0, 300, 200)
    x, y = status_origin(screen, minimap, 5, panel_width=100, row_height=20)
    assert 0 <= x <= 200
    assert 0 <= y <= 100
