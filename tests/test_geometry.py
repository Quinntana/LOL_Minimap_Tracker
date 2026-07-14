from itertools import permutations

from lol_minimap_tracker.domain.models import (
    ChampionView,
    EnemyIdentity,
    Role,
)
from lol_minimap_tracker.ui.geometry import (
    arrow_display_length,
    arrow_range_color,
    marker_dot_offsets,
    marker_icon_offsets,
    normalized_map_distance,
    segment_intersects_rect,
)


def view(name: str, position: tuple[int, int], age: float) -> ChampionView:
    return ChampionView(
        EnemyIdentity(name, Role.TOP, "#E69F00", "position-top.svg"),
        position,
        False,
        age,
    )


def test_coincident_dots_fan_out_deterministically() -> None:
    offsets = marker_dot_offsets(
        (
            view("Zed", (50, 50), 12),
            view("Alpha", (50, 50), 5),
            view("Solo", (80, 80), 2),
        )
    )
    assert offsets == {"Alpha": (-3, 0), "Zed": (3, 0), "Solo": (0, 0)}


def test_coincident_rich_markers_use_wider_offsets() -> None:
    offsets = marker_icon_offsets(
        (
            view("Zed", (50, 50), 12),
            view("Alpha", (50, 50), 5),
            view("Solo", (80, 80), 2),
        )
    )
    assert offsets == {"Alpha": (-11, 0), "Zed": (11, 0), "Solo": (0, 0)}


def test_nearby_rich_markers_cluster_but_nearby_dots_do_not() -> None:
    champions = (
        view("Zed", (50, 50), 12),
        view("Alpha", (54, 53), 5),
    )
    icon_offsets = marker_icon_offsets(champions)
    icon_centers = {
        champion.identity.champion_name: (
            champion.position[0] + icon_offsets[champion.identity.champion_name][0],
            champion.position[1] + icon_offsets[champion.identity.champion_name][1],
        )
        for champion in champions
        if champion.position is not None
    }
    assert icon_centers == {"Alpha": (63, 52), "Zed": (41, 52)}
    assert marker_dot_offsets(champions) == {"Zed": (0, 0), "Alpha": (0, 0)}


def test_bridge_connected_rich_cluster_is_order_independent() -> None:
    champions = (
        view("A", (0, 50), 12),
        view("B", (30, 50), 8),
        view("C", (15, 50), 5),
    )
    expected_centers = {"A": (3, 58), "B": (27, 58), "C": (15, 37)}
    for ordering in permutations(champions):
        offsets = marker_icon_offsets(ordering)
        centers = {
            champion.identity.champion_name: (
                champion.position[0] + offsets[champion.identity.champion_name][0],
                champion.position[1] + offsets[champion.identity.champion_name][1],
            )
            for champion in champions
            if champion.position is not None
        }
        assert centers == expected_centers


def test_segment_intersection_is_conservative() -> None:
    assert segment_intersects_rect((0, 0), (100, 100), (40, 40, 20, 20))
    assert not segment_intersects_rect((0, 0), (10, 10), (40, 40, 20, 20))


def test_distance_arrow_colors_are_resolution_independent() -> None:
    close = arrow_range_color(15, 100, 100)
    medium = arrow_range_color(35, 100, 100)
    far = arrow_range_color(55, 100, 100)
    assert close == (239, 68, 68)
    assert medium == (245, 158, 11)
    assert far == (34, 197, 94)
    assert arrow_range_color(70, 200, 200) == medium
    assert arrow_range_color(-1, 0, 0) == close
    assert normalized_map_distance(50, 200, 100) == 0.5


def test_arrow_length_is_longer_for_nearby_threats_and_scales_with_map() -> None:
    close = arrow_display_length(15, 100, 100)
    medium = arrow_display_length(35, 100, 100)
    far = arrow_display_length(55, 100, 100)
    assert close == 48.0
    assert close > medium > far
    assert round(far, 6) == 14.0
    assert arrow_display_length(70, 200, 200) == medium * 2
    assert arrow_display_length(-1, 100, 100) == close
