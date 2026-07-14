"""Pure overlay placement helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import permutations

from ..domain.models import ChampionView

RectTuple = tuple[int, int, int, int]

_DOT_CLUSTER_OFFSETS = {
    1: ((0, 0),),
    2: ((-3, 0), (3, 0)),
    3: ((0, -3), (-3, 3), (3, 3)),
    4: ((-3, -3), (3, -3), (-3, 3), (3, 3)),
    5: ((0, 0), (0, -5), (5, 0), (0, 5), (-5, 0)),
}

_ICON_CLUSTER_OFFSETS = {
    1: ((0, 0),),
    2: ((-11, 0), (11, 0)),
    3: ((0, -13), (-12, 8), (12, 8)),
    4: ((-11, -11), (11, -11), (-11, 11), (11, 11)),
    5: ((0, 0), (0, -24), (24, 0), (0, 24), (-24, 0)),
}

ARROW_CLOSE_RATIO = 0.15
ARROW_FAR_RATIO = 0.55
ARROW_CLOSE_COLOR = (239, 68, 68)
ARROW_MEDIUM_COLOR = (245, 158, 11)
ARROW_FAR_COLOR = (34, 197, 94)
ARROW_CLOSE_LENGTH_RATIO = 0.48
ARROW_FAR_LENGTH_RATIO = 0.14


def rectangles_intersect(first: RectTuple, second: RectTuple) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _marker_offsets(
    champions: tuple[ChampionView, ...],
    patterns: Mapping[int, tuple[tuple[int, int], ...]],
    minimum_radius: int,
    radius_factor: int,
    collision_distance: float | None = None,
) -> dict[str, tuple[int, int]]:
    candidates = [
        champion
        for champion in champions
        if not champion.is_current and champion.position is not None
    ]
    groups: list[list[ChampionView]] = []
    remaining = set(range(len(candidates)))
    while remaining:
        stack = [min(remaining)]
        remaining.remove(stack[0])
        component: list[ChampionView] = []
        while stack:
            index = stack.pop()
            champion = candidates[index]
            assert champion.position is not None
            component.append(champion)
            connected: set[int] = set()
            for other_index in remaining:
                other_position = candidates[other_index].position
                assert other_position is not None
                collides = (
                    other_position == champion.position
                    if collision_distance is None
                    else math.dist(other_position, champion.position) < collision_distance
                )
                if collides:
                    connected.add(other_index)
            remaining.difference_update(connected)
            stack.extend(connected)
        groups.append(component)

    result: dict[str, tuple[int, int]] = {}
    for group in groups:
        group.sort(key=lambda champion: champion.identity.champion_name.casefold())
        offsets = patterns.get(len(group))
        if offsets is None:
            radius = max(minimum_radius, math.ceil(len(group) * radius_factor / math.pi))
            offsets = tuple(
                (
                    round(math.cos(2 * math.pi * index / len(group)) * radius),
                    round(math.sin(2 * math.pi * index / len(group)) * radius),
                )
                for index in range(len(group))
            )
        positions = [champion.position for champion in group]
        assert all(position is not None for position in positions)
        anchor_x = round(
            sum(position[0] for position in positions if position is not None) / len(group)
        )
        anchor_y = round(
            sum(position[1] for position in positions if position is not None) / len(group)
        )

        best_assignment: tuple[tuple[int, int], ...] | None = None
        best_key: tuple[int, tuple[tuple[int, int], ...]] | None = None
        for assignment in permutations(offsets):
            cost = 0
            for champion, offset in zip(group, assignment, strict=True):
                assert champion.position is not None
                target_x = anchor_x + offset[0]
                target_y = anchor_y + offset[1]
                cost += (target_x - champion.position[0]) ** 2
                cost += (target_y - champion.position[1]) ** 2
            key = cost, assignment
            if best_key is None or key < best_key:
                best_key = key
                best_assignment = assignment
        assert best_assignment is not None
        for champion, offset in zip(group, best_assignment, strict=True):
            assert champion.position is not None
            result[champion.identity.champion_name] = (
                anchor_x + offset[0] - champion.position[0],
                anchor_y + offset[1] - champion.position[1],
            )
    return result


def marker_dot_offsets(
    champions: tuple[ChampionView, ...],
) -> dict[str, tuple[int, int]]:
    """Fan exact-position dot collisions into a tiny deterministic cluster."""
    return _marker_offsets(champions, _DOT_CLUSTER_OFFSETS, 5, 3)


def marker_icon_offsets(
    champions: tuple[ChampionView, ...],
) -> dict[str, tuple[int, int]]:
    """Fan nearby portrait/role markers around a shared centroid."""
    return _marker_offsets(
        champions,
        _ICON_CLUSTER_OFFSETS,
        13,
        7,
        collision_distance=16.0,
    )


def normalized_map_distance(distance: float, map_width: int, map_height: int) -> float:
    return max(0.0, distance) / max(1, min(map_width, map_height))


def _interpolate_color(
    first: tuple[int, int, int], second: tuple[int, int, int], amount: float
) -> tuple[int, int, int]:
    return (
        round(first[0] + (second[0] - first[0]) * amount),
        round(first[1] + (second[1] - first[1]) * amount),
        round(first[2] + (second[2] - first[2]) * amount),
    )


def arrow_range_color(
    distance: float,
    map_width: int,
    map_height: int,
) -> tuple[int, int, int]:
    """Return a near/red to far/green camera-relative range color."""
    ratio = normalized_map_distance(distance, map_width, map_height)
    amount = max(
        0.0,
        min(1.0, (ratio - ARROW_CLOSE_RATIO) / (ARROW_FAR_RATIO - ARROW_CLOSE_RATIO)),
    )
    if amount <= 0.5:
        color = _interpolate_color(ARROW_CLOSE_COLOR, ARROW_MEDIUM_COLOR, amount * 2)
    else:
        color = _interpolate_color(ARROW_MEDIUM_COLOR, ARROW_FAR_COLOR, (amount - 0.5) * 2)
    return color


def arrow_display_length(distance: float, map_width: int, map_height: int) -> float:
    """Return an inverse-distance arrow length so nearby enemies read as more urgent."""

    side = max(1, min(map_width, map_height))
    ratio = normalized_map_distance(distance, map_width, map_height)
    range_amount = max(
        0.0,
        min(1.0, (ratio - ARROW_CLOSE_RATIO) / (ARROW_FAR_RATIO - ARROW_CLOSE_RATIO)),
    )
    length_ratio = (
        ARROW_CLOSE_LENGTH_RATIO
        + (ARROW_FAR_LENGTH_RATIO - ARROW_CLOSE_LENGTH_RATIO) * range_amount
    )
    return side * length_ratio


def segment_intersects_rect(
    start: tuple[int, int], end: tuple[int, int], rectangle: RectTuple
) -> bool:
    x1, y1 = start
    x2, y2 = end
    bounds = (min(x1, x2), min(y1, y2), abs(x2 - x1) + 1, abs(y2 - y1) + 1)
    return rectangles_intersect(bounds, rectangle)
