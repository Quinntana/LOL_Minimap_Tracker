"""Pure overlay placement helpers."""

from __future__ import annotations

import math

from ..domain.models import ChampionView, MarkerLayout

RectTuple = tuple[int, int, int, int]


def rectangles_intersect(first: RectTuple, second: RectTuple) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def status_origin(
    screen: RectTuple,
    minimap: RectTuple,
    row_count: int,
    panel_width: int = 260,
    row_height: int = 22,
    margin: int = 10,
) -> tuple[int, int]:
    sx, sy, sw, sh = screen
    mx, my, mw, mh = minimap
    panel_height = max(row_height, row_count * row_height)
    candidates = (
        (mx - panel_width - margin, my),
        (mx + mw + margin, my),
        (mx, my - panel_height - margin),
        (mx, my + mh + margin),
        (sx + margin, sy + margin),
    )
    best = (sx + margin, sy + margin)
    best_overlap = float("inf")
    for candidate_x, candidate_y in candidates:
        x = max(sx + margin, min(candidate_x, sx + sw - panel_width - margin))
        y = max(sy + margin, min(candidate_y, sy + sh - panel_height - margin))
        panel = (x, y, panel_width, panel_height)
        if not rectangles_intersect(panel, minimap):
            return x, y
        ix = max(0, min(x + panel_width, mx + mw) - max(x, mx))
        iy = max(0, min(y + panel_height, my + mh) - max(y, my))
        overlap = ix * iy
        if overlap < best_overlap:
            best, best_overlap = (x, y), overlap
    return best


def marker_layouts(
    champions: tuple[ChampionView, ...], collision_distance: float = 12.0
) -> dict[str, MarkerLayout]:
    stale = [
        champion
        for champion in champions
        if not champion.is_current and champion.position is not None
    ]
    groups: list[list[ChampionView]] = []
    for champion in stale:
        assert champion.position is not None
        group = next(
            (
                existing
                for existing in groups
                if any(
                    member.position is not None
                    and math.dist(champion.position, member.position) < collision_distance
                    for member in existing
                )
            ),
            None,
        )
        if group is None:
            groups.append([champion])
        else:
            group.append(champion)

    result: dict[str, MarkerLayout] = {}
    for group in groups:
        group.sort(
            key=lambda champion: champion.seconds_since_seen or 0.0,
            reverse=True,
        )
        for index, champion in enumerate(group):
            result[champion.identity.champion_name] = MarkerLayout(
                champion_name=champion.identity.champion_name,
                radius=10 + index * 3,
                icon_opacity=max(0.15, 0.35 - index * 0.08),
            )
    return result


def segment_intersects_rect(
    start: tuple[int, int], end: tuple[int, int], rectangle: RectTuple
) -> bool:
    x1, y1 = start
    x2, y2 = end
    bounds = (min(x1, x2), min(y1, y2), abs(x2 - x1) + 1, abs(y2 - y1) + 1)
    return rectangles_intersect(bounds, rectangle)
