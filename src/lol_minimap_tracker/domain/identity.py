"""Deterministic enemy identity assignment."""

from __future__ import annotations

from collections.abc import Iterable

from .models import EnemyIdentity, Role, RosterMember

ROLE_ORDER = {
    Role.TOP: 0,
    Role.JUNGLE: 1,
    Role.MIDDLE: 2,
    Role.BOTTOM: 3,
    Role.UTILITY: 4,
    Role.UNKNOWN: 5,
}

PREFERRED_COLORS = {
    Role.TOP: "#E69F00",
    Role.JUNGLE: "#009E73",
    Role.MIDDLE: "#56B4E9",
    Role.BOTTOM: "#D55E00",
    Role.UTILITY: "#CC79A7",
    Role.UNKNOWN: "#B8B8B8",
}

EXTENDED_PALETTE = (
    "#E69F00",
    "#009E73",
    "#56B4E9",
    "#D55E00",
    "#CC79A7",
    "#0072B2",
    "#F0E442",
    "#B8B8B8",
)

ROLE_ICONS = {
    Role.TOP: "position-top.svg",
    Role.JUNGLE: "position-jungle.svg",
    Role.MIDDLE: "position-middle.svg",
    Role.BOTTOM: "position-bottom.svg",
    Role.UTILITY: "position-utility.svg",
    Role.UNKNOWN: "position-unknown.svg",
}


def assign_identities(members: Iterable[RosterMember]) -> tuple[EnemyIdentity, ...]:
    """Assign stable, unique colors in deterministic role/name order."""
    ordered = sorted(
        members,
        key=lambda member: (ROLE_ORDER[member.role], member.champion_name.casefold()),
    )
    used: set[str] = set()
    identities: list[EnemyIdentity] = []

    for member in ordered:
        preferred = PREFERRED_COLORS[member.role]
        color = (
            preferred
            if preferred not in used
            else next(candidate for candidate in EXTENDED_PALETTE if candidate not in used)
        )
        used.add(color)
        identities.append(
            EnemyIdentity(
                champion_name=member.champion_name,
                role=member.role,
                color=color,
                role_icon=ROLE_ICONS[member.role],
            )
        )
    return tuple(identities)
