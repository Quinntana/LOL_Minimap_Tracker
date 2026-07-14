"""Verify the current regional Data Dragon cooldown contract without a live game."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from lol_minimap_tracker.domain.models import Role, RosterMember, SummonerSpellRef
from lol_minimap_tracker.integrations.cooldown_data import CooldownDataDragonClient


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory(prefix="lol-cooldown-verify-") as directory:
        client = CooldownDataDragonClient(Path(directory), logging.getLogger("verify"))
        member = RosterMember(
            champion_name="Aatrox",
            champion_id="Aatrox",
            role=Role.TOP,
            participant_id="contract-check",
            level=11,
            summoner_spells=(
                SummonerSpellRef("SummonerFlash", "Flash"),
                SummonerSpellRef("SummonerDot", "Ignite"),
            ),
        )
        loadouts = client.get_loadouts((member,))
        if len(loadouts) != 1:
            raise RuntimeError("Data Dragon returned no Aatrox cooldown loadout")
        loadout = loadouts[0]
        ultimate = loadout.ultimate.duration_for_level(11)
        spells = tuple(spell.duration_for_level(11) for spell in loadout.summoner_spells)
        icons = (
            loadout.champion_icon_path,
            loadout.ultimate.icon_path,
            *(spell.icon_path for spell in loadout.summoner_spells),
        )
        if ultimate is None or any(duration is None for duration in spells):
            raise RuntimeError("Current static cooldown values could not be resolved")
        if any(path is None or not path.exists() for path in icons):
            raise RuntimeError("Current static cooldown icons could not be cached")
        print(f"Aatrox R at level 11: {ultimate:g}s")
        print(f"Flash / Ignite: {spells[0]:g}s / {spells[1]:g}s")
        print("Champion, ultimate, and summoner icons cached successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
