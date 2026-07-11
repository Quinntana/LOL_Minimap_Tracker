from lol_minimap_tracker.domain.identity import assign_identities
from lol_minimap_tracker.domain.models import Role, RosterMember


def test_role_aliases() -> None:
    assert Role.from_api("mid") is Role.MIDDLE
    assert Role.from_api("support") is Role.UTILITY
    assert Role.from_api(None) is Role.UNKNOWN


def test_normal_roster_receives_preferred_unique_colors() -> None:
    identities = assign_identities(
        (
            RosterMember("Support", Role.UTILITY),
            RosterMember("Top", Role.TOP),
            RosterMember("Mid", Role.MIDDLE),
            RosterMember("Bot", Role.BOTTOM),
            RosterMember("Jungle", Role.JUNGLE),
        )
    )
    assert [identity.champion_name for identity in identities] == [
        "Top",
        "Jungle",
        "Mid",
        "Bot",
        "Support",
    ]
    assert len({identity.color for identity in identities}) == 5
    assert identities[0].color == "#E69F00"
    assert identities[1].role_icon == "position-jungle.svg"


def test_duplicate_roles_use_deterministic_fallback_colors() -> None:
    members = (
        RosterMember("Zulu", Role.TOP),
        RosterMember("Alpha", Role.TOP),
        RosterMember("Mystery", Role.UNKNOWN),
    )
    first = assign_identities(members)
    second = assign_identities(reversed(members))
    assert first == second
    assert len({identity.color for identity in first}) == 3
