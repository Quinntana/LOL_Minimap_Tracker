"""Perform an explicit, non-destructive Data Dragon contract check."""

from __future__ import annotations

import requests

BASE_URL = "https://ddragon.leagueoflegends.com"


def verify() -> tuple[str, int, str]:
    versions_response = requests.get(f"{BASE_URL}/api/versions.json", timeout=15)
    versions_response.raise_for_status()
    versions = versions_response.json()
    if not isinstance(versions, list) or not versions or not isinstance(versions[0], str):
        raise ValueError("versions.json has an unexpected shape")
    version = versions[0]

    metadata_response = requests.get(
        f"{BASE_URL}/cdn/{version}/data/en_US/champion.json", timeout=15
    )
    metadata_response.raise_for_status()
    metadata = metadata_response.json().get("data")
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("champion.json has an unexpected shape")
    first = next(iter(metadata.values()))
    filename = first.get("image", {}).get("full")
    if not isinstance(filename, str):
        raise ValueError("champion metadata does not include image.full")

    portrait_response = requests.get(
        f"{BASE_URL}/cdn/{version}/img/champion/{filename}", timeout=15
    )
    portrait_response.raise_for_status()
    content_type = portrait_response.headers.get("Content-Type", "").split(";", 1)[0]
    if content_type != "image/png":
        raise ValueError(f"Unexpected portrait content type: {content_type}")
    return version, len(metadata), filename


if __name__ == "__main__":
    checked_version, champion_count, checked_portrait = verify()
    print(
        f"Data Dragon OK: version={checked_version}, "
        f"champions={champion_count}, portrait={checked_portrait}"
    )
