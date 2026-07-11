"""Plot recorded timeline positions over a minimap image."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_timeline(timeline: Path, minimap: Path) -> None:
    data = pd.read_csv(timeline)
    image = plt.imread(minimap)
    map_height, map_width = image.shape[:2]
    width = max(1, int(data["X"].max()) + 1)
    height = max(1, int(data["Y"].max()) + 1)
    data["x_scaled"] = data["X"] / width * map_width
    data["y_scaled"] = (height - data["Y"]) / height * map_height
    champions = sorted(data["champion"].unique())
    palette = sns.color_palette("husl", len(champions))

    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(image, extent=[0, map_width, 0, map_height], alpha=0.65)
    for color, champion in zip(palette, champions, strict=True):
        rows = data[data["champion"] == champion]
        axis.plot(
            rows["x_scaled"],
            rows["y_scaled"],
            marker="o",
            markersize=3,
            color=color,
            label=champion,
        )
    axis.set(xlim=(0, map_width), ylim=(0, map_height), title="Champion movement")
    axis.legend()
    figure.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--map", type=Path, default=Path("data/Minimap.png"))
    args = parser.parse_args()
    plot_timeline(args.timeline, args.map)


if __name__ == "__main__":
    main()
