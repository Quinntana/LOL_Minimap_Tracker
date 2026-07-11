"""Remove an inclusive row range from a timeline CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def trim_timeline(source: Path, start: int, end: int, output: Path) -> None:
    data = pd.read_csv(source)
    if start < 0 or end < start or end >= len(data):
        raise ValueError(f"Invalid inclusive row range {start}:{end} for {len(data)} rows")
    trimmed = pd.concat([data.iloc[:start], data.iloc[end + 1 :]], ignore_index=True)
    trimmed.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    trim_timeline(args.source, args.start, args.end, args.output or args.source)


if __name__ == "__main__":
    main()
