"""Fetch and store real data from Eurostat and ECB."""

from __future__ import annotations

from pathlib import Path

from .data_loader import fetch_indicators, save_raw_dataset
from .indicators import get_indicator_specs


def main() -> None:
    """Download default indicators and save to data/raw."""
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    indicators = get_indicator_specs()
    datasets = fetch_indicators(indicators)
    for indicator in indicators:
        name = str(indicator["name"])
        file_stem = str(indicator["file_stem"])
        df = datasets[name]
        output_path = raw_dir / f"{file_stem}.csv"
        save_raw_dataset(df, str(output_path))
        print(f"Saved {name} to {output_path}")


if __name__ == "__main__":
    main()
