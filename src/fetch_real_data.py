"""Fetch and store real data from Eurostat and ECB."""

from __future__ import annotations

from pathlib import Path

from .data_loader import fetch_default_indicators, save_raw_dataset


def main() -> None:
    """Download default indicators and save to data/raw."""
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    datasets = fetch_default_indicators()
    for name, df in datasets.items():
        output_path = raw_dir / f"{name}.csv"
        save_raw_dataset(df, str(output_path))
        print(f"Saved {name} to {output_path}")


if __name__ == "__main__":
    main()
