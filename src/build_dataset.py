"""Build a yearly EA-level dataset from raw sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import get_indicator_specs


def _synthetic_gdp_yearly(year_min: int = 1999, year_max: int | None = None) -> pd.DataFrame:
    """Minimal synthetic GDP growth (percent) for fallback when Eurostat is empty."""
    if year_max is None:
        import datetime
        year_max = datetime.date.today().year - 1
    years = np.arange(year_min, year_max + 1, dtype=int)
    rng = np.random.default_rng(42)
    values = rng.normal(1.5, 2.0, size=len(years))
    return pd.DataFrame({"year": years, "gdp_growth": values})


def _load_eurostat_yearly(raw_path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    if df.empty:
        raise ValueError(
            f"Eurostat file is empty: {raw_path}. Re-run fetch_real_data or adjust filters."
        )

    if "geo" in df.columns:
        if "EA20" in df["geo"].unique():
            df = df[df["geo"] == "EA20"]
        elif "EA19" in df["geo"].unique():
            df = df[df["geo"] == "EA19"]

    if "unit" in df.columns and "PC_GDP" in df["unit"].unique():
        df = df[df["unit"] == "PC_GDP"]

    if "freq" in df.columns and "A" in df["freq"].unique():
        df = df[df["freq"] == "A"]

    if "na_item" in df.columns and "B1GQ" in df["na_item"].unique():
        df = df[df["na_item"] == "B1GQ"]

    df = df.rename(columns={"time": "year", "value": value_name})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["year", value_name])
    df = df.groupby("year", as_index=False)[value_name].mean()
    return df


def _load_ecb_monthly_to_yearly(raw_path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df = df.rename(columns={"TIME_PERIOD": "time", "OBS_VALUE": value_name})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["time", value_name])
    df["year"] = df["time"].dt.year
    df = df.groupby("year", as_index=False)[value_name].mean()
    return df


def main() -> None:
    """Build a yearly processed dataset for modeling."""
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    indicators = get_indicator_specs()

    datasets = []
    indicator_rows = []
    for indicator in indicators:
        name = str(indicator["name"])
        source = indicator["source"]
        file_stem = str(indicator["file_stem"])
        value_name = str(indicator["value_name"])

        raw_path = raw_dir / f"{file_stem}.csv"
        if source == "eurostat":
            fallback_path = raw_dir / "eurostat_gdp_growth_fallback.csv"
            try:
                df = _load_eurostat_yearly(raw_path, value_name)
            except (ValueError, FileNotFoundError):
                if fallback_path.exists():
                    print(
                        "Eurostat file missing or empty. Using bundled backup "
                        "(data/raw/eurostat_gdp_growth_fallback.csv)."
                    )
                    df = _load_eurostat_yearly(fallback_path, value_name)
                else:
                    print(
                        "Eurostat file missing or empty. Using synthetic GDP growth for this run."
                    )
                    df = _synthetic_gdp_yearly()
        elif source == "ecb":
            df = _load_ecb_monthly_to_yearly(raw_path, value_name)
        else:
            raise ValueError(f"Unsupported source: {source}")

        datasets.append(df)
        indicator_rows.append(
            {
                "name": name,
                "source": source,
                "file_stem": file_stem,
                "value_name": value_name,
                "description": indicator.get("description", ""),
            }
        )

    panel = datasets[0]
    for df in datasets[1:]:
        panel = panel.merge(df, on="year", how="inner")
    panel = panel.sort_values("year").reset_index(drop=True)

    output_path = processed_dir / "panel_yearly.csv"
    panel.to_csv(output_path, index=False)
    print(f"Saved processed dataset to {output_path}")

    indicator_path = processed_dir / "indicator_list.csv"
    pd.DataFrame(indicator_rows).to_csv(indicator_path, index=False)
    print(f"Saved indicator list to {indicator_path}")


if __name__ == "__main__":
    main()
