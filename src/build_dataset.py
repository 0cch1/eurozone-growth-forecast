"""Build a yearly EA-level dataset from raw sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_eurostat_gdp(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    if df.empty:
        raise ValueError(
            "Eurostat GDP growth file is empty. Re-run fetch_real_data or adjust filters."
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

    df = df.rename(columns={"time": "year", "value": "gdp_growth"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["gdp_growth"] = pd.to_numeric(df["gdp_growth"], errors="coerce")
    df = df.dropna(subset=["year", "gdp_growth"])
    df = df.groupby("year", as_index=False)["gdp_growth"].mean()
    return df


def _load_ecb_fx(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df = df.rename(columns={"TIME_PERIOD": "time", "OBS_VALUE": "usd_eur_rate"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["usd_eur_rate"] = pd.to_numeric(df["usd_eur_rate"], errors="coerce")
    df = df.dropna(subset=["time", "usd_eur_rate"])
    df["year"] = df["time"].dt.year
    df = df.groupby("year", as_index=False)["usd_eur_rate"].mean()
    return df


def main() -> None:
    """Build a yearly processed dataset for modeling."""
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    eurostat_path = raw_dir / "eurostat_gdp_growth.csv"
    ecb_path = raw_dir / "ecb_exr_usd_eur.csv"

    eurostat_df = _load_eurostat_gdp(eurostat_path)
    ecb_df = _load_ecb_fx(ecb_path)

    panel = eurostat_df.merge(ecb_df, on="year", how="inner")
    panel = panel.sort_values("year").reset_index(drop=True)

    output_path = processed_dir / "panel_yearly.csv"
    panel.to_csv(output_path, index=False)
    print(f"Saved processed dataset to {output_path}")


if __name__ == "__main__":
    main()
