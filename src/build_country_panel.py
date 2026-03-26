"""Build a country-level GDP growth panel for map visualisation.

This script fetches Eurostat GDP growth data (tec00115) for a small set of
euro area countries plus a few non-EU comparator countries and aggregates it
to an annual country-year panel:

- Input: Eurostat dataset tec00115 via the existing data_loader helpers.
- Output: data/processed/panel_country_yearly.csv with columns:
  country (ISO2), year, gdp_growth (percent change).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import fetch_eurostat_dataset


# Euro area members (ISO2); Eurostat uses EL for Greece in tec00115.
EURO_AREA_COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "EL"]

# Non-EU comparators often available in Eurostat growth tables.
NON_EU_COMPARATORS = ["UK", "CH", "NO", "IS"]

COUNTRIES = EURO_AREA_COUNTRIES + NON_EU_COMPARATORS


def build_country_panel(since_year: int = 2000) -> Path:
    """Fetch Eurostat GDP growth (tec00115) and build a country-level panel.

    The panel is intended for visual comparison rather than for the primary
    modelling pipeline, so it includes the euro area sample and a small set of
    non-EU benchmark countries when Eurostat provides them.
    """
    df = fetch_eurostat_dataset(
        dataset_id="tec00115",
        filters=None,
        since_time_period=str(since_year),
    )

    # Keep annual frequency, real GDP growth in volume terms, and the main GDP aggregate.
    if "freq" in df.columns:
        df = df[df["freq"] == "A"]
    if "unit" in df.columns:
        # For tec00115 the main growth measure is CLV_PCH_PRE (volume, % change).
        df = df[df["unit"] == "CLV_PCH_PRE"]
    if "na_item" in df.columns:
        df = df[df["na_item"] == "B1GQ"]

    # Restrict to a small, illustrative subset of euro area countries plus
    # a few non-EU comparator countries for presentation/demo purposes.
    df = df[df["geo"].isin(COUNTRIES)].copy()
    if df.empty:
        raise ValueError("No matching country-level GDP rows found in tec00115.")

    # Normalise columns.
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column in Eurostat payload.")
    df["year"] = pd.to_numeric(df["time"], errors="coerce").astype("Int64")
    df["gdp_growth"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "gdp_growth"])

    df = df[["geo", "year", "gdp_growth"]].rename(columns={"geo": "country"})
    df = df.sort_values(["year", "country"]).reset_index(drop=True)

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "panel_country_yearly.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved country-level panel to {out_path}")
    countries_included = sorted(df["country"].unique())
    print(f"Countries included: {countries_included}")
    euro_area_included = sorted(c for c in countries_included if c in EURO_AREA_COUNTRIES)
    comparator_included = sorted(c for c in countries_included if c in NON_EU_COMPARATORS)
    print(f"Euro area countries: {euro_area_included}")
    print(f"Non-EU comparators: {comparator_included}")
    # Warn if the latest year has fewer countries (Eurostat release lag).
    latest_year = int(df["year"].max())
    in_latest = set(df[df["year"] == latest_year]["country"])
    missing = set(COUNTRIES) - in_latest
    if missing:
        print(f"Note: {latest_year} has partial data; not yet released for: {sorted(missing)}")
    return out_path


def main() -> None:
    """CLI entry point: build the country-level panel with default settings."""
    build_country_panel(since_year=2000)


if __name__ == "__main__":
    main()

