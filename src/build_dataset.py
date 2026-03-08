"""Build a yearly EA-level dataset from raw sources.

Each indicator has a dedicated loader that knows its exact column structure.
This avoids silent empty-dataframe bugs from generic filters that don't apply
to every dataset (e.g. 'PC_GDP' / 'B1GQ' filters that are GDP-specific).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .indicators import get_indicator_specs


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _synthetic_gdp_yearly(year_min: int = 1999, year_max: Optional[int] = None) -> pd.DataFrame:
    """Fallback synthetic GDP growth when Eurostat data is unavailable."""
    if year_max is None:
        import datetime
        year_max = datetime.date.today().year - 1
    years = np.arange(year_min, year_max + 1, dtype=int)
    rng = np.random.default_rng(42)
    values = rng.normal(1.5, 2.0, size=len(years))
    return pd.DataFrame({"year": years, "gdp_growth": values})


def _eurostat_to_yearly(
    raw_path: Path,
    value_name: str,
    geo_values: tuple[str, ...] = ("EA20", "EA19"),
    freq_value: Optional[str] = "A",
    unit_value: Optional[str] = None,
    na_item_value: Optional[str] = None,
    extra_filters: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Generic Eurostat CSV → yearly panel.

    Only applies the filters that are relevant for the given dataset;
    callers pass None to skip a filter entirely.
    """
    df = pd.read_csv(raw_path)
    if df.empty:
        raise ValueError(f"Eurostat file is empty: {raw_path}")

    # Geography
    if "geo" in df.columns and geo_values:
        for geo in geo_values:
            if geo in df["geo"].unique():
                df = df[df["geo"] == geo]
                break

    # Frequency
    if freq_value and "freq" in df.columns and freq_value in df["freq"].unique():
        df = df[df["freq"] == freq_value]

    # Unit (optional)
    if unit_value and "unit" in df.columns and unit_value in df["unit"].unique():
        df = df[df["unit"] == unit_value]

    # National-accounts item (optional)
    if na_item_value and "na_item" in df.columns and na_item_value in df["na_item"].unique():
        df = df[df["na_item"] == na_item_value]

    # Extra per-dataset filters (optional)
    if extra_filters:
        for col, val in extra_filters.items():
            if col in df.columns and val in df[col].unique():
                df = df[df[col] == val]

    df = df.rename(columns={"time": "year", "value": value_name})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["year", value_name])
    df = df.groupby("year", as_index=False)[value_name].mean()

    if df.empty:
        raise ValueError(
            f"No rows left after filtering '{raw_path.name}' for {value_name}. "
            "Check geo/unit/freq filters."
        )
    return df


def _ecb_monthly_to_yearly(raw_path: Path, value_name: str) -> pd.DataFrame:
    """ECB CSV (TIME_PERIOD / OBS_VALUE columns) → yearly mean."""
    df = pd.read_csv(raw_path)
    df = df.rename(columns={"TIME_PERIOD": "time", "OBS_VALUE": value_name})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["time", value_name])
    df["year"] = df["time"].dt.year
    df = df.groupby("year", as_index=False)[value_name].mean()
    return df


# ---------------------------------------------------------------------------
# Per-indicator loaders (know the exact column structure of each raw file)
# ---------------------------------------------------------------------------

def _load_gdp_growth(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "eurostat_gdp_growth.csv"
    fallback = raw_dir / "eurostat_gdp_growth_fallback.csv"
    try:
        return _eurostat_to_yearly(
            path,
            value_name="gdp_growth",
            unit_value="PC_GDP",
            na_item_value="B1GQ",
        )
    except (ValueError, FileNotFoundError):
        pass
    if fallback.exists():
        print("GDP growth: using bundled fallback CSV.")
        try:
            return _eurostat_to_yearly(
                fallback,
                value_name="gdp_growth",
                unit_value="PC_GDP",
                na_item_value="B1GQ",
            )
        except (ValueError, FileNotFoundError):
            pass
    print("GDP growth: using synthetic data (no real data available).")
    return _synthetic_gdp_yearly()


def _load_usd_eur_rate(raw_dir: Path) -> pd.DataFrame:
    return _ecb_monthly_to_yearly(raw_dir / "ecb_exr_usd_eur.csv", "usd_eur_rate")


def _load_hicp_inflation(raw_dir: Path) -> pd.DataFrame:
    return _ecb_monthly_to_yearly(raw_dir / "ecb_hicp_inflation.csv", "hicp_inflation")


def _load_short_term_rate(raw_dir: Path) -> pd.DataFrame:
    return _ecb_monthly_to_yearly(raw_dir / "ecb_short_term_rate.csv", "short_term_rate")


def _load_unemployment_rate(raw_dir: Path) -> pd.DataFrame:
    """Unemployment rate: Eurostat une_rt_a — no na_item/unit=PC_GDP filter needed."""
    return _eurostat_to_yearly(
        raw_dir / "eurostat_unemployment_rate.csv",
        value_name="unemployment_rate",
        unit_value="PC_ACT",
        na_item_value=None,   # une_rt_a has no na_item column
        extra_filters={"sex": "T", "age": "Y15-74"},
    )


def _load_gov_debt_gdp(raw_dir: Path) -> pd.DataFrame:
    """Government debt/GDP: Eurostat tipsgo10 — unit=PC_GDP, sector=S13, na_item=GD."""
    return _eurostat_to_yearly(
        raw_dir / "eurostat_gov_debt_gdp.csv",
        value_name="gov_debt_gdp",
        unit_value="PC_GDP",
        na_item_value="GD",
        extra_filters={"sector": "S13"},
    )


# Map indicator name → loader function
_LOADERS = {
    "gdp_growth": _load_gdp_growth,
    "usd_eur_rate": _load_usd_eur_rate,
    "hicp_inflation": _load_hicp_inflation,
    "short_term_rate": _load_short_term_rate,
    "unemployment_rate": _load_unemployment_rate,
    "gov_debt_gdp": _load_gov_debt_gdp,
}


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def main() -> None:
    """Build a yearly processed dataset with all 6 indicators for modeling."""
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    indicators = get_indicator_specs()
    datasets: list[pd.DataFrame] = []
    indicator_rows = []
    loaded_names: list[str] = []

    for indicator in indicators:
        name = str(indicator["name"])
        source = str(indicator["source"])
        file_stem = str(indicator["file_stem"])
        value_name = str(indicator["value_name"])

        loader = _LOADERS.get(name)
        if loader is None:
            print(f"  [skip] No loader registered for '{name}'.")
            continue

        try:
            df = loader(raw_dir)
            n = len(df)
            yr_range = f"{int(df['year'].min())}–{int(df['year'].max())}" if n else "empty"
            print(f"  [ok]   {name:<22}  {n:3d} rows  ({yr_range})")
            datasets.append(df)
            loaded_names.append(name)
            indicator_rows.append(
                {
                    "name": name,
                    "source": source,
                    "file_stem": file_stem,
                    "value_name": value_name,
                    "description": str(indicator.get("description", "")),
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {name}: skipped — {exc}")

    if not datasets:
        raise RuntimeError("No indicators loaded; cannot build panel.")

    # Merge on year — left join keeps all years from GDP, then ffill/bfill
    # handles indicators with shorter coverage (e.g. unemployment starts later).
    panel = datasets[0]
    for df in datasets[1:]:
        panel = panel.merge(df, on="year", how="left")
    panel = panel.sort_values("year").reset_index(drop=True)

    total_missing = panel.isnull().sum()
    cols_with_gaps = total_missing[total_missing > 0]
    if not cols_with_gaps.empty:
        print("\nMissing values per column (will be forward/backward-filled by modelling pipeline):")
        for col, n in cols_with_gaps.items():
            print(f"  {col}: {n} missing")

    output_path = processed_dir / "panel_yearly.csv"
    panel.to_csv(output_path, index=False)
    print(f"\nSaved processed panel  → {output_path}")
    print(f"Shape: {panel.shape}  |  Years: {int(panel['year'].min())}–{int(panel['year'].max())}")
    print(f"Columns: {list(panel.columns)}")

    indicator_path = processed_dir / "indicator_list.csv"
    pd.DataFrame(indicator_rows).to_csv(indicator_path, index=False)
    print(f"Saved indicator list   → {indicator_path}")


if __name__ == "__main__":
    main()
