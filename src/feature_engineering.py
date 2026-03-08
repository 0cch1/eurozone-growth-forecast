"""Feature engineering utilities for the Eurozone growth forecast pipeline."""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd


# Columns that should receive lag + change features by default when using
# build_full_features().  These match the 6-indicator panel_yearly.csv.
DEFAULT_LAG_COLS: List[str] = [
    "usd_eur_rate",
    "hicp_inflation",
    "short_term_rate",
    "unemployment_rate",
    "gov_debt_gdp",
]
DEFAULT_DIFF_COLS: List[str] = [
    "usd_eur_rate",
    "hicp_inflation",
    "short_term_rate",
]


def build_minimal_features(
    df: pd.DataFrame,
    target_col: str,
    lag_cols: Iterable[str],
    diff_cols: Iterable[str],
    lag: int = 1,
) -> pd.DataFrame:
    """Add basic lag and change features for time series models.

    Only operates on the columns explicitly requested; safe to call on
    a 2-column panel (backward-compatible with earlier runs).
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column missing: {target_col}")

    lag_cols = list(lag_cols)
    diff_cols = list(diff_cols)
    missing = [col for col in set(lag_cols + diff_cols) if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing columns in dataframe: {missing_str}")

    for col in lag_cols:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    for col in diff_cols:
        df[f"{col}_chg{lag}"] = df[col].diff(lag)
    return df.dropna()


def build_full_features(
    df: pd.DataFrame,
    target_col: str = "gdp_growth",
    lag_cols: Optional[List[str]] = None,
    diff_cols: Optional[List[str]] = None,
    lag: int = 1,
) -> pd.DataFrame:
    """Build the full feature set from a 6-indicator panel.

    - Keeps all raw level columns as features.
    - Adds 1-year lag for each column in lag_cols.
    - Adds 1-year first difference for each column in diff_cols.
    - Silently skips any column not present in df (handles subsets gracefully).
    - Drops the first row(s) that become NaN after lagging/differencing.

    Args:
        df: Yearly panel with at minimum ``target_col`` and level columns.
        target_col: Name of the dependent variable (excluded from features).
        lag_cols: Columns to lag (defaults to DEFAULT_LAG_COLS).
        diff_cols: Columns to difference (defaults to DEFAULT_DIFF_COLS).
        lag: Lag/difference order (default 1 year).

    Returns:
        DataFrame with original levels + derived features, NaN rows dropped.
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    _lag_cols = [c for c in (lag_cols or DEFAULT_LAG_COLS) if c in df.columns]
    _diff_cols = [c for c in (diff_cols or DEFAULT_DIFF_COLS) if c in df.columns]

    for col in _lag_cols:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    for col in _diff_cols:
        df[f"{col}_chg{lag}"] = df[col].diff(lag)

    return df.dropna()
