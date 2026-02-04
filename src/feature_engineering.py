"""Minimal feature engineering utilities."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def build_minimal_features(
    df: pd.DataFrame,
    target_col: str,
    lag_cols: Iterable[str],
    diff_cols: Iterable[str],
    lag: int = 1,
) -> pd.DataFrame:
    """Add basic lag and change features for time series models."""
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
