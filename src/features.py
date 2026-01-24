"""Feature engineering utilities (lags and interactions)."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import pandas as pd


def add_lag_features(df: pd.DataFrame, columns: Iterable[str], lags: Iterable[int]) -> pd.DataFrame:
    """Create lagged features for time series.

    Args:
        df: Input dataframe indexed by time.
        columns: Columns to lag.
        lags: List of lag steps.

    Returns:
        Dataframe with lagged columns added.
    """
    cols = list(columns)
    missing = [col for col in cols if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing columns in dataframe: {missing_str}")

    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_interactions(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Create pairwise interaction terms.

    Args:
        df: Input dataframe.
        columns: Columns to create interactions from.

    Returns:
        Dataframe with interaction features.
    """
    cols = list(columns)
    missing = [col for col in cols if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing columns in dataframe: {missing_str}")

    df = df.copy()
    for left, right in combinations(cols, 2):
        df[f"{left}_x_{right}"] = df[left] * df[right]
    return df
