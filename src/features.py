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
    df = df.copy()
    for col in columns:
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
    df = df.copy()
    for left, right in combinations(columns, 2):
        df[f"{left}_x_{right}"] = df[left] * df[right]
    return df
