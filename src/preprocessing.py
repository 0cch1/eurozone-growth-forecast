"""Preprocessing utilities for missing values and scaling."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.preprocessing import StandardScaler


def fill_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Fill missing values using forward/backward fill or mean.

    Args:
        df: Input dataframe.
        method: "ffill", "bfill", or "mean".

    Returns:
        Dataframe with missing values handled.
    """
    if method == "mean":
        return df.fillna(df.mean(numeric_only=True))
    if method == "bfill":
        return df.bfill()
    return df.ffill()


def standardize_features(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Standardize selected columns using z-score scaling.

    Args:
        df: Input dataframe.
        columns: Feature columns to scale.

    Returns:
        Dataframe with scaled columns.
    """
    scaler = StandardScaler()
    df = df.copy()
    df[list(columns)] = scaler.fit_transform(df[list(columns)])
    return df
