"""Preprocessing utilities for missing values and scaling."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

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


def standardize_features(
    df: pd.DataFrame,
    columns: Iterable[str],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize selected columns using z-score scaling.

    Args:
        df: Input dataframe.
        columns: Feature columns to scale.
        scaler: Optional fitted scaler to reuse.

    Returns:
        Tuple of (scaled dataframe, fitted scaler).
    """
    cols = list(columns)
    if not cols:
        raise ValueError("columns must contain at least one feature name.")
    missing = [col for col in cols if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing columns in dataframe: {missing_str}")

    scaler = scaler or StandardScaler()
    df = df.copy()
    if hasattr(scaler, "mean_"):
        df[cols] = scaler.transform(df[cols])
    else:
        df[cols] = scaler.fit_transform(df[cols])
    return df, scaler
