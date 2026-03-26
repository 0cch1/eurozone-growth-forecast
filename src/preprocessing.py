"""Preprocessing utilities for missing values and scaling."""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def fill_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Fill missing values using forward/backward fill or mean.

    Args:
        df: Input dataframe.
        method: ``"ffill"`` (default), ``"bfill"``, or ``"mean"``.

    Returns:
        Dataframe with missing values handled.

    Warning:
        Using ``method="mean"`` computes the column mean over the entire
        dataframe. In a train/test split setting this can leak test-set
        information into training data.  Prefer ``"ffill"`` (the default)
        for time-series workflows, or call this function on the training
        partition only.
    """
    if method == "mean":
        warnings.warn(
            "fill_missing(method='mean') uses the global column mean which "
            "may cause data leakage in a train/test split setting. "
            "Prefer 'ffill' for time-series data or apply on training data only.",
            UserWarning,
            stacklevel=2,
        )
        return df.fillna(df.mean(numeric_only=True))
    if method == "bfill":
        return df.bfill()
    return df.ffill()


def standardize_features(
    df: pd.DataFrame,
    columns: Iterable[str],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardise selected columns using z-score scaling.

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
