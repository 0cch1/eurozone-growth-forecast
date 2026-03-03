"""Time-aware evaluation utilities."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def time_series_cv_splits(n_samples: int, n_splits: int = 5) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Generate time series cross-validation splits.

    Args:
        n_samples: Total number of observations.
        n_splits: Number of splits.

    Returns:
        Generator of train/test indices.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    return splitter.split(np.arange(n_samples))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute common regression metrics (PDD §4.4: MAE, RMSE, R²).

    Returns:
        Dictionary with mae, rmse, r2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}
