"""Minimal runnable pipeline demo using the project modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import regression_metrics, time_series_cv_splits
from features import add_interactions, add_lag_features
from models import build_models
from preprocessing import fill_missing, standardize_features


def build_demo_dataset(n_periods: int = 24) -> pd.DataFrame:
    """Create a synthetic dataset for a quick smoke run."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=n_periods, freq="Q")
    df = pd.DataFrame(
        {
            "gdp_growth": rng.normal(0.4, 0.2, size=n_periods),
            "inflation": rng.normal(1.8, 0.3, size=n_periods),
            "unemployment": rng.normal(7.5, 0.5, size=n_periods),
        },
        index=dates,
    )
    df.iloc[5, 1] = np.nan
    return df


def run_demo() -> None:
    """Run a tiny end-to-end demo (no real API calls)."""
    df = build_demo_dataset()
    df = fill_missing(df, method="ffill")
    df = add_lag_features(df, columns=["inflation", "unemployment"], lags=[1])
    df = add_interactions(df, columns=["inflation", "unemployment"])
    df = df.dropna()

    feature_cols = [col for col in df.columns if col != "gdp_growth"]
    df = standardize_features(df, columns=feature_cols)

    X = df[feature_cols].to_numpy()
    y = df["gdp_growth"].to_numpy()

    models = build_models()
    model = models["linear"]

    split_metrics = []
    for train_idx, test_idx in time_series_cv_splits(len(df), n_splits=3):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        split_metrics.append(regression_metrics(y[test_idx], preds))

    print("Demo metrics:")
    for idx, metrics in enumerate(split_metrics, start=1):
        print(f"Split {idx}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")


if __name__ == "__main__":
    run_demo()
