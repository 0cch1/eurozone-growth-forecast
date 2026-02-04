"""Minimal runnable pipeline demo using the project modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .evaluation import regression_metrics, time_series_cv_splits
from .feature_engineering import build_minimal_features
from .features import add_interactions, add_lag_features
from .models import build_models
from .preprocessing import fill_missing, standardize_features


def build_demo_dataset(n_periods: int = 24) -> pd.DataFrame:
    """Create a synthetic dataset for a quick smoke run."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=n_periods, freq="QE")
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


def load_processed_dataset() -> pd.DataFrame:
    """Load the processed yearly dataset if available."""
    project_root = Path(__file__).resolve().parents[1]
    processed_path = project_root / "data" / "processed" / "panel_yearly.csv"
    if not processed_path.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run `python -m src.build_dataset` first."
        )
    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed dataset is empty.")
    return df


def run_demo(use_real_data: bool = True) -> None:
    """Run an end-to-end demo with real or synthetic data."""
    if use_real_data:
        df = load_processed_dataset()
        df = df.sort_values("year").reset_index(drop=True)
        df = build_minimal_features(
            df,
            target_col="gdp_growth",
            lag_cols=["usd_eur_rate"],
            diff_cols=["usd_eur_rate"],
        )
    else:
        df = build_demo_dataset()
        df = fill_missing(df, method="ffill")
        df = add_lag_features(df, columns=["inflation", "unemployment"], lags=[1])
        df = add_interactions(df, columns=["inflation", "unemployment"])
        df = df.dropna()

    feature_cols = [col for col in df.columns if col != "gdp_growth"]
    X_df = df[feature_cols]
    y = df["gdp_growth"].to_numpy()

    models = build_models()
    model = models["linear"]

    split_metrics = []
    for train_idx, test_idx in time_series_cv_splits(len(df), n_splits=3):
        X_train, scaler = standardize_features(
            X_df.iloc[train_idx],
            columns=feature_cols,
        )
        X_test, _ = standardize_features(
            X_df.iloc[test_idx],
            columns=feature_cols,
            scaler=scaler,
        )
        model.fit(X_train.to_numpy(), y[train_idx])
        preds = model.predict(X_test.to_numpy())
        split_metrics.append(regression_metrics(y[test_idx], preds))

    print("Demo metrics:")
    for idx, metrics in enumerate(split_metrics, start=1):
        print(f"Split {idx}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")


if __name__ == "__main__":
    run_demo()
