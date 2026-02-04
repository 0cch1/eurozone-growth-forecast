"""Compare baseline models on the processed dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone

from .evaluation import regression_metrics, time_series_cv_splits
from .feature_engineering import build_minimal_features
from .models import build_models
from .preprocessing import standardize_features


def _load_processed_dataset() -> pd.DataFrame:
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


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("year").reset_index(drop=True)
    df = build_minimal_features(
        df,
        target_col="gdp_growth",
        lag_cols=["usd_eur_rate"],
        diff_cols=["usd_eur_rate"],
    )
    return df


def compare_models(n_splits: int = 3) -> pd.DataFrame:
    """Run time-series CV and compare model performance."""
    df = _prepare_features(_load_processed_dataset())
    feature_cols = [col for col in df.columns if col != "gdp_growth"]

    X_df = df[feature_cols]
    y = df["gdp_growth"].to_numpy()

    results: List[Dict[str, float]] = []
    model_templates = build_models()

    for name, model_template in model_templates.items():
        split_metrics = []
        for train_idx, test_idx in time_series_cv_splits(len(df), n_splits=n_splits):
            X_train, scaler = standardize_features(
                X_df.iloc[train_idx],
                columns=feature_cols,
            )
            X_test, _ = standardize_features(
                X_df.iloc[test_idx],
                columns=feature_cols,
                scaler=scaler,
            )
            model = clone(model_template)
            model.fit(X_train.to_numpy(), y[train_idx])
            preds = model.predict(X_test.to_numpy())
            split_metrics.append(regression_metrics(y[test_idx], preds))

        mae = float(np.mean([m["mae"] for m in split_metrics]))
        rmse = float(np.mean([m["rmse"] for m in split_metrics]))
        results.append({"model": name, "mae": mae, "rmse": rmse})

    return pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)


def main() -> None:
    """Entry point for CLI usage."""
    results = compare_models()
    print("Model comparison (lower is better):")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
