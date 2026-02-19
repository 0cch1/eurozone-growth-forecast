"""Optional AutoML baseline with time-based holdout (same data as compare_models)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import regression_metrics
from .feature_engineering import build_minimal_features
from .preprocessing import standardize_features

FLAML_AVAILABLE = False
try:
    from flaml import AutoML  # type: ignore
    FLAML_AVAILABLE = True
except ImportError:
    AutoML = None  # type: ignore


def _load_and_prepare():
    """Same data pipeline as compare_models / run_interpretation."""
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "data" / "processed" / "panel_yearly.csv"
    if not path.exists():
        raise FileNotFoundError("Run python -m src.build_dataset first.")
    df = pd.read_csv(path).sort_values("year").reset_index(drop=True)
    df = build_minimal_features(
        df,
        target_col="gdp_growth",
        lag_cols=["usd_eur_rate"],
        diff_cols=["usd_eur_rate"],
    )
    return df


def run_automl_baseline(
    test_frac: float = 0.2,
    time_budget: int = 120,
    random_state: int = 42,
) -> dict:
    """Train FLAML AutoML on temporal train set and evaluate on holdout.

    Uses a single time-based split (last test_frac of samples as test) to match
    time-aware evaluation. Compare the returned MAE/RMSE with compare_models output.

    Args:
        test_frac: Fraction of samples for holdout (from the end).
        time_budget: FLAML time budget in seconds.
        random_state: Seed for reproducibility.

    Returns:
        Dict with mae, rmse, n_train, n_test, best_estimator (name).
    """
    if not FLAML_AVAILABLE:
        return {
            "error": "FLAML not installed. Install with: pip install flaml",
            "mae": None,
            "rmse": None,
        }

    df = _load_and_prepare()
    feature_cols = [c for c in df.columns if c != "gdp_growth"]
    X_df = df[feature_cols]
    y = df["gdp_growth"].to_numpy()
    n = len(y)
    n_test = max(1, int(n * test_frac))
    n_train = n - n_test
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n)

    X_train_df, scaler = standardize_features(X_df.iloc[train_idx], columns=feature_cols)
    X_test_df, _ = standardize_features(
        X_df.iloc[test_idx], columns=feature_cols, scaler=scaler
    )
    X_train = X_train_df.to_numpy()
    X_test = X_test_df.to_numpy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    automl = AutoML()
    automl.fit(
        X_train,
        y_train,
        task="regression",
        time_budget=time_budget,
        metric="mae",
        seed=random_state,
        verbose=0,
    )
    preds = automl.predict(X_test)
    metrics = regression_metrics(y_test, preds)
    out = {
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "n_train": n_train,
        "n_test": n_test,
        "best_estimator": getattr(automl, "best_estimator", "AutoML"),
    }
    return out


def main() -> None:
    """Run AutoML baseline and print results for comparison with compare_models."""
    if not FLAML_AVAILABLE:
        print("FLAML not installed. Install with: pip install flaml")
        return
    print("Running FLAML AutoML (time-based holdout, last 20% as test)...")
    result = run_automl_baseline(test_frac=0.2, time_budget=120)
    if result.get("error"):
        print(result["error"])
        return
    print("AutoML baseline (holdout):")
    print(f"  MAE  = {result['mae']:.4f}")
    print(f"  RMSE = {result['rmse']:.4f}")
    print(f"  Best estimator: {result.get('best_estimator', '?')}")
    print("Compare with: python -m src.compare_models")


if __name__ == "__main__":
    main()
