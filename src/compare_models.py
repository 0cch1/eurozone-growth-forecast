"""Compare baseline models on the processed dataset with optional visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

from .data_utils import load_and_prepare
from .evaluation import regression_metrics, time_series_cv_splits
from .models import build_models
from .preprocessing import standardize_features


def compare_models(n_splits: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run time-series CV and compare model performance.

    Returns:
        Tuple of (summary DataFrame sorted by RMSE, per-fold DataFrame).
    """
    df = load_and_prepare()
    feature_cols = [col for col in df.columns if col not in ("gdp_growth", "year")]

    X_df = df[feature_cols]
    y = df["gdp_growth"].to_numpy()

    results: List[Dict[str, float]] = []
    fold_records: List[Dict[str, object]] = []
    model_templates = build_models()

    for name, model_template in model_templates.items():
        split_metrics = []
        for fold_i, (train_idx, test_idx) in enumerate(
            time_series_cv_splits(len(df), n_splits=n_splits), start=1
        ):
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
            m = regression_metrics(y[test_idx], preds)
            split_metrics.append(m)
            fold_records.append({"model": name, "fold": fold_i, **m})

        mae_vals = [m["mae"] for m in split_metrics]
        rmse_vals = [m["rmse"] for m in split_metrics]
        r2_vals = [m["r2"] for m in split_metrics]
        results.append({
            "model": name,
            "mae": float(np.mean(mae_vals)),
            "mae_std": float(np.std(mae_vals)) if len(mae_vals) > 1 else 0.0,
            "rmse": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)) if len(rmse_vals) > 1 else 0.0,
            "r2": float(np.mean(r2_vals)),
            "r2_std": float(np.std(r2_vals)) if len(r2_vals) > 1 else 0.0,
        })

    # Random-walk baseline: predict y_{t} = y_{t-1} (previous year's actual).
    rw_split_metrics = []
    for fold_i, (train_idx, test_idx) in enumerate(
        time_series_cv_splits(len(df), n_splits=n_splits), start=1
    ):
        preds = y[np.array(test_idx) - 1]  # each test obs predicted by prior year
        m = regression_metrics(y[test_idx], preds)
        rw_split_metrics.append(m)
        fold_records.append({"model": "random_walk", "fold": fold_i, **m})

    rw_mae = [m["mae"] for m in rw_split_metrics]
    rw_rmse = [m["rmse"] for m in rw_split_metrics]
    rw_r2 = [m["r2"] for m in rw_split_metrics]
    results.append({
        "model": "random_walk",
        "mae": float(np.mean(rw_mae)),
        "mae_std": float(np.std(rw_mae)) if len(rw_mae) > 1 else 0.0,
        "rmse": float(np.mean(rw_rmse)),
        "rmse_std": float(np.std(rw_rmse)) if len(rw_rmse) > 1 else 0.0,
        "r2": float(np.mean(rw_r2)),
        "r2_std": float(np.std(rw_r2)) if len(rw_r2) > 1 else 0.0,
    })

    summary = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    folds = pd.DataFrame(fold_records).sort_values(["model", "fold"]).reset_index(drop=True)
    return summary, folds


def plot_comparison(
    results: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (8, 4),
    show: bool = False,
) -> None:
    """Plot MAE and RMSE comparison across models (bar chart)."""
    import matplotlib.pyplot as plt

    # Plot MAE and RMSE only; R² is in the results table (PDD §4.4 supplementary).
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    models = results["model"].tolist()
    x = range(len(models))

    axes[0].bar(x, results["mae"], color="steelblue", edgecolor="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("MAE")
    # Title removed – caption is in the report
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(x, results["rmse"], color="coral", edgecolor="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel("RMSE")
    # Title removed – caption is in the report
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(save_plot: bool = True) -> None:
    """Entry point for CLI usage: run comparison and optionally save plot."""
    results, _folds = compare_models()
    print("Model comparison (lower is better):")
    print(results.to_string(index=False))

    if save_plot:
        project_root = Path(__file__).resolve().parents[1]
        plot_path = project_root / "data" / "processed" / "model_comparison.png"
        try:
            plot_comparison(results, save_path=plot_path)
        except Exception as e:
            print(f"Plot skipped ({e}). Install matplotlib to enable visualisation.")


if __name__ == "__main__":
    main()
