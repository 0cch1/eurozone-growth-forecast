"""Entry point for main time-series CV results and plots.

This module runs the baseline model comparison using time-aware
cross-validation and writes canonical outputs under the ``results/``
directory:

- ``results/main_cv_results.csv``: table of model-level MAE, RMSE, R² (mean ± std over CV folds).
- ``results/main_cv_results.png``: bar chart visualisation of MAE/RMSE.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .compare_models import compare_models, plot_comparison


def main(n_splits: int = 3) -> None:
    """Run main time-series CV comparison and persist results/figures."""
    results, folds = compare_models(n_splits=n_splits)

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "main_cv_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"Saved main CV results to {csv_path}")

    folds_path = results_dir / "main_cv_per_fold.csv"
    folds.to_csv(folds_path, index=False)
    print(f"Saved per-fold results to {folds_path}")

    png_path = results_dir / "main_cv_results.png"
    try:
        plot_comparison(results, save_path=png_path)
    except Exception as e:
        print(f"Plot skipped ({e}). Install matplotlib to enable visualisation.")
    else:
        print(f"Saved main CV plot to {png_path}")


if __name__ == "__main__":
    main()

