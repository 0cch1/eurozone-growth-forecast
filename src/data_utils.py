"""Shared data loading and preparation utilities used across pipeline entry points."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .feature_engineering import build_full_features


# Keep the raw processed panel intact, but exclude variables with insufficient
# early-year coverage from the shared modelling pipeline. This avoids
# time-leaking backfills while preserving a usable sample size for the final
# forecasting comparison.
MODEL_EXCLUDED_COLS = {"unemployment_rate"}


def get_project_root() -> Path:
    """Return the project root directory (one level above ``src/``)."""
    return Path(__file__).resolve().parents[1]


def load_processed_panel() -> pd.DataFrame:
    """Load ``data/processed/panel_yearly.csv`` and raise if missing/empty."""
    path = get_project_root() / "data" / "processed" / "panel_yearly.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run `python -m src.build_dataset` first."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Processed dataset is empty.")
    return df


def load_and_prepare(target_col: str = "gdp_growth") -> pd.DataFrame:
    """Load processed panel, fill gaps, and build full feature set.

    This is the standard data preparation shared by compare_models,
    run_interpretation, and automl_experiments.
    """
    df = load_processed_panel()
    df = df.sort_values("year").reset_index(drop=True)
    df = df.drop(columns=[c for c in MODEL_EXCLUDED_COLS if c in df.columns])
    # Forward-fill preserves time ordering; avoid backward-fill here because it
    # would leak future observations into earlier rows before CV splitting.
    df = df.ffill()
    df = build_full_features(df, target_col=target_col)
    return df
