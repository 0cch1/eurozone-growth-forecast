"""Smoke tests for core pipeline utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import regression_metrics, time_series_cv_splits
from src.features import add_lag_features, add_interactions
from src.feature_engineering import build_minimal_features, build_full_features
from src.models import build_models
from src.preprocessing import fill_missing, standardize_features
from src.interpretation import feature_display_name


# ---------------------------------------------------------------------------
# Original smoke test
# ---------------------------------------------------------------------------

def test_smoke_pipeline() -> None:
    df = pd.DataFrame(
        {
            "gdp_growth": [0.2, 0.3, 0.5, 0.4, 0.6],
            "inflation": [1.8, 1.9, None, 2.0, 1.7],
        }
    )
    df = fill_missing(df, method="ffill")
    df = add_lag_features(df, columns=["inflation"], lags=[1]).dropna()

    X = df[["inflation", "inflation_lag1"]].to_numpy()
    y = df["gdp_growth"].to_numpy()

    models = build_models()
    model = models["linear"]
    model.fit(X, y)
    preds = model.predict(X)

    metrics = regression_metrics(y, preds)
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert "r2" in metrics


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _sample_panel() -> pd.DataFrame:
    """Synthetic 6-indicator panel resembling panel_yearly.csv."""
    rng = np.random.default_rng(0)
    n = 12
    return pd.DataFrame({
        "year": range(2000, 2000 + n),
        "gdp_growth": rng.normal(1.5, 2.0, n),
        "usd_eur_rate": rng.normal(1.15, 0.1, n),
        "hicp_inflation": rng.normal(2.0, 0.5, n),
        "unemployment_rate": rng.normal(9.0, 1.0, n),
        "short_term_rate": rng.normal(2.0, 1.5, n),
        "gov_debt_gdp": rng.normal(85.0, 5.0, n),
    })


def test_build_minimal_features() -> None:
    df = _sample_panel()
    result = build_minimal_features(
        df,
        target_col="gdp_growth",
        lag_cols=["usd_eur_rate"],
        diff_cols=["usd_eur_rate"],
    )
    assert "usd_eur_rate_lag1" in result.columns
    assert "usd_eur_rate_chg1" in result.columns
    assert result["usd_eur_rate_lag1"].notna().all()
    assert len(result) == len(df) - 1


def test_build_full_features() -> None:
    df = _sample_panel()
    result = build_full_features(df, target_col="gdp_growth")
    expected_lags = [
        "usd_eur_rate_lag1", "hicp_inflation_lag1",
        "unemployment_rate_lag1", "short_term_rate_lag1", "gov_debt_gdp_lag1",
    ]
    expected_diffs = [
        "usd_eur_rate_chg1", "hicp_inflation_chg1", "short_term_rate_chg1",
    ]
    for col in expected_lags + expected_diffs:
        assert col in result.columns, f"Missing derived column: {col}"
    assert result.notna().all().all()
    # Loses 1 row from lag/diff NaN at start + 1 row from target shift at end
    assert len(result) == len(df) - 2


def test_build_full_features_no_shift() -> None:
    """When forecast_horizon=0, no target shift; only lag rows are dropped."""
    df = _sample_panel()
    result = build_full_features(df, target_col="gdp_growth", forecast_horizon=0)
    assert result.notna().all().all()
    assert len(result) == len(df) - 1


def test_build_full_features_missing_target_raises() -> None:
    df = _sample_panel().drop(columns=["gdp_growth"])
    with pytest.raises(ValueError, match="Target column"):
        build_full_features(df, target_col="gdp_growth")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_standardize_features_and_reuse_scaler() -> None:
    df = _sample_panel()
    cols = ["usd_eur_rate", "hicp_inflation"]
    scaled, scaler = standardize_features(df, columns=cols)
    for col in cols:
        assert abs(scaled[col].mean()) < 1e-10
        assert abs(scaled[col].std(ddof=0) - 1.0) < 1e-10

    new_row = df.iloc[:1].copy()
    scaled_new, _ = standardize_features(new_row, columns=cols, scaler=scaler)
    assert scaled_new.shape == new_row.shape


def test_fill_missing_methods() -> None:
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, None]})
    assert fill_missing(df, "ffill")["a"].iloc[1] == 1.0
    assert fill_missing(df, "bfill")["b"].iloc[0] == 2.0
    filled_mean = fill_missing(df, "mean")
    assert abs(filled_mean["a"].iloc[1] - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_time_series_cv_splits_ordering() -> None:
    splits = list(time_series_cv_splits(20, n_splits=3))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min(), "Train must precede test in time"


def test_regression_metrics_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_build_models_keys() -> None:
    models = build_models()
    assert "linear" in models
    assert "lasso" in models
    assert "mlp" in models
    for name, model in models.items():
        assert hasattr(model, "fit"), f"{name} has no fit method"
        assert hasattr(model, "predict"), f"{name} has no predict method"


# ---------------------------------------------------------------------------
# Interpretation display names
# ---------------------------------------------------------------------------

def test_feature_display_name_known() -> None:
    assert feature_display_name("hicp_inflation") == "HICP inflation (%)"
    assert feature_display_name("unemployment_rate_lag1") == "Unemployment rate (previous year)"


def test_feature_display_name_unknown_fallback() -> None:
    result = feature_display_name("some_new_feature")
    assert result == "Some New Feature"


# ---------------------------------------------------------------------------
# Features (legacy module)
# ---------------------------------------------------------------------------

def test_add_interactions() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = add_interactions(df, columns=["a", "b"])
    assert "a_x_b" in result.columns
    assert list(result["a_x_b"]) == [4, 10, 18]
