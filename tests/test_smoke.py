"""Minimal smoke tests for core utilities."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from evaluation import regression_metrics
from features import add_lag_features
from models import build_models
from preprocessing import fill_missing


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
