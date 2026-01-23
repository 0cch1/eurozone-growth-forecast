"""Model interpretation utilities for SHAP and PDP."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def shap_summary_placeholder(model, X: pd.DataFrame, feature_names: Iterable[str]) -> dict:
    """Placeholder for SHAP summary computation.

    Returns:
        Dictionary describing SHAP setup (stub).
    """
    return {
        "model": type(model).__name__,
        "n_samples": len(X),
        "features": list(feature_names),
    }


def pdp_placeholder(feature: str) -> str:
    """Placeholder for Partial Dependence Plot generation."""
    return f"PDP for {feature}"
