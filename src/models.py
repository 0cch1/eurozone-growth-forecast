"""Model definitions for Linear, XGBoost, and MLP."""

from __future__ import annotations

from typing import Dict

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor  # type: ignore
except ImportError:  # pragma: no cover
    XGBRegressor = None


def build_models(random_state: int = 42) -> Dict[str, object]:
    """Instantiate model objects used in experiments.

    Args:
        random_state: Seed for reproducibility.

    Returns:
        Dictionary of model name to estimator.
    """
    models: Dict[str, object] = {
        "linear": LinearRegression(),
        "mlp": MLPRegressor(
            random_state=random_state,
            max_iter=2000,
            hidden_layer_sizes=(64, 32),
        ),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    return models
