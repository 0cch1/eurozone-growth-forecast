"""Model definitions for Linear/Ridge, XGBoost, MLP, and optional symbolic regression."""

from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor  # type: ignore
except ImportError:  # pragma: no cover
    XGBRegressor = None

try:
    from pyoperon.sklearn import SymbolicRegressor  # type: ignore
except ImportError:  # pragma: no cover
    SymbolicRegressor = None


def build_models(random_state: int = 42) -> Dict[str, Any]:
    """Instantiate model objects used in experiments.

    Uses Ridge instead of plain OLS to handle the small-sample / many-feature
    regime (n≈25, p≈14).  Alpha=1.0 is a mild default; in the report this
    choice is motivated by the regularisation discussion in §4.3.

    Args:
        random_state: Seed for reproducibility.

    Returns:
        Dictionary of model name to estimator.
    """
    models: Dict[str, Any] = {
        "linear": Ridge(alpha=1.0),
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
    if SymbolicRegressor is not None:
        models["symbolic"] = SymbolicRegressor(
            random_state=random_state,
            max_length=20,
            max_depth=6,
            time_limit=60,
            n_threads=1,
        )
    return models
