"""Model interpretation: SHAP summary, PDP, and local explanations (PDD §4.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

# Optional SHAP and sklearn.inspection
try:
    import shap  # type: ignore
except ImportError:
    shap = None

try:
    from sklearn.inspection import PartialDependenceDisplay, partial_dependence
except ImportError:
    PartialDependenceDisplay = None  # type: ignore
    partial_dependence = None  # type: ignore


def _get_explainer(model: object, X: np.ndarray, feature_names: List[str]):
    """Return appropriate SHAP explainer for model type."""
    if shap is None:
        return None
    model_type = type(model).__name__
    try:
        if "XGBRegressor" in model_type or "XGBClassifier" in model_type:
            return shap.TreeExplainer(model, X, feature_names=feature_names)
        if "LinearRegression" in model_type or "Ridge" in model_type or "Lasso" in model_type:
            return shap.LinearExplainer(model, X, feature_names=feature_names)
        # Fallback for MLP, SymbolicRegressor, etc.: use KernelExplainer with a small background
        n_bg = min(50, len(X))
        bg = X[np.random.default_rng(42).choice(len(X), n_bg, replace=False)]
        return shap.KernelExplainer(
            model.predict,
            bg,
            feature_names=feature_names,
        )
    except Exception:
        return None


def shap_summary(
    model: object,
    X: pd.DataFrame,
    feature_names: Optional[Iterable[str]] = None,
    save_path: Optional[Path] = None,
    max_display: int = 10,
) -> dict:
    """Compute SHAP values and plot summary (global explanation).

    Args:
        model: Fitted regressor with .predict().
        X: Feature matrix (DataFrame or array).
        feature_names: Optional list of names; defaults to X.columns.
        save_path: If set, save the summary plot to this path.
        max_display: Max number of features to show in summary plot.

    Returns:
        Dict with keys: model, n_samples, features, shap_values (array), explainer_type.
    """
    if feature_names is None and hasattr(X, "columns"):
        feature_names = list(X.columns)
    else:
        feature_names = list(feature_names) if feature_names else [f"x{i}" for i in range(X.shape[1])]
    X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X

    out = {
        "model": type(model).__name__,
        "n_samples": len(X_arr),
        "features": feature_names,
    }

    if shap is None:
        return out

    explainer = _get_explainer(model, X_arr, feature_names)
    if explainer is None:
        return out

    try:
        shap_values = explainer.shap_values(X_arr)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        out["shap_values"] = np.asarray(shap_values)
        out["explainer_type"] = type(explainer).__name__

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            import matplotlib.pyplot as mplt
            shap.summary_plot(
                shap_values,
                X_arr,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
            )
            mplt.savefig(save_path, dpi=150, bbox_inches="tight")
            mplt.close()
            out["summary_plot_path"] = str(save_path)
    except Exception as e:
        out["error"] = str(e)
    return out


def pdp_plot(
    model: object,
    X: pd.DataFrame,
    features: List[str],
    save_path: Optional[Path] = None,
    grid_resolution: int = 50,
) -> dict:
    """Partial dependence plot(s) for one or more features (global interpretation).

    Args:
        model: Fitted regressor with .predict().
        X: Feature DataFrame (used for grid and column names).
        features: List of feature names (1 or 2 for 1D/2D PDP).
        save_path: If set, save figure to this path.
        grid_resolution: Number of grid points per feature.

    Returns:
        Dict with pdp values and optional path.
    """
    out = {"features": features}
    if partial_dependence is None or PartialDependenceDisplay is None:
        return out

    X_arr = np.asarray(X)
    if hasattr(X, "columns"):
        col_idx = [list(X.columns).index(f) for f in features]
    else:
        col_idx = [int(f) for f in features]  # indices when X is array

    try:
        pd_avg, grid_vals = partial_dependence(
            model,
            X_arr,
            features=col_idx,
            grid_resolution=grid_resolution,
        )
        out["grid_values"] = [np.asarray(g).tolist() for g in grid_vals]
        out["average"] = np.asarray(pd_avg).tolist()

        if save_path is not None and len(features) <= 2:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            PartialDependenceDisplay.from_estimator(
                model,
                X_arr,
                features=col_idx,
                grid_resolution=grid_resolution,
                ax=ax,
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            out["pdp_plot_path"] = str(save_path)
    except Exception as e:
        out["error"] = str(e)
    return out


def local_explanation(
    model: object,
    X: pd.DataFrame,
    instance_idx: int = 0,
    feature_names: Optional[Iterable[str]] = None,
    save_path: Optional[Path] = None,
) -> dict:
    """Local explanation for a single instance (SHAP waterfall or force-style).

    Args:
        model: Fitted regressor.
        X: Feature matrix.
        instance_idx: Row index of the instance to explain.
        feature_names: Optional feature names.
        save_path: If set, save HTML or image to this path.

    Returns:
        Dict with base_value, shap_values for instance, predicted value.
    """
    if feature_names is None and hasattr(X, "columns"):
        feature_names = list(X.columns)
    else:
        feature_names = list(feature_names) if feature_names else [f"x{i}" for i in range(X.shape[1])]
    X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
    x = X_arr[instance_idx : instance_idx + 1]

    out = {
        "instance_idx": instance_idx,
        "features": feature_names,
    }

    if shap is None:
        return out

    explainer = _get_explainer(model, X_arr, feature_names)
    if explainer is None:
        return out

    try:
        shap_values = explainer.shap_values(x)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        sv = np.asarray(shap_values).flatten()
        pred = float(model.predict(x)[0])
        base = float(explainer.expected_value) if hasattr(explainer, "expected_value") else pred - sv.sum()
        out["base_value"] = base
        out["shap_values"] = sv.tolist()
        out["prediction"] = pred

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # Waterfall needs a single row of SHAP values
            try:
                import matplotlib.pyplot as plt
                shap.waterfall_plot(
                    shap.Explanation(
                        values=sv,
                        base_values=base,
                        data=x.flatten(),
                        feature_names=feature_names,
                    ),
                    show=False,
                )
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                out["local_plot_path"] = str(save_path)
            except Exception:
                # Fallback: save force plot as HTML if available
                if hasattr(shap, "force_plot"):
                    html = shap.force_plot(
                        base, sv, x,
                        feature_names=feature_names,
                        matplotlib=False,
                    )
                    if hasattr(html, "save_html"):
                        html_path = save_path.with_suffix(".html")
                        html.save_html(str(html_path))
                        out["local_plot_path"] = str(html_path)
    except Exception as e:
        out["error"] = str(e)
    return out


# Backward compatibility: keep placeholders as thin wrappers
def shap_summary_placeholder(model: object, X: pd.DataFrame, feature_names: Iterable[str]) -> dict:
    """Legacy name: calls shap_summary (no plot)."""
    return shap_summary(model, X, feature_names=feature_names)


def pdp_placeholder(feature: str) -> str:
    """Legacy placeholder; use pdp_plot() for real PDPs."""
    return f"PDP for {feature}"
