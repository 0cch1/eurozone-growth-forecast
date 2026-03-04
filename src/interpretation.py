"""Model interpretation: SHAP summary, PDP, and local explanations (PDD §4.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, cast

import numpy as np
import pandas as pd

ArrayLike = pd.DataFrame | np.ndarray

# Optional SHAP and sklearn.inspection
try:
    import shap  # type: ignore
except ImportError:
    shap = None

try:
    from sklearn.inspection import (
        PartialDependenceDisplay,
        partial_dependence,
        permutation_importance as sklearn_permutation_importance,
    )
except ImportError:
    PartialDependenceDisplay = None  # type: ignore
    partial_dependence = None  # type: ignore
    sklearn_permutation_importance = None  # type: ignore


def _get_explainer(model: Any, X: np.ndarray, feature_names: List[str]) -> Any | None:
    """Return an appropriate SHAP explainer for model type."""
    if shap is None:
        return None
    model_type = type(model).__name__
    try:
        if "XGBRegressor" in model_type or "XGBClassifier" in model_type:
            return shap.TreeExplainer(model)
        if "LinearRegression" in model_type or "Ridge" in model_type or "Lasso" in model_type:
            return shap.LinearExplainer(model, X)
        # Fallback for MLP, SymbolicRegressor, etc.: use KernelExplainer with a small background
        n_bg = min(50, len(X))
        bg = X[np.random.default_rng(42).choice(len(X), n_bg, replace=False)]
        return shap.KernelExplainer(cast(Any, model).predict, bg)
    except Exception:
        try:
            # Generic fallback for newer SHAP versions.
            return shap.Explainer(model, X, feature_names=feature_names)
        except Exception:
            return None


def _compute_shap_values(explainer: object, X_arr: np.ndarray) -> np.ndarray:
    """Compute SHAP values across SHAP APIs (legacy and modern)."""
    # Newer API: explainer(X) -> Explanation(values=...)
    try:
        explanation = explainer(X_arr)  # type: ignore[misc]
        values = getattr(explanation, "values", None)
        if values is not None:
            arr = np.asarray(values)
            if arr.ndim == 3:
                arr = arr[..., 0]
            return arr
    except Exception:
        pass

    # Legacy API: explainer.shap_values(X)
    shap_values = explainer.shap_values(X_arr)  # type: ignore[attr-defined]
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.asarray(shap_values)


def shap_summary(
    model: Any,
    X: ArrayLike,
    feature_names: Optional[Iterable[str]] = None,
    save_path: Optional[Path] = None,
    max_display: int = 10,
) -> dict[str, Any]:
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
    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = list(cast(pd.DataFrame, X).columns)
    else:
        feature_names = list(feature_names) if feature_names else [f"x{i}" for i in range(X.shape[1])]
    X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X

    out: dict[str, Any] = {
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
        shap_values = _compute_shap_values(explainer, X_arr)
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


def _feature_display_name(name: str) -> str:
    """Human-readable axis label for known features."""
    labels = {
        "year": "Year",
        "usd_eur_rate": "USD/EUR exchange rate",
        "usd_eur_rate_lag1": "USD/EUR rate (lag 1 year)",
        "usd_eur_rate_chg1": "USD/EUR rate change (1 year)",
        "hicp_inflation": "HICP inflation (%)",
        "unemployment_rate": "Unemployment rate (%)",
        "short_term_rate": "Short-term interest rate (%)",
        "gov_debt_gdp": "Government debt (% of GDP)",
    }
    return labels.get(name, name.replace("_", " ").title())


def pdp_plot(
    model: Any,
    X: ArrayLike,
    features: List[str],
    feature_labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    grid_resolution: int = 50,
) -> dict[str, Any]:
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
    out: dict[str, Any] = {"features": features}
    if partial_dependence is None or PartialDependenceDisplay is None:
        return out

    X_input = X
    if isinstance(X, pd.DataFrame):
        col_idx = [list(cast(pd.DataFrame, X).columns).index(f) for f in features]
    else:
        X_input = np.asarray(X)
        col_idx = [int(f) for f in features]  # indices when X is array

    try:
        pd_avg, grid_vals = partial_dependence(
            model,
            X_input,
            features=col_idx,
            grid_resolution=grid_resolution,
        )
        out["grid_values"] = [np.asarray(g).tolist() for g in grid_vals]
        out["average"] = np.asarray(pd_avg).tolist()

        if save_path is not None and len(features) <= 2:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter
            fig, ax = plt.subplots(figsize=(6, 4))
            # Pass DataFrame and feature names so axes show readable labels.
            PartialDependenceDisplay.from_estimator(
                model,
                X_input,
                features=col_idx,
                grid_resolution=grid_resolution,
                ax=ax,
            )
            if len(features) == 1:
                feat_key: str
                if feature_labels is not None and len(feature_labels) > col_idx[0]:
                    feat_key = str(feature_labels[col_idx[0]])
                else:
                    feat_key = str(features[0])
                feat_label = _feature_display_name(feat_key)
                ax.set_xlabel(feat_label)
                ax.set_ylabel("Predicted GDP growth (%)")
                ax.set_title(f"Partial dependence: effect of {feat_label} on prediction")
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            out["pdp_plot_path"] = str(save_path)
    except Exception as e:
        out["error"] = str(e)
    return out


def _save_local_explanation_barchart(
    shap_values: np.ndarray,
    feature_names: List[str],
    base_value: float,
    prediction: float,
    save_path: Path,
) -> None:
    """Save a bar chart of SHAP contributions for one instance (fallback when waterfall fails)."""
    import matplotlib.pyplot as plt

    sv = np.asarray(shap_values).flatten()
    n = len(feature_names)
    labels = [_feature_display_name(f) for f in feature_names]
    colors = ["steelblue" if v >= 0 else "coral" for v in sv]
    y_pos = range(n)
    fig, ax = plt.subplots(figsize=(6, max(3, n * 0.4)))
    ax.barh(y_pos, sv, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("SHAP contribution to prediction")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="-")
    ax.set_title(
        f"Local explanation (instance)\nBaseline = {base_value:.2f}%  →  Prediction = {prediction:.2f}%"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def local_explanation(
    model: Any,
    X: ArrayLike,
    instance_idx: int = 0,
    feature_names: Optional[Iterable[str]] = None,
    save_path: Optional[Path] = None,
) -> dict[str, Any]:
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
    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
    else:
        feature_names = list(feature_names) if feature_names else [f"x{i}" for i in range(X.shape[1])]
    X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
    if isinstance(X, pd.DataFrame):
        x_for_predict = cast(pd.DataFrame, X).iloc[instance_idx : instance_idx + 1]
        x = np.asarray(x_for_predict)
    else:
        x = X_arr[instance_idx : instance_idx + 1]
        x_for_predict = x

    out: dict[str, Any] = {
        "instance_idx": instance_idx,
        "features": feature_names,
    }

    # Fallback for linear models without SHAP: use coefficients × (x - mean) as contributions.
    model_any = cast(Any, model)
    if hasattr(model_any, "coef_") and hasattr(model_any, "intercept_") and save_path is not None:
        try:
            coef = np.asarray(model_any.coef_).flatten()
            if len(coef) == X_arr.shape[1]:
                x_flat = X_arr[instance_idx]
                base_value = float(model_any.intercept_)
                contribs = coef * (x_flat - np.mean(X_arr, axis=0))
                pred = float(base_value + contribs.sum())
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                _save_local_explanation_barchart(
                    shap_values=contribs,
                    feature_names=list(feature_names),
                    base_value=base_value,
                    prediction=pred,
                    save_path=save_path,
                )
                out["base_value"] = base_value
                out["prediction"] = pred
                out["local_plot_path"] = str(save_path)
                return out
        except Exception:
            pass

    if shap is None:
        return out

    explainer = _get_explainer(model_any, X_arr, feature_names)
    if explainer is None:
        return out

    try:
        sv = _compute_shap_values(explainer, x).flatten()
        pred = float(model_any.predict(x_for_predict)[0])
        expected_value = getattr(explainer, "expected_value", None)
        if expected_value is None:
            base = float(pred - sv.sum())
        else:
            expected_arr = np.asarray(expected_value).flatten()
            base = float(expected_arr[0]) if expected_arr.size else float(pred - sv.sum())
        out["base_value"] = base
        out["shap_values"] = sv.tolist()
        out["prediction"] = pred

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            saved = False
            # Try SHAP waterfall first (PNG).
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
                saved = True
            except Exception:
                pass
            # Fallback: always produce a PNG bar chart of SHAP contributions for this instance.
            if not saved:
                _save_local_explanation_barchart(
                    shap_values=sv,
                    feature_names=list(feature_names),
                    base_value=base,
                    prediction=pred,
                    save_path=save_path,
                )
                out["local_plot_path"] = str(save_path)
    except Exception as e:
        out["error"] = str(e)
    return out


def permutation_importance(
    model: Any,
    X: ArrayLike,
    y: np.ndarray,
    feature_names: Optional[Iterable[str]] = None,
    save_path: Optional[Path] = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """Permutation importance (global feature importance, PDD §4.5).

    Measures how much score degrades when a feature is randomly permuted.
    Model-agnostic; works with any fitted regressor.

    Args:
        model: Fitted regressor with .predict().
        X: Feature matrix.
        y: Target vector.
        feature_names: Optional names for features; used for plot and result.
        save_path: If set, save bar chart of importances to this path.
        n_repeats: Number of permutation repeats (default 10).
        random_state: Seed for permutation shuffle.

    Returns:
        Dict with importances_mean, importances_std, feature_names, optional path.
    """
    out: dict[str, Any] = {"feature_names": None, "importances_mean": None, "importances_std": None}
    if sklearn_permutation_importance is None:
        return out

    X_arr = np.asarray(X)
    X_input = X if isinstance(X, pd.DataFrame) else X_arr
    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = list(cast(pd.DataFrame, X).columns)
    else:
        feature_names = list(feature_names) if feature_names else [f"x{i}" for i in range(X_arr.shape[1])]
    out["feature_names"] = feature_names

    try:
        perm = sklearn_permutation_importance(
            model,
            X_input,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="neg_mean_absolute_error",
        )
        perm_any = cast(Any, perm)
        out["importances_mean"] = perm_any.importances_mean.tolist()
        out["importances_std"] = perm_any.importances_std.tolist()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            import matplotlib.pyplot as plt
            # sklearn returns change in neg_MAE (negative when permuting hurts); show MAE increase.
            mean_inc = -np.asarray(perm_any.importances_mean)
            std_inc = np.asarray(perm_any.importances_std)
            fig, ax = plt.subplots(figsize=(6, max(3, len(feature_names) * 0.4)))
            idx = np.argsort(mean_inc)[::-1]  # most important first (top)
            y_pos = range(len(feature_names))
            ax.barh(
                y_pos,
                mean_inc[idx],
                xerr=std_inc[idx],
                color="steelblue",
                capsize=3,
                edgecolor="black",
                linewidth=0.8,
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels([_feature_display_name(feature_names[i]) for i in idx])
            ax.set_xlabel("Increase in MAE when feature is permuted")
            ax.set_title("Feature importance (permutation)\nHigher bar = feature matters more for predictions")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="-")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            out["perm_plot_path"] = str(save_path)
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
