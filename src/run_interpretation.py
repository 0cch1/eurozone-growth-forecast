"""Run XAI: SHAP summary, PDP, and local explanation for a trained model (PDD §4.5)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from .feature_engineering import build_minimal_features
from .interpretation import local_explanation, pdp_plot, permutation_importance, shap_summary
from .models import build_models
from .preprocessing import standardize_features


def _load_and_prepare():
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "data" / "processed" / "panel_yearly.csv"
    if not path.exists():
        raise FileNotFoundError("Run python -m src.build_dataset first.")
    df = pd.read_csv(path).sort_values("year").reset_index(drop=True)
    df = build_minimal_features(
        df,
        target_col="gdp_growth",
        lag_cols=["usd_eur_rate"],
        diff_cols=["usd_eur_rate"],
    )
    return df


def main(model_name: str = "linear", out_dir: str = "data/processed/figures") -> None:
    """Train the chosen model on full data and produce SHAP summary, PDP, and one local explanation."""
    df = _load_and_prepare()
    feature_cols = [c for c in df.columns if c != "gdp_growth"]
    X_df = cast(pd.DataFrame, df[feature_cols])
    y = df["gdp_growth"].to_numpy()
    X_scaled, _ = standardize_features(X_df, columns=feature_cols)
    X_scaled_df = pd.DataFrame(X_scaled.to_numpy(), columns=feature_cols)
    X_arr = X_scaled_df.to_numpy()

    models = build_models()
    if model_name not in models:
        print(f"Unknown model {model_name}. Choose from: {list(models.keys())}")
        return
    model = models[model_name]
    model.fit(X_arr, y)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name}")
    # SHAP summary
    summary = shap_summary(
        model,
        X_arr,
        feature_names=feature_cols,
        save_path=out_path / f"shap_summary_{model_name}.png",
        max_display=min(10, len(feature_cols)),
    )
    if summary.get("summary_plot_path"):
        print(f"  SHAP summary: {summary['summary_plot_path']}")
    elif "shap_values" not in summary:
        print("  SHAP summary skipped (install shap to enable).")
    if summary.get("error"):
        print(f"  SHAP error: {summary['error']}")

    # Permutation importance (PDD §4.5)
    perm = permutation_importance(
        model, X_arr, y,
        feature_names=feature_cols,
        save_path=out_path / f"permutation_importance_{model_name}.png",
        n_repeats=10,
        random_state=42,
    )
    if perm.get("perm_plot_path"):
        print(f"  Permutation importance: {perm['perm_plot_path']}")
    if perm.get("error"):
        print(f"  Permutation error: {perm['error']}")

    # PDP for first two features (use scaled data; pass feature names via DataFrame for column index)
    X_for_pdp = X_arr
    for feat_idx, feat in enumerate(feature_cols[:2]):
        pdp = pdp_plot(
            model,
            X_for_pdp,
            [str(feat_idx)],
            feature_labels=feature_cols,
            save_path=out_path / f"pdp_{model_name}_{feat}.png",
            grid_resolution=30,
        )
        if pdp.get("pdp_plot_path"):
            print(f"  PDP {feat}: {pdp['pdp_plot_path']}")
        if pdp.get("error"):
            print(f"  PDP error: {pdp['error']}")

    # Local explanation for first test instance (use scaled X so model input is consistent)
    X_for_local = X_arr
    local = local_explanation(
        model, X_for_local, instance_idx=0, feature_names=feature_cols,
        save_path=out_path / f"local_{model_name}_instance0.png",
    )
    if local.get("local_plot_path"):
        print(f"  Local explanation: {local['local_plot_path']}")
    if local.get("error"):
        print(f"  Local error: {local['error']}")
    print("Done.")


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "linear"
    main(model_name=name)
