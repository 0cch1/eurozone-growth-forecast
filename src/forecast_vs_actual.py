"""Generate actual vs predicted GDP growth chart and 2025 forecast.

Produces:
- A line chart comparing actual EA GDP growth with XGBoost backtest predictions
- A one-step-ahead 2025 forecast using the latest available features
- Saves outputs to results/ and data/processed/figures/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data_utils import get_project_root, load_and_prepare
from .models import build_models
from .preprocessing import standardize_features


def backtest_predictions() -> pd.DataFrame:
    """Run expanding-window backtest: train on years up to t, predict t+1.

    Returns DataFrame with columns: year, actual, predicted, model.
    """
    df = load_and_prepare()
    feature_cols = [c for c in df.columns if c not in ("gdp_growth", "year")]
    years = df["year"].values
    X_df = df[feature_cols]
    y = df["gdp_growth"].values

    models = build_models()
    xgb = models.get("xgboost")
    if xgb is None:
        raise ImportError("XGBoost is required for forecast_vs_actual.")

    records = []
    # Need at least 8 training observations before first prediction
    min_train = 8
    for i in range(min_train, len(df)):
        train_idx = list(range(i))
        test_idx = [i]

        X_train, scaler = standardize_features(X_df.iloc[train_idx], columns=feature_cols)
        X_test, _ = standardize_features(X_df.iloc[test_idx], columns=feature_cols, scaler=scaler)

        from sklearn.base import clone
        model = clone(xgb)
        model.fit(X_train.to_numpy(), y[train_idx])
        pred = model.predict(X_test.to_numpy())[0]

        records.append({
            "year": int(years[i]),
            "actual": float(y[i]),
            "predicted": float(pred),
        })

    return pd.DataFrame(records)


def forecast_next_year() -> dict:
    """Train XGBoost on all available data and forecast next-year EA GDP growth.

    load_and_prepare() returns years 2001-2023 (target-shifted, NaN-dropped).
    To forecast 2025, we need 2024 features. We rebuild features WITHOUT
    target shifting to access the 2024 row, then train on the shifted subset.

    Returns dict with forecast value and metadata.
    """
    from .feature_engineering import build_full_features
    from .data_utils import load_processed_panel, MODEL_EXCLUDED_COLS

    # --- 1. Build features WITHOUT target shift to keep 2024 row ---
    raw = load_processed_panel().sort_values("year").reset_index(drop=True)
    raw = raw.drop(columns=[c for c in MODEL_EXCLUDED_COLS if c in raw.columns])
    raw = raw.ffill()
    df_all = build_full_features(raw, target_col="gdp_growth", forecast_horizon=0)
    # df_all: 2001-2024 (24 rows), target is contemporaneous (not shifted)

    feature_cols = [c for c in df_all.columns if c not in ("gdp_growth", "year")]

    # --- 2. Training set: use shifted target (features at t, target at t+1) ---
    # This matches the main pipeline's load_and_prepare() output
    df_train = load_and_prepare()  # 2001-2023, target already shifted
    X_train_df = df_train[feature_cols]
    y_train = df_train["gdp_growth"].values

    models = build_models()
    xgb = models["xgboost"]

    X_train, scaler = standardize_features(X_train_df, columns=feature_cols)
    xgb.fit(X_train.to_numpy(), y_train)

    # --- 3. Forecast: use 2024 features to predict 2025 GDP growth ---
    last_row = df_all[df_all["year"] == df_all["year"].max()]
    X_forecast, _ = standardize_features(last_row[feature_cols], columns=feature_cols, scaler=scaler)
    forecast_val = float(xgb.predict(X_forecast.to_numpy())[0])

    forecast_year = int(df_all["year"].max()) + 1  # 2024 + 1 = 2025

    return {
        "forecast_year": forecast_year,
        "predicted_gdp_growth": round(forecast_val, 2),
        "training_years": f"{int(df_train['year'].min())}\u2013{int(df_train['year'].max())}",
        "n_train": len(df_train),
    }


def plot_actual_vs_predicted(
    backtest_df: pd.DataFrame,
    forecast_next_year_val: Optional[float] = None,
    forecast_year: int = 2025,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plot actual vs predicted GDP growth with optional forecast point."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(backtest_df["year"], backtest_df["actual"],
            "o-", color="steelblue", linewidth=2, markersize=5, label="Actual GDP Growth")
    ax.plot(backtest_df["year"], backtest_df["predicted"],
            "s--", color="coral", linewidth=2, markersize=5, label="XGBoost Predicted")

    # Shade the error
    ax.fill_between(backtest_df["year"],
                     backtest_df["actual"], backtest_df["predicted"],
                     alpha=0.15, color="gray")

    if forecast_next_year_val is not None:
        ax.plot(forecast_year, forecast_next_year_val, "D", color="red", markersize=10,
                zorder=5, label=f"{forecast_year} Forecast ({forecast_next_year_val:.1f}%)")
        ax.axvline(x=forecast_year - 0.5, color="gray", linestyle=":", alpha=0.5)
        ax.annotate("Forecast", xy=(forecast_year, forecast_next_year_val),
                     xytext=(forecast_year, forecast_next_year_val + 1.5),
                     fontsize=9, ha="center", color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.2))

    # Fix x-axis: integer years, every 2 years
    year_min = int(backtest_df["year"].min())
    year_max = forecast_year if forecast_next_year_val is not None else int(backtest_df["year"].max())
    ticks = list(range(year_min, year_max + 1, 2))
    if year_max not in ticks:
        ticks.append(year_max)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(y) for y in ticks], rotation=45, ha="right")
    ax.set_xlim(year_min - 0.5, year_max + 0.5)

    ax.set_xlabel("Year")
    ax.set_ylabel("GDP Growth (%)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved actual vs predicted plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    """CLI entry point."""
    project_root = get_project_root()

    print("Running expanding-window backtest...")
    bt = backtest_predictions()
    print(bt.to_string(index=False))

    print("\nForecasting next year...")
    fc = forecast_next_year()
    fy = fc["forecast_year"]
    print(f"  {fy} EA GDP Growth Forecast: {fc['predicted_gdp_growth']}%")
    print(f"  Training: {fc['training_years']} ({fc['n_train']} observations)")

    # Save backtest CSV
    bt_path = project_root / "results" / "backtest_actual_vs_predicted.csv"
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(bt_path, index=False)
    print(f"\nSaved backtest results to {bt_path}")

    # Save plot
    fig_path = project_root / "data" / "processed" / "figures" / "actual_vs_predicted.png"
    try:
        plot_actual_vs_predicted(bt, forecast_next_year_val=fc["predicted_gdp_growth"],
                                 forecast_year=fy, save_path=fig_path)
    except Exception as e:
        print(f"Plot skipped ({e}). Install matplotlib to enable.")


if __name__ == "__main__":
    main()
