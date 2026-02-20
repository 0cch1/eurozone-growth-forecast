# Reproducibility

This document describes how to reproduce the project’s data, model training, and evaluation (PDD §4.6). Follow the steps below in order.

---

## 1. Environment

- **Python:** 3.8 or higher (3.10+ recommended).
- **Dependencies:** Install from the project root:
  ```bash
  pip install -r requirements.txt
  ```
- **Optional (for extra models / baselines):**
  - `pip install pyoperon` — adds symbolic regression to `build_models()`.
  - `pip install flaml` — enables `python -m src.automl_experiments`.

---

## 2. Random seeds

- **Models:** All estimators use `random_state=42` (see `src/models.py` and `build_models()`).
- **Data:** Synthetic fallbacks use `np.random.default_rng(42)` (e.g. in `build_dataset.py`, `run_pipeline.py`).
- **Splits:** `TimeSeriesSplit` is deterministic for a given `n_samples` and `n_splits`; no extra seed is set.

Reproducing the same run: use the same Python version, same `requirements.txt`, and the same commands below. No manual seed setting is required beyond what is already in the code.

---

## 3. Recommended run order

Run these in sequence from the project root.

| Step | Command | Purpose |
|------|--------|--------|
| 1 | `python -m src.fetch_real_data` | Download raw data from Eurostat/ECB into `data/raw/`. (If the API fails, step 2 will use the bundled backup.) |
| 2 | `python -m src.build_dataset` | Build the yearly panel and save `data/processed/panel_yearly.csv`. |
| 3 | `python -m src.run_pipeline` | Run the demo pipeline (time-series CV, linear model, MAE/RMSE/R² per split). |
| 4 | `python -m src.compare_models` | Compare all available models and save the table + `data/processed/model_comparison.png`. |
| 5 (optional) | `python -m src.run_interpretation linear` | Generate SHAP/PDP/local explanation plots under `data/processed/figures/`. |
| 6 (optional) | `python -m src.automl_experiments` | Run the FLAML AutoML baseline (requires `pip install flaml`). Uses MAE, 60s budget, rf/extra_tree (+ xgboost if installed); see `automl_experiments.py` docstring. |

**Minimal reproduction (if data already built):**  
Steps 2 → 3 and 2 → 4 are enough to reproduce the main results. Step 1 is only needed when (re)downloading raw data.

---

## 4. Outputs to expect

- **Data:** `data/processed/panel_yearly.csv` (columns: `year`, `gdp_growth`, `usd_eur_rate`).
- **Model comparison:** Printed table with columns `model`, `mae`, `rmse`, `r2`; figure `data/processed/model_comparison.png` (MAE and RMSE bars).
- **Interpretation:** Under `data/processed/figures/` (SHAP summary if available, permutation importance, PDPs, local explanation image).

Variable definitions and raw file formats are in [data_dictionary.md](data_dictionary.md).
