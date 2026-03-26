# Reproducibility

This document describes how to reproduce the project's data, model training, evaluation, and interpretation outputs (PDD §4.6). Follow the steps below in order.

---

## 1. Environment

- **Python:** 3.10 or higher recommended (tested on 3.13).
- **Dependencies:** Install from the project root:
  ```bash
  pip install -r requirements.txt
  ```
- **Optional (for extra models / baselines):**
  - `pip install pyoperon` — adds symbolic regression to `build_models()`.
  - `pip install "flaml[automl]"` — enables `python -m src.automl_experiments`.

---

## 2. Random seeds

- **Models:** All estimators use `random_state=42` (see `src/models.py`).
- **Data:** Synthetic fallbacks use `np.random.default_rng(42)`.
- **Splits:** `TimeSeriesSplit` is deterministic for a given `n_samples` and `n_splits`.
- **Permutation importance:** Uses `random_state=42`.

No manual seed setting is required beyond what is already in the code.

---

## 3. Recommended run order

Run these in sequence from the project root.

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `python -m src.fetch_real_data` | Download raw data from Eurostat/ECB into `data/raw/`. |
| 2 | `python -m src.build_dataset` | Build the yearly EA panel → `data/processed/panel_yearly.csv`. |
| 3 | `python -m src.main_results` | Run time-series CV across all models → `results/main_cv_results.csv` and `.png`. |
| 4 | `python -m src.run_interpretation linear` | Generate SHAP, PDP, permutation importance, local explanation for the linear (Ridge) model → `data/processed/figures/`. |
| 5 | `python -m src.run_interpretation xgboost` | Same interpretation outputs for XGBoost. |
| 6 | `python -m src.build_country_panel` | Build country-level GDP panel → `data/processed/panel_country_yearly.csv`. |
| 7 | `python -m src.country_map_dashboard` | Generate interactive choropleth map → `results/country_gdp_growth_map.html`. |
| 8 (optional) | `python -m src.automl_experiments` | Run FLAML AutoML holdout baseline → `results/robustness_automl_holdout.csv`. |

**Minimal reproduction (if raw data already fetched):**
Steps 2 → 3 → 4 → 5 are sufficient to reproduce the main results and interpretation figures.

---

## 4. Outputs to expect

### Main results (`results/`)

| File | Content |
|------|---------|
| `main_cv_results.csv` | Model-level MAE, RMSE, R² (mean ± std) from time-series CV (includes naive mean baseline). |
| `main_cv_per_fold.csv` | Per-fold metrics for every model and CV split (matches report Table A.1). |
| `main_cv_results.png` | Bar chart of MAE/RMSE across models. |
| `robustness_automl_holdout.csv` | AutoML holdout MAE/RMSE (optional). |
| `country_gdp_growth_map.html` | Interactive choropleth of country-level GDP growth. |

### Interpretation figures (`data/processed/figures/`)

| File pattern | Content |
|-------------|---------|
| `shap_summary_{model}.png` | SHAP summary plot (global feature effects). |
| `permutation_importance_{model}.png` | Permutation importance (feature ranking). |
| `pdp_{model}_{feature}.png` | Partial dependence plots (top 2 features). |
| `local_{model}_instance0.png` | Local explanation for the first observation. |

### Data (`data/processed/`)

| File | Content |
|------|---------|
| `panel_yearly.csv` | Main yearly EA panel (6 indicators). |
| `panel_country_yearly.csv` | Country-level GDP panel (11 EA countries). |
| `indicator_list.csv` | Indicator metadata. |

---

## 5. Tests

Run the test suite with:

```bash
pytest tests/ -v
```

---

## 6. Notes

- Variable definitions and raw file formats are in [data_dictionary.md](data_dictionary.md).
- Indicator sources and codes are in `INDICATORS.md` and `src/indicators.py`.
- Feature engineering (lags, first differences) is in `src/feature_engineering.py`.
- The `year` column is retained in the dataset but **excluded from model features** in both `compare_models.py` and `run_interpretation.py`.
