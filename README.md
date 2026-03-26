# Eurozone Growth Forecast

Modelling Eurozone economic growth with interpretable machine learning.

**Delivery scope:** Minimum standard + one optional (AutoML). For environment, seeds, and run order, see [docs/Reproducibility.md](docs/Reproducibility.md).

## Project structure

```
eurozone-growth-forecast/
├── data/
│   ├── raw/                 # Raw API extracts
│   └── processed/           # Cleaned and feature-ready datasets
├── docs/                    # Documentation (e.g. data dictionary)
├── notebooks/               # EDA and prototype experiments
├── src/                     # Core pipeline modules
├── tests/                   # Minimal smoke tests
├── requirements.txt         # Python dependencies
└── .gitignore
```

## Quick start

1. Create/activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run the demo pipeline:

```
python -m src.run_pipeline
```

## Real data (Eurostat / ECB)

To fetch real data into `data/raw`, run:

```
python -m src.fetch_real_data
```

To build the yearly EA dataset into `data/processed`, run:

```
python -m src.build_dataset
```

**Model comparison (with visualisation):** To compare baseline models (linear, MLP, XGBoost) and get a bar-chart of MAE/RMSE:

```
python -m src.build_dataset   # if not already done
python -m src.compare_models
```

The script prints a comparison table and saves `data/processed/model_comparison.png`. In code use `compare_models()` for the table and `plot_comparison(results, save_path=..., show=True)` to customise the plot.

The default fetch uses:
- Eurostat GDP growth dataset `tec00115` (filtered to `EA19`).
- ECB SDW exchange rate series `EXR/M.USD.EUR.SP00.A`.

You can customise sources in `src/data_loader.py`.

**Eurostat backup:** If the API returns no data, `python -m src.build_dataset` will use the bundled backup `data/raw/eurostat_gdp_growth_fallback.csv` (EA19 annual GDP growth, Eurostat-style, 1999–2024). You can replace this file with a fresh download from [Eurostat bulk/databrowser](https://ec.europa.eu/eurostat/databrowser/view/tec00115/default/table) if needed.

## Main results and robustness outputs

To generate the **main time-series cross-validation results** used in the report:

```
python -m src.build_dataset      # if not already done
python -m src.main_results
```

This writes:

- `results/main_cv_results.csv`: model-level MAE/RMSE from time-aware cross-validation.
- `results/main_cv_results.png`: bar-chart visualisation of MAE/RMSE across models.

To generate the **AutoML robustness result** (single time-based holdout, last 20% as test):

```
pip install flaml
python -m src.automl_experiments
```

This writes:

- `results/robustness_automl_holdout.csv`: MAE/RMSE and basic metadata (train/test sizes, best estimator).

These outputs are intended to be referenced in the dissertation/report as the primary
time-series CV result and a complementary robustness check respectively.

## Interpretability (XAI, PDD §4.5)

After building the dataset and (optionally) comparing models, run SHAP summary, PDP, and local explanations for a chosen model:

```
python -m src.build_dataset
python -m src.run_interpretation linear    # or: mlp, xgboost
```

Outputs are saved under `data/processed/figures/` (SHAP summary plot, PDPs for first two features, one local explanation). Use the `interpretation` module in code: `shap_summary()`, `pdp_plot()`, `local_explanation()`.

## Optional: symbolic regression (Py-OPERON)

For an interpretable formula-like model (e.g. `gdp_growth = f(inflation, ...)`), install [pyoperon](https://pypi.org/project/pyoperon/) and it will be included in `build_models()` as `"symbolic"`:

```
pip install pyoperon
```

Then run the pipeline or model comparison as usual; the symbolic model will appear in the comparison table.

## Optional: AutoML baseline (FLAML)

To compare a time-based AutoML baseline (same data, last 20% holdout) with your manual models:

```
pip install flaml
python -m src.automl_experiments
```

Results (MAE/RMSE) are also written to `results/robustness_automl_holdout.csv` and can be
compared with the main CV table in `results/main_cv_results.csv`.

## Current progress

- **Data ingestion:** 6 macroeconomic indicators from Eurostat and ECB (`src/indicators.py`, `INDICATORS.md`).
- **Dataset build:** Yearly euro-area panel with lag and change features (`python -m src.build_dataset`).
- **Model comparison:** Ridge, MLP, XGBoost with time-series CV; MAE, RMSE, R² (`python -m src.main_results`).
- **Interpretability (XAI):** SHAP summary, PDP, permutation importance, and local explanations (`python -m src.run_interpretation <model>`).
- **Country-level visualisation:** Country GDP growth panel for 11 euro-area countries plus selected non-EU comparator countries (`UK`, `CH`, `NO`, `IS`) when available from Eurostat, with an interactive Plotly choropleth map and year slider:
  - `python -m src.build_country_panel` → `data/processed/panel_country_yearly.csv`
  - `python -m src.country_map_dashboard [YEAR]` → `results/country_gdp_growth_map.html`
  - This dashboard is an optional practical extension and does not affect the primary evaluation reported in the dissertation.
- **AutoML robustness:** FLAML AutoML baseline (`python -m src.automl_experiments`).
- **Documentation:** Data dictionary, reproducibility guide, indicator registry.
- **Tests:** Smoke tests in `tests/`.

## Future work

- Rolling/expanding window evaluation as an alternative to fixed-fold time-series CV.
- Additional model families (e.g. GAM, Elastic Net, polynomial regression).
- Sub-period robustness analysis (e.g. pre/post financial crisis).
- Country-level model predictions overlaid on the choropleth map.
- Symbolic regression (Py-OPERON) for formula-like interpretable models.

## Notes

- `data_loader.py` includes minimal Eurostat/ECB fetch helpers (extend as needed).
- `interpretation.py` implements SHAP (Tree/Linear/Kernel explainer), sklearn PDP, permutation importance, and local explanation bar charts.
- Full reproducibility instructions are in [docs/Reproducibility.md](docs/Reproducibility.md).
