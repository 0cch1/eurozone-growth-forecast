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

Results (MAE/RMSE) can be compared with `python -m src.compare_models`.

## Current progress

- Real data ingestion from Eurostat/ECB is wired up (`src/data_loader.py`).
- Raw data download script and yearly EA-level dataset builder (`python -m src.fetch_real_data`, `python -m src.build_dataset`).
- Pipeline runs on processed real data (`python -m src.run_pipeline`).
- Model comparison with MAE/RMSE bar-chart visualisation (`python -m src.compare_models`).
- XAI: SHAP summary, PDP, and local explanations in `src/interpretation.py`; runner `python -m src.run_interpretation <model>`.
- Optional symbolic regression (Py-OPERON) in `models.py` when installed.
- Optional AutoML baseline: `python -m src.automl_experiments` (requires `pip install flaml`).
- Basic smoke tests and import setup are in place.
- Indicator registry is defined in `src/indicators.py` and documented in `INDICATORS.md`.

## Future work

- **Country-level panel and map**: Extend data to per-country (e.g. DE, FR, IT, ES, …) and add a lightweight map visualisation (e.g. Plotly/Folium choropleth) to show GDP growth or model outputs by country. See PDD for panel scope.

## Notes

- `data_loader.py` includes minimal Eurostat/ECB fetch helpers (extend as needed).
- `interpretation.py` implements SHAP (Tree/Linear/Kernel explainer), sklearn PDP, permutation importance, and local waterfall/force-style plots.
