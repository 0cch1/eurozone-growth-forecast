# Eurozone Growth Forecast

Modeling Eurozone economic growth with interpretable machine learning.

## Project structure

```
eurozone-growth-forecast/
├── data/
│   ├── raw/                 # Raw API extracts
│   └── processed/           # Cleaned and feature-ready datasets
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

**Model comparison (with visualization):** To compare baseline models (linear, MLP, XGBoost) and get a bar-chart of MAE/RMSE:

```
python -m src.build_dataset   # if not already done
python -m src.compare_models
```

The script prints a comparison table and saves `data/processed/model_comparison.png`. In code use `compare_models()` for the table and `plot_comparison(results, save_path=..., show=True)` to customize the plot.

The default fetch uses:
- Eurostat GDP growth dataset `tec00115` (filtered to `EA19`).
- ECB SDW exchange rate series `EXR/M.USD.EUR.SP00.A`.

You can customize sources in `src/data_loader.py`.

## Current progress

- Real data ingestion from Eurostat/ECB is wired up (`src/data_loader.py`).
- Raw data download script is available (`python -m src.fetch_real_data`).
- Yearly EA-level dataset builder is available (`python -m src.build_dataset`).
- Pipeline now supports running on processed real data.
- Model comparison with MAE/RMSE bar-chart visualization (`python -m src.compare_models`).
- Basic smoke tests and import setup are in place.
- Indicator registry is defined in `src/indicators.py` and documented in `INDICATORS.md`.

## Notes

- `data_loader.py` includes minimal Eurostat/ECB fetch helpers (extend as needed).
- `interpretation.py` contains placeholders for SHAP/PDP routines.
