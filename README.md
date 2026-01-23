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

## Notes

- `data_loader.py` is a stub that will be extended for Eurostat/ECB APIs.
- `interpretation.py` contains placeholders for SHAP/PDP routines.
