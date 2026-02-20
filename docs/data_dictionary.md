# Data dictionary

This document describes all variables used in the Eurozone growth forecast project: the processed modelling dataset, derived features, and the raw data file formats. It supports reproducibility and aligns with PDD §3.1 / §3.4.

---

## 1. Processed dataset: `data/processed/panel_yearly.csv`

This is the main dataset used for training and evaluation. One row per year; geography is Euro area (EA19 or EA20, depending on data source).

| Column         | Type   | Unit / format | Description | Source |
|----------------|--------|----------------|-------------|--------|
| `year`         | integer | Calendar year (e.g. 2000) | Reference year of the observation. | — |
| `gdp_growth`   | float   | Percent (%)    | Real GDP growth rate: annual percentage change in real GDP (volume, constant prices). Target variable for forecasting. | Eurostat `tec00115` (or backup), filtered to EA19/EA20, annual, unit PC_GDP, na_item B1GQ. |
| `usd_eur_rate` | float   | Level (USD per 1 EUR) | Exchange rate: number of US dollars per one euro. In the build step, monthly ECB values are averaged to one value per year. | ECB SDW series `EXR/M.USD.EUR.SP00.A` (monthly), then yearly mean in `build_dataset.py`. |

- **Time coverage:** Depends on data availability; typically from the late 1990s / 2000 to the latest full year (e.g. 2024). Exact range is in the file.
- **Missing values:** Rows with missing `year` or target/source values are dropped during build. The merged panel is an inner join on `year`, so the final coverage is the intersection of years across all indicators.

---

## 2. Derived variables (after feature engineering)

The pipeline applies `build_minimal_features()` to `panel_yearly.csv` before modelling (see `src/feature_engineering.py`). The following columns are added; all are used as **features** except `gdp_growth`, which remains the **target**.

| Column               | Type  | Definition | Use |
|----------------------|-------|------------|-----|
| `usd_eur_rate_lag1`  | float | `usd_eur_rate` shifted by 1 year (previous year’s rate). | Feature. |
| `usd_eur_rate_chg1`  | float | First difference of `usd_eur_rate` over 1 year: `usd_eur_rate(t) - usd_eur_rate(t-1)`. | Feature. |

- **Dropped rows:** After adding lags and differences, the first row(s) with NaN are dropped (`.dropna()`), so the modelling sample starts from the second year onward.

---

## 3. Raw data file formats

### 3.1 Eurostat GDP growth: `data/raw/eurostat_gdp_growth.csv` (or backup)

- **Source:** Eurostat dataset [tec00115](https://ec.europa.eu/eurostat/web/products-datasets/-/tec00115) (Real GDP growth rate – volume), or the bundled backup `eurostat_gdp_growth_fallback.csv` when the API returns no data.
- **Filters used in build:** `freq = A` (annual), `unit = PC_GDP` (percentage change of GDP), `na_item = B1GQ` (GDP), `geo = EA19` or `EA20` (Euro area).
- **Columns in the raw CSV:**

| Column   | Description |
|----------|-------------|
| `freq`   | Frequency (e.g. `A` = annual). |
| `unit`   | Unit of measure (e.g. `PC_GDP` = percentage change of GDP). |
| `na_item`| National accounts item (e.g. `B1GQ` = GDP). |
| `geo`    | Geography (e.g. `EA19`, `EA20`). |
| `time`   | Time period (year, e.g. `2000`). |
| `value`  | Real GDP growth rate (%). |

### 3.2 ECB exchange rate: `data/raw/ecb_exr_usd_eur.csv`

- **Source:** ECB Statistical Data Warehouse, series key `EXR/M.USD.EUR.SP00.A` (ECB reference exchange rate, US dollar per euro, monthly).
- **Columns used in the build:** The ECB CSV export may contain many columns (e.g. `KEY`, `FREQ`, `CURRENCY`, `OBS_STATUS`, …). Only the following are read and used:

| Column        | Description |
|---------------|-------------|
| `TIME_PERIOD` | Reference period (e.g. `2000-01` for January 2000). |
| `OBS_VALUE`  | Exchange rate: USD per 1 EUR. |

- **Aggregation:** In `build_dataset.py`, monthly observations are converted to **yearly** by taking the **mean** of `OBS_VALUE` within each calendar year. The result is written to `panel_yearly.csv` as `usd_eur_rate`.

---

## 4. Other processed outputs

- **`data/processed/indicator_list.csv`:** List of indicators used in the build (name, source, file_stem, value_name, description). For reference only; not used as model input.
- **Modelling inputs:** The model always sees the **feature-engineered** table (with `year`, `gdp_growth`, `usd_eur_rate`, `usd_eur_rate_lag1`, `usd_eur_rate_chg1`). The target is `gdp_growth`; all other columns except `gdp_growth` are features (after standardisation in the pipeline).

---

## 5. Version and reproducibility

- **Indicator definitions:** `src/indicators.py` and `INDICATORS.md`.
- **Build script:** `python -m src.build_dataset` (after `python -m src.fetch_real_data` if using API).
- **Data dictionary version:** This file; update when adding new indicators or columns to `panel_yearly.csv`.
