# Data dictionary

This document describes all variables used in the Eurozone growth forecast project: the processed modelling dataset, derived features, and the raw data file formats. It supports reproducibility and aligns with PDD §3.1 / §3.4.

---

## 1. Processed dataset: `data/processed/panel_yearly.csv`

This is the main dataset used for training and evaluation. One row per year; geography is Euro area (EA19 or EA20, depending on data source).

| Column | Type | Unit / format | Description | Source |
|--------|------|---------------|-------------|--------|
| `year` | integer | Calendar year (e.g. 2000) | Reference year of the observation. Used as a time index; **excluded from model features**. | — |
| `gdp_growth` | float | Percent (%) | Real GDP growth rate: annual percentage change in real GDP (volume, constant prices). **Target variable** for forecasting. | Eurostat `tec00115`, filtered to EA19/EA20, annual, unit PC_GDP, na_item B1GQ. |
| `usd_eur_rate` | float | Level (USD per 1 EUR) | Exchange rate: number of US dollars per one euro. Monthly ECB values are averaged to one value per year. | ECB SDW series `EXR/M.USD.EUR.SP00.A` (monthly → yearly mean). |
| `hicp_inflation` | float | Percent (%) | HICP inflation: annual rate of change of the Harmonised Index of Consumer Prices, overall index, euro area. Monthly ECB values are averaged to one value per year. | ECB SDW series `ICP/M.U2.N.000000.4.ANR` (monthly → yearly mean). |
| `unemployment_rate` | float | Percent (%) | Unemployment rate: percentage of active population aged 15–74, euro area, both sexes. | Eurostat `une_rt_a`, filtered to EA20, sex T, age Y15-74, unit PC_ACT. |
| `short_term_rate` | float | Percent (%) | Short-term money market interest rate (3-month Euribor). Monthly ECB values are averaged to one value per year. | ECB SDW series `FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA` (monthly → yearly mean). |
| `gov_debt_gdp` | float | Percent of GDP (%) | General government gross debt as a percentage of GDP, euro area. | Eurostat `tipsgo10`, filtered to EA20. |

- **Time coverage:** Depends on data availability; typically from the late 1990s / 2000 to the latest full year. Exact range is in the file.
- **Missing values:** Rows with missing `year` or target/source values are dropped per indicator during build. The merged panel uses a **left join** on `year` (anchored to GDP years), so indicators with shorter coverage will have NaN in early years. These are forward/backward-filled (`ffill().bfill()`) by the modelling pipeline before feature engineering.
- **Monthly-to-annual aggregation:** ECB monthly series (`usd_eur_rate`, `hicp_inflation`, `short_term_rate`) are aggregated to annual by simple arithmetic mean of all months in each calendar year. Eurostat series are published as annual.

---

## 2. Derived variables (after feature engineering)

The pipeline applies `build_full_features()` (see `src/feature_engineering.py`) to `panel_yearly.csv` before modelling. The following columns are added; all are used as **features** except `gdp_growth` (target) and `year` (excluded from features).

### 2.1 Lag features (previous year's value)

| Column | Definition |
|--------|-----------|
| `usd_eur_rate_lag1` | USD/EUR exchange rate shifted by 1 year. |
| `hicp_inflation_lag1` | HICP inflation shifted by 1 year. |
| `unemployment_rate_lag1` | Unemployment rate shifted by 1 year. |
| `short_term_rate_lag1` | Short-term interest rate shifted by 1 year. |
| `gov_debt_gdp_lag1` | Government debt-to-GDP shifted by 1 year. |

### 2.2 Change features (year-on-year first difference)

| Column | Definition |
|--------|-----------|
| `usd_eur_rate_chg1` | First difference: `usd_eur_rate(t) − usd_eur_rate(t−1)`. |
| `hicp_inflation_chg1` | First difference: `hicp_inflation(t) − hicp_inflation(t−1)`. |
| `short_term_rate_chg1` | First difference: `short_term_rate(t) − short_term_rate(t−1)`. |

- **Dropped rows:** After adding lags and differences, the first row(s) with NaN are dropped (`.dropna()`), so the modelling sample starts from the second year onward.
- **Total model features:** 5 raw levels + 5 lags + 3 changes = **13 features** (year and gdp_growth excluded).

---

## 3. Raw data file formats

### 3.1 Eurostat GDP growth: `data/raw/eurostat_gdp_growth.csv`

- **Source:** Eurostat dataset [tec00115](https://ec.europa.eu/eurostat/databrowser/view/tec00115) (Real GDP growth rate – volume).
- **Filters:** `freq=A`, `unit=PC_GDP`, `na_item=B1GQ`, `geo=EA19` or `EA20`.
- **Key columns:** `freq`, `unit`, `na_item`, `geo`, `time` (year), `value` (GDP growth %).

### 3.2 ECB exchange rate: `data/raw/ecb_exr_usd_eur.csv`

- **Source:** ECB Statistical Data Warehouse, series `EXR/M.USD.EUR.SP00.A`.
- **Key columns:** `TIME_PERIOD` (e.g. `2000-01`), `OBS_VALUE` (USD per 1 EUR).
- **Aggregation:** Monthly → yearly mean in `build_dataset.py`.

### 3.3 ECB HICP inflation: `data/raw/ecb_hicp_inflation.csv`

- **Source:** ECB SDW, series `ICP/M.U2.N.000000.4.ANR` (HICP, annual rate of change).
- **Key columns:** `TIME_PERIOD`, `OBS_VALUE` (%).
- **Aggregation:** Monthly → yearly mean.

### 3.4 Eurostat unemployment rate: `data/raw/eurostat_unemployment_rate.csv`

- **Source:** Eurostat dataset `une_rt_a` (annual unemployment rate).
- **Filters:** `geo=EA20`, `sex=T`, `age=Y15-74`, `unit=PC_ACT`.
- **Key columns:** `time` (year), `value` (%).

### 3.5 ECB short-term interest rate: `data/raw/ecb_short_term_rate.csv`

- **Source:** ECB SDW, series `FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA` (3-month Euribor).
- **Key columns:** `TIME_PERIOD`, `OBS_VALUE` (%).
- **Aggregation:** Monthly → yearly mean.

### 3.6 Eurostat government debt: `data/raw/eurostat_gov_debt_gdp.csv`

- **Source:** Eurostat dataset `tipsgo10` (general government gross debt, % of GDP).
- **Filters:** `geo=EA20`.
- **Key columns:** `time` (year), `value` (% of GDP).

---

## 4. Country-level panel: `data/processed/panel_country_yearly.csv`

Produced by `python -m src.build_country_panel`. One row per country per year.

| Column | Type | Description |
|--------|------|-------------|
| `country` | string (ISO-2) | Country code (e.g. DE, FR, IT, ES, NL, BE, AT, PT, IE, FI, EL). |
| `year` | integer | Calendar year. |
| `gdp_growth` | float | Real GDP growth rate (% change, volume) from Eurostat `tec00115` (unit CLV_PCH_PRE). |

---

## 5. Other processed outputs

- **`data/processed/indicator_list.csv`:** Metadata for each indicator (name, source, file_stem, value_name, description). For reference only.
- **`data/processed/figures/`:** SHAP summary, PDP, permutation importance, and local explanation plots from `run_interpretation`.
- **`data/processed/model_comparison.png`:** Bar chart from `compare_models`.

---

## 6. Version and reproducibility

- **Indicator definitions:** `src/indicators.py` and `INDICATORS.md`.
- **Build script:** `python -m src.build_dataset` (after `python -m src.fetch_real_data` if using API).
- **Data dictionary version:** This file; update when adding new indicators or columns to `panel_yearly.csv`.
