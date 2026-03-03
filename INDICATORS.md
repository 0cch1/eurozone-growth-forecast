## Indicator list

This list is used by the ingestion and dataset build scripts.
Edit `src/indicators.py` to add more indicators.

### Current indicators (implemented)

| Name | Source | Code / Series | Frequency | Description |
| --- | --- | --- | --- | --- |
| gdp_growth | Eurostat | `tec00115` | Annual | Real GDP growth rate (percent change). |
| usd_eur_rate | ECB | `EXR/M.USD.EUR.SP00.A` | Monthly | ECB reference exchange rate, USD/EUR. |
| hicp_inflation | ECB | `ICP/M.U2.N.000000.4.ANR` | Monthly | HICP inflation, overall index, annual rate of change, euro area (ECB). |
| unemployment_rate | Eurostat | `une_rt_a` | Annual | Unemployment rate (% of active population, PC_ACT), euro area. |
| short_term_rate | ECB | `FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA` | Monthly | Short-term money market interest rate (3-month Euribor). |
| gov_debt_gdp | Eurostat | `tipsgo10` | Annual | General government gross debt as a percentage of GDP, euro area. |

### Planned indicators (TBD)

- Government expenditure.
- Investment.
- External balance indicators.
