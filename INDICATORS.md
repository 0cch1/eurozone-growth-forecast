## Indicator list

This list is used by the ingestion and dataset build scripts.
Edit `src/indicators.py` to add more indicators.

### Current indicators (implemented)

| Name | Source | Code / Series | Frequency | Description |
| --- | --- | --- | --- | --- |
| gdp_growth | Eurostat | `tec00115` | Annual | Real GDP growth rate (percent change). |
| usd_eur_rate | ECB | `EXR/M.USD.EUR.SP00.A` | Monthly | ECB reference exchange rate, USD/EUR. |

### Planned indicators (TBD)

- Inflation (HICP, annual).
- Unemployment rate (annual).
- Short-term interest rates.
- Government expenditure.
- Public debt.
- Investment.
- External balance indicators.
