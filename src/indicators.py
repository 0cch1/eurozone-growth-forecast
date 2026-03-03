"""Indicator registry for data ingestion."""

from __future__ import annotations

from typing import Dict, List


def get_indicator_specs() -> List[Dict[str, object]]:
    """Return indicator specs for ingestion and dataset build."""
    return [
        {
            "name": "gdp_growth",
            "source": "eurostat",
            "dataset_id": "tec00115",
            "file_stem": "eurostat_gdp_growth",
            "value_name": "gdp_growth",
            "filters": {
                "unit": "PC_GDP",
                "geo": "EA20",
                "freq": "A",
                "na_item": "B1GQ",
            },
            "description": "Real GDP growth rate (percent change).",
        },
        {
            "name": "usd_eur_rate",
            "source": "ecb",
            "flow_ref": "EXR",
            "series_key": "M.USD.EUR.SP00.A",
            "file_stem": "ecb_exr_usd_eur",
            "value_name": "usd_eur_rate",
            "description": "ECB reference exchange rate, USD/EUR (monthly).",
        },
        {
            "name": "hicp_inflation",
            "source": "ecb",
            "flow_ref": "ICP",
            "series_key": "M.U2.N.000000.4.ANR",
            "file_stem": "ecb_hicp_inflation",
            "value_name": "hicp_inflation",
            "description": "HICP inflation, overall index, annual rate of change, euro area (monthly, ECB).",
        },
        {
            "name": "unemployment_rate",
            "source": "eurostat",
            "dataset_id": "une_rt_a",
            "file_stem": "eurostat_unemployment_rate",
            "value_name": "unemployment_rate",
            "filters": {
                "geo": "EA20",
                "sex": "T",
                "age": "Y15-74",
                "unit": "PC_ACT",
            },
            "description": "Unemployment rate, percentage of active population (PC_ACT), annual, euro area.",
        },
        {
            "name": "short_term_rate",
            "source": "ecb",
            "flow_ref": "FM",
            "series_key": "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
            "file_stem": "ecb_short_term_rate",
            "value_name": "short_term_rate",
            "description": "Short-term money market interest rate (3-month Euribor, monthly).",
        },
        {
            "name": "gov_debt_gdp",
            "source": "eurostat",
            "dataset_id": "tipsgo10",
            "file_stem": "eurostat_gov_debt_gdp",
            "value_name": "gov_debt_gdp",
            "filters": {
                "geo": "EA20",
            },
            "description": "General government gross debt as a percentage of GDP, annual, euro area.",
        },
    ]
