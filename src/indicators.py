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
    ]
