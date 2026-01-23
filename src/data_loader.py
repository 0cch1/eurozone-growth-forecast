"""Data acquisition utilities for Eurostat/ECB APIs."""

from __future__ import annotations

from typing import Dict, Optional


def fetch_eurostat_series(series_id: str, params: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Placeholder for Eurostat API fetch.

    Args:
        series_id: Eurostat series identifier.
        params: Optional query parameters.

    Returns:
        A dictionary representing raw API payload (stub).
    """
    return {
        "source": "eurostat",
        "series_id": series_id,
        "params": params or {},
    }


def fetch_ecb_series(series_key: str, params: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Placeholder for ECB API fetch.

    Args:
        series_key: ECB SDW series key.
        params: Optional query parameters.

    Returns:
        A dictionary representing raw API payload (stub).
    """
    return {
        "source": "ecb",
        "series_key": series_key,
        "params": params or {},
    }
