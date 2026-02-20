"""Data acquisition utilities for Eurostat/ECB APIs."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from io import StringIO
import gzip
import json
from itertools import product
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pandas as pd

EUROSTAT_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
EUROSTAT_BULK_BASE_URL = (
    "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing"
)
ECB_BASE_URL = "https://data-api.ecb.europa.eu/service/data"


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


def fetch_eurostat_indicator(
    dataset_id: str,
    filters: Optional[Dict[str, str]],
    since_time_period: Optional[str],
) -> pd.DataFrame:
    """Fetch a Eurostat dataset with fallback behaviour."""
    attempts = [
        (filters, since_time_period),
        (filters, None),
        (None, since_time_period),
        (None, None),
    ]
    last_error: Optional[Exception] = None
    for attempt_filters, attempt_since in attempts:
        try:
            return fetch_eurostat_dataset(
                dataset_id=dataset_id,
                filters=attempt_filters,
                since_time_period=attempt_since,
            )
        except (HTTPError, ValueError) as exc:
            last_error = exc
            continue
    raise ValueError("Eurostat fetch failed after retries.") from last_error


def fetch_indicators(
    indicators: Iterable[Dict[str, object]],
    eurostat_since: str = "2000",
    ecb_start: str = "2000-01",
) -> Dict[str, pd.DataFrame]:
    """Fetch datasets described by indicator specs."""
    datasets: Dict[str, pd.DataFrame] = {}
    for indicator in indicators:
        name = str(indicator["name"])
        source = indicator["source"]
        if source == "eurostat":
            dataset_id = str(indicator["dataset_id"])
            filters = indicator.get("filters")
            df = fetch_eurostat_indicator(
                dataset_id=dataset_id,
                filters=filters if isinstance(filters, dict) else None,
                since_time_period=eurostat_since,
            )
        elif source == "ecb":
            flow_ref = str(indicator["flow_ref"])
            series_key = str(indicator["series_key"])
            df = fetch_ecb_series_csv(
                flow_ref=flow_ref,
                series_key=series_key,
                start_period=ecb_start,
            )
        else:
            raise ValueError(f"Unsupported source: {source}")
        datasets[name] = df
    return datasets


def fetch_eurostat_dataset(
    dataset_id: str,
    filters: Optional[Dict[str, str]] = None,
    since_time_period: Optional[str] = None,
    lang: str = "EN",
) -> pd.DataFrame:
    """Fetch a Eurostat dataset in TSV and return a tidy dataframe.

    Args:
        dataset_id: Eurostat dataset code (e.g., "tec00115").
        filters: Optional dimension filters (e.g., {"geo": "EA19"}).
        since_time_period: Start period filter (e.g., "2000").
        lang: Response language (default "EN").

    Returns:
        Long-format dataframe with dimensions, time, and value.
    """
    params = {"format": "JSON", "lang": lang}
    if since_time_period:
        params["sinceTimePeriod"] = since_time_period
    if filters:
        params.update(filters)

    url = f"{EUROSTAT_BASE_URL}/{dataset_id}?{urlencode(params)}"
    try:
        content = _http_get_text(url, timeout=30)
    except HTTPError as exc:
        if exc.code == 400:
            content = _fetch_eurostat_bulk(dataset_id)
        else:
            raise HTTPError(
                url,
                exc.code,
                f"{exc.msg}. Check dataset filters or code lists.",
                exc.hdrs,
                exc.fp,
            ) from exc

    if content.lstrip().startswith("{"):
        payload = json.loads(content)
        dataset = _extract_jsonstat_dataset(payload)
        return _parse_eurostat_jsonstat(dataset)

    df_raw = pd.read_csv(StringIO(content), sep="\t")
    first_col = df_raw.columns[0]
    if "\\\\" not in first_col:
        raise ValueError("Unexpected Eurostat TSV format: missing dimension header.")

    dim_part, _ = first_col.split("\\\\", maxsplit=1)
    dim_cols = dim_part.split(",")

    df_long = df_raw.melt(id_vars=[first_col], var_name="time", value_name="value")
    df_long[dim_cols] = df_long[first_col].str.split(",", expand=True)
    df_long.drop(columns=[first_col], inplace=True)

    df_long["value"] = (
        df_long["value"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .str.strip()
        .replace("", pd.NA)
    )
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    return df_long


def _fetch_eurostat_bulk(dataset_id: str) -> str:
    """Fetch Eurostat bulk TSV (gz) and return decoded text."""
    candidates = [
        f"{EUROSTAT_BULK_BASE_URL}?downfile=data/{dataset_id}.tsv.gz",
        f"{EUROSTAT_BULK_BASE_URL}?file=data/{dataset_id}.tsv.gz",
        f"{EUROSTAT_BULK_BASE_URL}?downfile=data/{dataset_id}.tsv",
        f"{EUROSTAT_BULK_BASE_URL}?file=data/{dataset_id}.tsv",
        f"{EUROSTAT_BULK_BASE_URL}?downfile=data/{dataset_id}.tsv.gz&lang=en",
        f"{EUROSTAT_BULK_BASE_URL}?file=data/{dataset_id}.tsv.gz&lang=en",
    ]
    last_error: Optional[Exception] = None
    for url in candidates:
        try:
            raw = _http_get_bytes(url, timeout=60)
            if raw[:2] == b"\x1f\x8b":
                return gzip.decompress(raw).decode("utf-8")
            return raw.decode("utf-8")
        except Exception as exc:  # noqa: BLE001 - try next candidate
            last_error = exc
            continue
    raise HTTPError(
        candidates[-1],
        410,
        "Eurostat bulk download URLs failed.",
        None,
        None,
    ) from last_error


def _extract_jsonstat_dataset(payload: Dict[str, object]) -> Dict[str, object]:
    """Extract the dataset object from a JSON-stat response."""
    if "error" in payload:
        error = payload.get("error", {})
        if isinstance(error, dict):
            message = error.get("message") or error.get("description") or error
        else:
            message = error
        raise ValueError(f"Eurostat API error: {message}")
    if "dataset" in payload and isinstance(payload["dataset"], dict):
        return payload["dataset"]
    return payload


def _parse_eurostat_jsonstat(payload: Dict[str, object]) -> pd.DataFrame:
    """Parse a JSON-stat response into a tidy dataframe."""
    dimension = payload.get("dimension", {})
    dim_ids = dimension.get("id", [])
    dim_sizes = dimension.get("size", [])
    if not dim_ids or not dim_sizes:
        dim_ids = payload.get("id", [])
        dim_sizes = payload.get("size", [])
    if not dim_ids or not dim_sizes:
        keys = ", ".join(sorted(payload.keys()))
        raise ValueError(f"Unexpected Eurostat JSON format: missing dimensions (keys: {keys}).")

    categories = []
    for dim_id in dim_ids:
        dim_info = dimension.get(dim_id, {})
        cat = dim_info.get("category", {})
        index = cat.get("index")
        if isinstance(index, dict):
            codes = [code for code, pos in sorted(index.items(), key=lambda item: item[1])]
        elif isinstance(index, list):
            codes = index
        else:
            labels = cat.get("label", {})
            codes = list(labels.keys())
        categories.append(codes)

    total = 1
    for size in dim_sizes:
        total *= size

    values_raw = payload.get("value", {})
    values = [pd.NA] * total
    if isinstance(values_raw, list):
        values = values_raw
    elif isinstance(values_raw, dict):
        for key, value in values_raw.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < total:
                values[idx] = value

    records = list(product(*categories))
    df = pd.DataFrame(records, columns=dim_ids)
    df["value"] = pd.to_numeric(values, errors="coerce")
    return df


def _http_get_text(url: str, timeout: int) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _http_get_bytes(url: str, timeout: int) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_ecb_series_csv(
    flow_ref: str,
    series_key: str,
    start_period: Optional[str] = None,
    end_period: Optional[str] = None,
    extra_params: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Fetch an ECB SDW series in CSV format.

    Args:
        flow_ref: Dataflow reference (e.g., "EXR").
        series_key: SDMX series key (e.g., "M.USD.EUR.SP00.A").
        start_period: Optional start period (e.g., "2000-01").
        end_period: Optional end period.
        extra_params: Optional extra query params.

    Returns:
        Dataframe parsed from ECB CSV response.
    """
    params = {"format": "csvdata"}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period
    if extra_params:
        params.update(extra_params)

    url = f"{ECB_BASE_URL}/{flow_ref}/{series_key}?{urlencode(params)}"
    return pd.read_csv(url)


def save_raw_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Save a dataframe to disk as CSV."""
    df.to_csv(output_path, index=False)


def fetch_default_indicators(
    eurostat_since: str = "2000",
    ecb_start: str = "2000-01",
) -> Dict[str, pd.DataFrame]:
    """Fetch default Eurostat GDP growth and an ECB FX series.

    Returns:
        Dictionary with raw dataframes keyed by indicator name.
    """
    eurostat_attempts = [
        ({"unit": "PC_GDP", "geo": "EA20"}, eurostat_since),
        ({"unit": "PC_GDP", "geo": "EA19"}, eurostat_since),
        ({"unit": "PC_GDP", "geo": "EA20"}, None),
        ({"unit": "PC_GDP", "geo": "EA19"}, None),
        (None, eurostat_since),
        (None, None),
    ]
    last_error: Optional[Exception] = None
    eurostat_df = None
    for filters, since_time in eurostat_attempts:
        try:
            eurostat_df = fetch_eurostat_dataset(
                dataset_id="tec00115",
                filters=filters,
                since_time_period=since_time,
            )
            break
        except (HTTPError, ValueError) as exc:
            last_error = exc
            continue
    if eurostat_df is None:
        raise ValueError("Eurostat fetch failed after retries.") from last_error

    if "geo" in eurostat_df.columns:
        if "EA20" in eurostat_df["geo"].unique():
            eurostat_df = eurostat_df[eurostat_df["geo"] == "EA20"]
        elif "EA19" in eurostat_df["geo"].unique():
            eurostat_df = eurostat_df[eurostat_df["geo"] == "EA19"]

    if "unit" in eurostat_df.columns:
        if "PC_GDP" in eurostat_df["unit"].unique():
            eurostat_df = eurostat_df[eurostat_df["unit"] == "PC_GDP"]
    ecb_df = fetch_ecb_series_csv(
        flow_ref="EXR",
        series_key="M.USD.EUR.SP00.A",
        start_period=ecb_start,
    )
    return {"eurostat_gdp_growth": eurostat_df, "ecb_exr_usd_eur": ecb_df}
