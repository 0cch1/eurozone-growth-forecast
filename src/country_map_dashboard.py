"""Minimal country-level GDP growth map (MVP dashboard).

This script reads ``data/processed/panel_country_yearly.csv`` and produces a
static HTML choropleth map. By default it includes all years with a slider;
optional single-year mode is supported via CLI.

- No login/back-end, just an HTML file.
- One file with year slider, or one map for a chosen year.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional dependency
    px = None  # type: ignore[misc,assignment]
    go = None  # type: ignore[misc,assignment]

# Plotly choropleth accepts ISO-3 only; map Eurostat geo (ISO-2) to ISO-3.
# Eurostat uses EL for Greece.
ISO2_TO_ISO3 = {
    "DE": "DEU", "FR": "FRA", "IT": "ITA", "ES": "ESP", "NL": "NLD", "BE": "BEL",
    "AT": "AUT", "PT": "PRT", "IE": "IRL", "FI": "FIN", "GR": "GRC", "EL": "GRC",
}

# Approximate (lat, lon) centroids for text labels on the map.
ISO3_CENTROIDS = {
    "DEU": (51.2, 10.5), "FRA": (46.2, 2.2), "ITA": (41.9, 12.6),
    "ESP": (40.4, -3.7), "NLD": (52.1, 5.3), "BEL": (50.5, 4.5),
    "AUT": (47.6, 14.6), "PRT": (39.4, -8.2), "IRL": (53.1, -8.0),
    "FIN": (64.0, 26.0), "GRC": (39.1, 21.8),
}


def build_map(
    year: Optional[int] = None,
    all_years: bool = True,
    output_html: str = "results/country_gdp_growth_map.html",
) -> Path:
    """Create a choropleth map of country-level GDP growth.

    If all_years is True (default), the HTML includes a year slider. If year is
    set and all_years is False, only that year is shown.
    """
    project_root = Path(__file__).resolve().parents[1]
    panel_path = project_root / "data" / "processed" / "panel_country_yearly.csv"
    if not panel_path.exists():
        raise FileNotFoundError(
            "Country panel not found. Run `python -m src.build_country_panel` first."
        )

    if px is None:
        raise ImportError(
            "plotly is not installed. Install with `pip install plotly` to enable "
            "the country-level map visualisation."
        )

    df = pd.read_csv(panel_path)
    if "year" not in df.columns or "country" not in df.columns:
        raise ValueError("panel_country_yearly.csv must contain 'country' and 'year'.")

    df = df.copy()
    df["country_iso3"] = df["country"].map(ISO2_TO_ISO3)
    df = df.dropna(subset=["country_iso3"])

    if not all_years and year is not None:
        df = df[df["year"] == year]
        if df.empty:
            raise ValueError(f"No data available for year {year} in country panel.")
        year_title = str(year)
    else:
        year_title = "by year (use slider)"

    min_val = float(df["gdp_growth"].min())
    max_val = float(df["gdp_growth"].max())

    fig = px.choropleth(
        df,
        locations="country_iso3",
        locationmode="ISO-3",
        color="gdp_growth",
        hover_name="country",
        hover_data={"country_iso3": False, "gdp_growth": ":.2f"},
        color_continuous_scale="RdYlGn",
        range_color=(min_val, max_val),
        labels={"gdp_growth": "GDP growth (%)"},
        title=f"Real GDP growth (%) — {year_title}\nSource: Eurostat tec00115 (real GDP, volume % change). Recent years may show fewer countries (release lag).",
        animation_frame="year" if all_years else None,
    )

    if all_years:
        fig.layout.sliders[0].currentvalue.prefix = "Year: "
        # Visual transition of the slider thumb when changing year.
        fig.layout.sliders[0].transition = dict(duration=2000)
        # Frame duration when using the play button (ms per year).
        if fig.layout.updatemenus and fig.layout.updatemenus[0].buttons:
            play_button_args = fig.layout.updatemenus[0].buttons[0].args
            if len(play_button_args) > 1 and "frame" in play_button_args[1]:
                play_button_args[1]["frame"]["duration"] = 2000

    # World map (no scope restriction).
    fig.update_geos(showcountries=True)

    # Add a Scattergeo trace to show GDP growth (%) as text at country centroids.
    def _scatter_text_trace(df_sub: pd.DataFrame):
        df_sub = df_sub.sort_values("country_iso3")
        lats = [ISO3_CENTROIDS.get(iso3, (0, 0))[0] for iso3 in df_sub["country_iso3"]]
        lons = [ISO3_CENTROIDS.get(iso3, (0, 0))[1] for iso3 in df_sub["country_iso3"]]
        texts = [f"{v:.1f}%" for v in df_sub["gdp_growth"]]
        return go.Scattergeo(
            lat=lats,
            lon=lons,
            text=texts,
            mode="text",
            textfont=dict(size=12, color="black"),
            showlegend=False,
            geo="geo",
            hoverinfo="skip",
        )

    if all_years:
        years_sorted = sorted(df["year"].unique())
        first_year = int(years_sorted[0])
        df_first = df[df["year"] == first_year]
        fig.add_trace(_scatter_text_trace(df_first))
        for frame in fig.frames:
            y = int(frame.name)
            df_y = df[df["year"] == y]
            frame.data = list(frame.data) + [_scatter_text_trace(df_y)]
    else:
        fig.add_trace(_scatter_text_trace(df))

    fig.update_layout(margin=dict(l=0, r=0, t=80, b=0))

    out_path = project_root / output_html
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    print(f"Saved country GDP growth map to {out_path}" + (" (all years, use slider)" if all_years else f" (year {year})"))
    return out_path


def main() -> None:
    """CLI entry point: build a map with year slider (default) or for one year."""
    import sys

    year_arg: Optional[int] = None
    if len(sys.argv) > 1:
        try:
            year_arg = int(sys.argv[1])
        except ValueError:
            raise SystemExit(f"Expected integer year, got: {sys.argv[1]!r}")

    # With no argument: all years + slider. With a year: single-year map.
    build_map(year=year_arg, all_years=(year_arg is None))


if __name__ == "__main__":
    main()

