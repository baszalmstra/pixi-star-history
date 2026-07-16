"""Generate the static interactive GitHub Pages visualization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .collector import DEFAULT_CSV, StarRow, read_rows
from .crossover import Crossover, pairwise_crossovers
from .forecast import Forecast, forecast

COLORS = {
    "conda/conda": "#2563eb",
    "prefix-dev/pixi": "#e59f00",
    "mamba-org/mamba": "#0f9d78",
}
LABELS = {
    "conda/conda": "Conda",
    "prefix-dev/pixi": "Pixi",
    "mamba-org/mamba": "Mamba",
}
FORECAST_DAYS = 183


@dataclass(frozen=True)
class ChartAnalysis:
    figure: go.Figure
    forecasts: dict[str, Forecast]
    crossovers: list[Crossover]


def rgba(hex_color: str, alpha: float) -> str:
    values = tuple(int(hex_color[index : index + 2], 16) for index in (1, 3, 5))
    return f"rgba({values[0]}, {values[1]}, {values[2]}, {alpha})"


def daily_series(rows: list[StarRow]) -> tuple[list[date], np.ndarray, list[str], np.ndarray]:
    """Fill rare missed snapshot dates with linear estimates for time-series fitting."""
    ordered = sorted(rows, key=lambda row: row.day)
    exact = {row.day: row for row in ordered}
    first, last = ordered[0].day, ordered[-1].day
    dates = [first + timedelta(days=offset) for offset in range((last - first).days + 1)]
    known_x = np.array([(row.day - first).days for row in ordered], dtype=float)
    known_y = np.array([row.stars for row in ordered], dtype=float)
    values = np.rint(np.interp(np.arange(len(dates)), known_x, known_y))
    observations = [exact[day].observation if day in exact else "estimated" for day in dates]
    changes = np.diff(np.concatenate(([0.0], values)))
    return dates, values, observations, changes


def add_repository(
    fig: go.Figure, repository: str, rows: list[StarRow]
) -> tuple[date, float, Forecast]:
    dates, values, observations, changes = daily_series(rows)
    result = forecast(values, FORECAST_DAYS)
    color = COLORS[repository]
    label = LABELS[repository]
    latest_day = dates[-1]
    future_dates = [latest_day + timedelta(days=offset) for offset in range(FORECAST_DAYS + 1)]
    predicted = np.rint(np.concatenate(([values[-1]], result.predicted)))
    lower = np.rint(np.concatenate(([values[-1]], result.lower)))
    upper = np.rint(np.concatenate(([values[-1]], result.upper)))

    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=lower,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=repository,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=upper,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=rgba(color, 0.10),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=repository,
        )
    )

    source_labels = np.array(
        ["estimated" if observation == "estimated" else "observed" for observation in observations]
    )
    historical_custom = np.column_stack((changes.astype(int), source_labels))
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            name=label,
            mode="lines",
            line={"color": color, "width": 2.4},
            customdata=historical_custom,
            hovertemplate=(
                f"<b>{label}</b> · %{{y:,.0f}} stars<br>"
                "%{customdata[0]:+,.0f} that day · %{customdata[1]}"
                "<extra></extra>"
            ),
            legendgroup=repository,
        )
    )

    anchors = [row for row in rows if row.observation in {"wayback", "snapshot"}]
    fig.add_trace(
        go.Scatter(
            x=[row.day for row in anchors],
            y=[row.stars for row in anchors],
            mode="markers",
            marker={
                "color": "white",
                "line": {"color": color, "width": 1.4},
                "size": 5,
            },
            name=f"{label} observations",
            hoverinfo="skip",
            showlegend=False,
            legendgroup=repository,
        )
    )

    forecast_custom = np.column_stack((lower, upper))
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predicted,
            mode="lines",
            line={"color": color, "width": 2.2, "dash": "dash"},
            customdata=forecast_custom,
            hovertemplate=(
                f"<b>{label}</b> · %{{y:,.0f}} forecast<br>"
                "95% range %{customdata[0]:,.0f}–%{customdata[1]:,.0f}"
                "<extra></extra>"
            ),
            showlegend=False,
            legendgroup=repository,
        )
    )
    return latest_day, values[-1], result


def add_crossovers(fig: go.Figure, crossovers: list[Crossover]) -> None:
    """Add probable overtake dates and conditional 95% date ranges."""
    for crossover in crossovers:
        if crossover.probability < 0.10 or crossover.median_date is None:
            continue
        assert crossover.p05_date is not None
        assert crossover.p95_date is not None
        assert crossover.standard_deviation_days is not None
        assert crossover.variance_days is not None
        assert crossover.median_stars is not None
        challenger = LABELS[crossover.challenger]
        incumbent = LABELS[crossover.incumbent]
        color = COLORS[crossover.challenger]
        fig.add_vrect(
            x0=crossover.p05_date.isoformat(),
            x1=crossover.p95_date.isoformat(),
            fillcolor=rgba(color, 0.055),
            line_width=0,
            layer="below",
        )
        fig.add_trace(
            go.Scatter(
                x=[crossover.median_date],
                y=[crossover.median_stars],
                mode="markers+text",
                marker={
                    "symbol": "diamond",
                    "size": 9,
                    "color": color,
                    "line": {"color": "white", "width": 1.5},
                },
                text=[f"{challenger} → {incumbent}"],
                textposition="top center",
                textfont={"color": color, "size": 11},
                customdata=[
                    [
                        round(crossover.probability, 6),
                        crossover.p05_date.isoformat(),
                        crossover.p95_date.isoformat(),
                        round(crossover.standard_deviation_days, 2),
                        round(crossover.variance_days, 2),
                    ]
                ],
                hovertemplate=(
                    f"<b>{challenger} → {incumbent}</b><br>"
                    "%{customdata[0]:.0%} chance within six months"
                    "<extra></extra>"
                ),
                showlegend=False,
                cliponaxis=False,
            )
        )


def build_figure(rows: list[StarRow]) -> ChartAnalysis:
    grouped: dict[str, list[StarRow]] = defaultdict(list)
    for row in rows:
        grouped[row.repository].append(row)
    if not grouped:
        raise ValueError("The star-history CSV contains no rows")

    figure = go.Figure()
    repository_forecasts = {
        repository: add_repository(figure, repository, grouped[repository]) for repository in COLORS
    }
    forecast_start = max(item[0] for item in repository_forecasts.values())
    crossovers = pairwise_crossovers(
        {
            repository: (current, result)
            for repository, (_, current, result) in repository_forecasts.items()
        },
        forecast_start,
    )
    add_crossovers(figure, crossovers)
    figure.add_shape(
        type="line",
        x0=forecast_start.isoformat(),
        x1=forecast_start.isoformat(),
        y0=0,
        y1=1,
        yref="paper",
        line={"color": "#94a3b8", "width": 1, "dash": "dot"},
    )
    figure.add_annotation(
        x=forecast_start.isoformat(),
        y=0.99,
        yref="paper",
        text="forecast",
        font={"color": "#64748b", "size": 11},
        showarrow=False,
        xanchor="left",
    )
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#fbfcfe",
        plot_bgcolor="#fbfcfe",
        margin={"l": 62, "r": 28, "t": 22, "b": 58},
        hovermode="x unified",
        hoverlabel={
            "align": "left",
            "bgcolor": "white",
            "bordercolor": "#dbe1ea",
            "font": {"family": "Inter, sans-serif", "size": 12, "color": "#172033"},
        },
        font={"family": "Inter, ui-sans-serif, system-ui, sans-serif", "color": "#172033"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 13},
        },
        xaxis={
            "title": None,
            "showgrid": False,
            "unifiedhovertitle": {"text": "<b>%{x|%e %b %Y}</b>"},
            "linecolor": "#dbe1ea",
            "rangeselector": {
                "buttons": [
                    {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                    {"count": 3, "label": "3y", "step": "year", "stepmode": "backward"},
                    {"label": "all", "step": "all"},
                ],
                "bgcolor": "white",
                "bordercolor": "#dbe1ea",
                "borderwidth": 1,
                "font": {"size": 11},
                "x": 1,
                "xanchor": "right",
                "y": 1.01,
                "yanchor": "bottom",
            },
        },
        yaxis={
            "title": "GitHub stars",
            "rangemode": "tozero",
            "gridcolor": "#e8ecf2",
            "gridwidth": 1,
            "zeroline": False,
            "tickformat": ",~s",
        },
    )
    return ChartAnalysis(
        figure=figure,
        forecasts={
            repository: result for repository, (_, _, result) in repository_forecasts.items()
        },
        crossovers=crossovers,
    )


def compact_date(day: date, *, include_year: bool = False) -> str:
    value = f"{day.day} {day.strftime('%b')}"
    return f"{value} {day.year}" if include_year else value


def render_crossover_summary(crossovers: list[Crossover]) -> str:
    items: list[str] = []
    for crossover in crossovers:
        if crossover.probability < 0.10 or crossover.median_date is None:
            continue
        assert crossover.p05_date is not None
        assert crossover.p95_date is not None
        assert crossover.standard_deviation_days is not None
        challenger = LABELS[crossover.challenger]
        incumbent = LABELS[crossover.incumbent]
        items.append(
            '<div class="overtake">'
            f'<strong>{challenger} <span aria-hidden="true">→</span> {incumbent}</strong>'
            f'<span class="overtake-date">{compact_date(crossover.median_date, include_year=True)}</span>'
            f'<span class="probability">{crossover.probability:.0%}</span>'
            '<span class="range">'
            f"95%: {compact_date(crossover.p05_date)}–{compact_date(crossover.p95_date)}"
            f" · σ {crossover.standard_deviation_days:.1f}d"
            "</span></div>"
        )
    return "".join(items) or '<span class="muted">No likely crossover within six months</span>'


def render_model_details(forecasts: dict[str, Forecast], crossovers: list[Crossover]) -> str:
    diagnostics: list[str] = []
    for repository, result in forecasts.items():
        strongest = sorted(result.weights.items(), key=lambda item: item[1], reverse=True)[:3]
        weights = " · ".join(f"{name} {weight:.0%}" for name, weight in strongest)
        diagnostics.append(
            '<div class="diagnostic">'
            f"<strong>{LABELS[repository]}</strong>"
            f"<span>30-day backtest error ±{result.backtest_rmse:,.0f} stars</span>"
            f"<span>{weights}</span>"
            "</div>"
        )
    variances = []
    for crossover in crossovers:
        if crossover.probability < 0.10 or crossover.variance_days is None:
            continue
        variances.append(
            f"{LABELS[crossover.challenger]} → {LABELS[crossover.incumbent]} "
            f"date variance {crossover.variance_days:,.0f} days²"
        )
    variance_note = " · ".join(variances)
    return "".join(diagnostics) + f'<p class="variance-note">{variance_note}</p>'


def build_site(csv_path: Path = DEFAULT_CSV, output: Path = Path("site/index.html")) -> str:
    rows = read_rows(csv_path)
    analysis = build_figure(rows)
    chart = analysis.figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id="star-history-plot",
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    latest = max(row.day for row in rows).isoformat()
    crossover_summary = render_crossover_summary(analysis.crossovers)
    model_details = render_model_details(analysis.forecasts, analysis.crossovers)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Historical and forecast GitHub stars for Conda, Pixi, and Mamba.">
  <title>Conda, Pixi &amp; Mamba — star history</title>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; min-height: 100%; background: #fbfcfe; color: #172033; }}
    body {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ min-height: 100vh; display: grid; grid-template-rows: auto auto minmax(560px, 1fr) auto auto; padding: 32px 4vw 20px; }}
    header {{ display: flex; align-items: end; justify-content: space-between; gap: 24px; padding: 0 8px 14px; }}
    h1 {{ margin: 0; font-size: clamp(1.45rem, 3vw, 2.35rem); letter-spacing: -0.04em; font-weight: 650; }}
    .subtitle {{ margin: 7px 0 0; color: #64748b; font-size: 0.92rem; }}
    .updated, .muted {{ color: #64748b; font-size: 0.78rem; white-space: nowrap; }}
    .guide {{ margin: 0 0 10px; padding: 10px 14px; border: 1px solid #e2e7ee; border-radius: 12px; background: white; display: flex; align-items: center; justify-content: space-between; gap: 16px 28px; flex-wrap: wrap; color: #526079; font-size: 0.76rem; }}
    .visual-key, .overtake-summary, .overtake {{ display: flex; align-items: center; gap: 7px 13px; flex-wrap: wrap; }}
    .key-item {{ display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }}
    .sample {{ display: inline-block; width: 18px; height: 8px; position: relative; flex: none; }}
    .sample.history::before, .sample.forecast::before {{ content: ""; position: absolute; left: 0; right: 0; top: 4px; border-top: 2px solid #64748b; }}
    .sample.forecast::before {{ border-top-style: dashed; }}
    .sample.range {{ border-radius: 3px; background: rgba(37, 99, 235, 0.12); }}
    .sample.anchor::before {{ content: ""; position: absolute; width: 6px; height: 6px; left: 6px; top: 1px; border: 1.5px solid #64748b; border-radius: 50%; background: white; }}
    .sample.crossing::before {{ content: ""; position: absolute; width: 7px; height: 7px; left: 6px; top: 1px; background: #e59f00; transform: rotate(45deg); }}
    .guide-label {{ color: #172033; font-weight: 650; }}
    .overtake {{ gap: 6px; padding-left: 12px; border-left: 1px solid #e2e7ee; }}
    .overtake strong {{ color: #172033; }}
    .overtake-date {{ color: #344258; }}
    .probability {{ padding: 2px 6px; border-radius: 999px; color: #8a5c00; background: #fff5d6; font-weight: 650; }}
    .range {{ color: #64748b; }}
    #chart {{ min-height: 560px; border: 1px solid #e2e7ee; border-radius: 16px; overflow: hidden; background: #fbfcfe; box-shadow: 0 12px 35px rgba(30, 50, 80, 0.06); }}
    #chart > div, #chart .plot-container, #chart .svg-container {{ width: 100% !important; height: 100% !important; }}
    details {{ margin: 10px 8px 0; color: #64748b; font-size: 0.74rem; }}
    summary {{ cursor: pointer; width: fit-content; color: #52647e; }}
    .diagnostics {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 9px; }}
    .diagnostic {{ display: flex; flex-direction: column; gap: 3px; padding: 9px 11px; border: 1px solid #e2e7ee; border-radius: 9px; background: white; }}
    .diagnostic strong {{ color: #172033; }}
    .variance-note {{ margin: 8px 0 0; }}
    footer {{ display: flex; justify-content: space-between; gap: 24px; padding: 12px 8px 0; color: #7a879b; font-size: 0.72rem; line-height: 1.5; }}
    footer p {{ margin: 0; max-width: 90ch; }}
    a {{ color: #52647e; text-decoration-thickness: 1px; text-underline-offset: 2px; }}
    @media (max-width: 800px) {{
      main {{ padding: 22px 12px 14px; grid-template-rows: auto auto minmax(500px, 1fr) auto auto; }}
      header, footer {{ align-items: start; flex-direction: column; gap: 8px; }}
      .guide {{ align-items: flex-start; flex-direction: column; }}
      .overtake-summary {{ align-items: flex-start; flex-direction: column; }}
      .overtake {{ padding: 0; border: 0; }}
      .diagnostics {{ grid-template-columns: 1fr; }}
      #chart {{ min-height: 500px; border-radius: 12px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Conda, Pixi &amp; Mamba</h1>
        <p class="subtitle">GitHub star history · six-month ensemble forecasts · probable overtake dates</p>
      </div>
      <div class="updated">Updated {escape(latest)} UTC</div>
    </header>
    <section class="guide" aria-label="How to read the chart">
      <div class="visual-key">
        <span class="key-item"><i class="sample history"></i>History</span>
        <span class="key-item"><i class="sample forecast"></i>Forecast</span>
        <span class="key-item"><i class="sample range"></i>95% range</span>
        <span class="key-item"><i class="sample anchor"></i>Observed count</span>
        <span class="key-item"><i class="sample crossing"></i>Median overtake</span>
      </div>
      <div class="overtake-summary">
        <span class="guide-label">Likely overtakes</span>
        {crossover_summary}
      </div>
    </section>
    <section id="chart" aria-label="Interactive star history chart">{chart}</section>
    <details>
      <summary>Forecast methodology and diagnostics</summary>
      <div class="diagnostics">{model_details}</div>
    </details>
    <footer>
      <p>History is estimated between archived aggregate observations. Overtake ranges are conditional on crossing within six months. Forecasts are exploratory, not promises.</p>
      <p><a href="https://github.com/baszalmstra/pixi-star-history/blob/main/data/star_history.csv">CSV</a> · <a href="https://github.com/baszalmstra/pixi-star-history">Source</a></p>
    </footer>
  </main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return f"Built {output} from {len(rows):,} CSV rows."
