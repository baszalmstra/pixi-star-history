from datetime import date

import plotly.graph_objects as go

from star_history.crossover import Crossover
from star_history.site import add_crossovers


def test_add_crossovers_draws_interval_and_variance_hover() -> None:
    figure = go.Figure()
    crossover = Crossover(
        challenger="prefix-dev/pixi",
        incumbent="mamba-org/mamba",
        probability=0.81,
        median_date=date(2026, 10, 31),
        p05_date=date(2026, 10, 12),
        p95_date=date(2027, 1, 9),
        standard_deviation_days=29.1,
        variance_days=845.1,
        median_stars=8169,
    )

    add_crossovers(figure, [crossover])

    assert len(figure.layout.shapes) == 1
    assert figure.layout.shapes[0].x0 == "2026-10-12"
    assert len(figure.data) == 1
    assert figure.data[0].text == ("Pixi → Mamba",)
    assert "Date variance" in figure.data[0].hovertemplate
