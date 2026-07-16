from datetime import date

import numpy as np
import pytest

from star_history.crossover import analyze_crossover, simulate_paths
from star_history.forecast import forecast


def test_analyze_crossover_reports_probability_and_spread() -> None:
    challenger = np.array(
        [
            [9, 11, 12],
            [9, 10, 10],
            [11, 12, 13],
        ],
        dtype=float,
    )
    incumbent = np.full((3, 3), 10.0)

    result = analyze_crossover(
        "challenger/repo",
        8,
        challenger,
        "incumbent/repo",
        10,
        incumbent,
        date(2026, 1, 1),
    )

    assert result.challenger == "challenger/repo"
    assert result.incumbent == "incumbent/repo"
    assert result.probability == pytest.approx(2 / 3)
    assert result.p05_date == date(2026, 1, 2)
    assert result.p95_date == date(2026, 1, 3)
    assert result.standard_deviation_days == pytest.approx(2**-0.5)
    assert result.variance_days == pytest.approx(0.5)


def test_simulated_paths_are_reproducible_and_monotonic() -> None:
    values = 100 + np.arange(500, dtype=float) * 2
    result = forecast(values, horizon=60)

    first = simulate_paths(result, values[-1], simulations=100, seed=42)
    second = simulate_paths(result, values[-1], simulations=100, seed=42)

    assert np.array_equal(first, second)
    assert np.all(np.diff(first, axis=1) >= 0)
    assert np.all(first >= values[-1])
