"""Probabilistic repository overtake dates derived from forecast uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from .forecast import Forecast

DEFAULT_SIMULATIONS = 12_000
SEEDS = {
    "conda/conda": 7_473,
    "prefix-dev/pixi": 7_434,
    "mamba-org/mamba": 8_069,
}


@dataclass(frozen=True)
class Crossover:
    """Distribution of the first date a challenger exceeds an incumbent."""

    challenger: str
    incumbent: str
    probability: float
    median_date: date | None
    p05_date: date | None
    p95_date: date | None
    standard_deviation_days: float | None
    variance_days: float | None
    median_stars: int | None


def simulate_paths(
    result: Forecast,
    current_stars: float,
    *,
    simulations: int = DEFAULT_SIMULATIONS,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Sample monotonic paths from model choice and empirical trend error."""
    if simulations < 1:
        raise ValueError("Simulation count must be positive")
    names = tuple(result.candidate_predictions)
    probabilities = np.round(np.array([result.weights[name] for name in names]), decimals=10)
    probabilities /= probabilities.sum()
    model_paths = np.round(
        np.stack([result.candidate_predictions[name] for name in names]), decimals=8
    )
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(names), size=simulations, p=probabilities)

    daily_error = round(result.backtest_rmse / np.sqrt(30.0), 8)
    innovations = rng.normal(0.0, daily_error, size=(simulations, model_paths.shape[1]))
    paths = model_paths[selected] + np.cumsum(innovations, axis=1)
    return np.maximum.accumulate(np.maximum(paths, current_stars), axis=1)


def analyze_crossover(
    repository_a: str,
    current_a: float,
    paths_a: NDArray[np.float64],
    repository_b: str,
    current_b: float,
    paths_b: NDArray[np.float64],
    start_day: date,
) -> Crossover:
    """Summarize the first crossing distribution for one repository pair."""
    if paths_a.shape != paths_b.shape:
        raise ValueError("Paired path matrices must have the same shape")
    if current_a == current_b:
        a_is_challenger = paths_a[:, -1].mean() < paths_b[:, -1].mean()
    else:
        a_is_challenger = current_a < current_b
    if a_is_challenger:
        challenger, incumbent = repository_a, repository_b
        challenger_paths, incumbent_paths = paths_a, paths_b
    else:
        challenger, incumbent = repository_b, repository_a
        challenger_paths, incumbent_paths = paths_b, paths_a

    crossings = challenger_paths > incumbent_paths
    crossed = crossings.any(axis=1)
    probability = float(crossed.mean())
    if not crossed.any():
        return Crossover(challenger, incumbent, probability, None, None, None, None, None, None)

    first_offsets = np.argmax(crossings[crossed], axis=1) + 1
    first_indices = first_offsets - 1
    crossing_paths = challenger_paths[crossed]
    crossing_stars = crossing_paths[np.arange(len(first_indices)), first_indices]

    def percentile_day(percentile: float) -> date:
        offset = int(np.quantile(first_offsets, percentile, method="nearest"))
        return start_day + timedelta(days=offset)

    standard_deviation = float(np.std(first_offsets, ddof=1)) if len(first_offsets) > 1 else 0.0
    return Crossover(
        challenger=challenger,
        incumbent=incumbent,
        probability=probability,
        median_date=percentile_day(0.50),
        p05_date=percentile_day(0.05),
        p95_date=percentile_day(0.95),
        standard_deviation_days=standard_deviation,
        variance_days=standard_deviation**2,
        median_stars=round(float(np.median(crossing_stars))),
    )


def pairwise_crossovers(
    forecasts: dict[str, tuple[float, Forecast]],
    start_day: date,
    *,
    simulations: int = DEFAULT_SIMULATIONS,
) -> list[Crossover]:
    """Analyze every unordered repository pair over the forecast horizon."""
    paths = {
        repository: simulate_paths(
            result,
            current,
            simulations=simulations,
            seed=SEEDS.get(repository, 0),
        )
        for repository, (current, result) in forecasts.items()
    }
    return [
        analyze_crossover(
            repository_a,
            forecasts[repository_a][0],
            paths[repository_a],
            repository_b,
            forecasts[repository_b][0],
            paths[repository_b],
            start_day,
        )
        for repository_a, repository_b in combinations(forecasts, 2)
    ]
