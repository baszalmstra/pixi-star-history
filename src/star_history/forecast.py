"""Backtested ensemble forecasts for cumulative repository star counts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

CANDIDATES = (
    ("linear-30d", 30, None),
    ("linear-90d", 90, None),
    ("linear-180d", 180, None),
    ("linear-365d", 365, None),
    ("damped-90d", 90, 180.0),
    ("damped-180d", 180, 365.0),
)
BACKTEST_OFFSETS = (30, 60, 90, 180)
BACKTEST_HORIZON = 30


@dataclass(frozen=True)
class Forecast:
    """Point forecast, 95% interval, and model diagnostics."""

    predicted: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    weights: dict[str, float]
    backtest_rmse: float
    candidate_predictions: dict[str, NDArray[np.float64]]


def trend_slope(values: NDArray[np.float64], window: int) -> float:
    """Estimate a recent least-squares daily slope."""
    sample = values[-min(window, len(values)) :]
    if len(sample) < 2:
        return 0.0
    x = np.arange(len(sample), dtype=float)
    slope = float(np.polyfit(x, sample, 1)[0])
    return max(0.0, slope)


def candidate_prediction(
    values: NDArray[np.float64], horizon: int, window: int, damping: float | None
) -> NDArray[np.float64]:
    """Forecast from the latest observation using a linear or damped trend."""
    slope = trend_slope(values, window)
    steps = np.arange(1, horizon + 1, dtype=float)
    growth = slope * steps
    if damping is not None:
        growth = slope * damping * (1 - np.exp(-steps / damping))
    return values[-1] + growth


def _backtest_predictions(
    values: NDArray[np.float64],
) -> tuple[dict[str, list[NDArray[np.float64]]], list[NDArray[np.float64]]]:
    predictions = {name: [] for name, _, _ in CANDIDATES}
    actuals: list[NDArray[np.float64]] = []
    for offset in BACKTEST_OFFSETS:
        cutoff = len(values) - offset
        horizon = min(BACKTEST_HORIZON, offset)
        if cutoff < 31:
            continue
        training = values[:cutoff]
        actual = values[cutoff : cutoff + horizon]
        actuals.append(actual)
        for name, window, damping in CANDIDATES:
            predictions[name].append(candidate_prediction(training, horizon, window, damping))
    return predictions, actuals


def forecast(values: NDArray[np.float64], horizon: int = 183) -> Forecast:
    """Create a monotonic weighted forecast with an empirical 95% interval."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 61:
        raise ValueError("At least 61 daily observations are required")
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")

    backtests, actuals = _backtest_predictions(values)
    model_rmse: dict[str, float] = {}
    for name, predictions in backtests.items():
        residuals = [
            prediction - actual for prediction, actual in zip(predictions, actuals, strict=True)
        ]
        model_rmse[name] = float(
            np.sqrt(np.mean(np.concatenate([residual**2 for residual in residuals])))
        )

    inverse_variance = {name: 1.0 / max(error, 1.0) ** 2 for name, error in model_rmse.items()}
    weight_total = sum(inverse_variance.values())
    weights = {name: weight / weight_total for name, weight in inverse_variance.items()}

    final_models = {
        name: candidate_prediction(values, horizon, window, damping)
        for name, window, damping in CANDIDATES
    }
    predicted = sum(weights[name] * prediction for name, prediction in final_models.items())
    predicted = np.maximum.accumulate(np.maximum(predicted, values[-1]))

    ensemble_residuals: list[NDArray[np.float64]] = []
    for fold, actual in enumerate(actuals):
        fold_prediction = sum(weights[name] * backtests[name][fold] for name in backtests)
        ensemble_residuals.append(fold_prediction - actual)
    backtest_rmse = float(
        np.sqrt(np.mean(np.concatenate([residual**2 for residual in ensemble_residuals])))
    )

    model_matrix = np.vstack(list(final_models.values()))
    model_weights = np.array([weights[name] for name in final_models])[:, None]
    disagreement = np.sqrt(np.sum(model_weights * (model_matrix - predicted[None, :]) ** 2, axis=0))
    steps = np.arange(1, horizon + 1, dtype=float)
    empirical_error = backtest_rmse * np.sqrt(steps / BACKTEST_HORIZON)
    margin = 1.96 * np.sqrt(empirical_error**2 + disagreement**2)
    lower = np.maximum.accumulate(np.maximum(values[-1], predicted - margin))
    upper = np.maximum.accumulate(predicted + margin)

    return Forecast(
        predicted,
        lower,
        upper,
        weights,
        backtest_rmse,
        final_models,
    )
