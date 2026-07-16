import numpy as np
import pytest

from star_history.forecast import forecast


def test_forecast_is_monotonic_and_weighted() -> None:
    rng = np.random.default_rng(7)
    daily_growth = np.maximum(0, rng.normal(3, 1, 500))
    values = 100 + np.cumsum(daily_growth)

    result = forecast(values, horizon=90)

    assert len(result.predicted) == 90
    assert np.all(np.diff(result.predicted) >= 0)
    assert np.all(result.lower >= values[-1])
    assert np.all(result.lower <= result.predicted)
    assert np.all(result.upper >= result.predicted)
    assert sum(result.weights.values()) == pytest.approx(1)
    assert result.backtest_rmse >= 0


def test_forecast_rejects_short_history() -> None:
    with pytest.raises(ValueError, match="daily observations"):
        forecast(np.arange(30, dtype=float))
