"""Test reporting uncertainty in estimated weights."""

import numpy as np
import pytest

from forecast_combo.forecast_combo import get_weights

UNCERTAINTY_NOT_ESTIMATED = ["average", "constrained_least_squares", "huber"]
UNCERTAINTY_ESTIMATED = ["least_squares", "rmse", "mse", "mae"]


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X @ np.array([0.2, 0.3, 0.5]) + rng.normal(scale=0.1, size=60)
    return X, y


@pytest.mark.parametrize("method", UNCERTAINTY_NOT_ESTIMATED + UNCERTAINTY_ESTIMATED)
def test_std_error_has_one_entry_per_source(sample, method):
    X, y = sample
    weights, std_error = get_weights(X, y, method=method, window_size=None, discount_param=1.0)

    assert weights.shape == (X.shape[1],)
    assert std_error.shape == (X.shape[1],)


@pytest.mark.parametrize("method", UNCERTAINTY_NOT_ESTIMATED)
def test_unsupported_uncertainty_is_nan_not_zero(sample, method):
    """Methods without uncertainty estimates return ``NaN``, not zero."""
    X, y = sample
    _, std_error = get_weights(X, y, method=method, window_size=None, discount_param=1.0)

    assert np.all(np.isnan(std_error))
