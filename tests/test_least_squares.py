import numpy as np
import pytest

from forecast_combo.combinations.static_combinations import least_squares


def test_least_squares_full_rank_returns_finite_standard_errors():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([0.2, 0.3, 0.5]) + rng.normal(scale=0.1, size=50)

    weights, std_errors = least_squares(X, y)

    assert np.all(np.isfinite(weights))
    assert np.all(np.isfinite(std_errors))
    assert np.all(std_errors > 0)


def test_least_squares_n_equals_k_returns_nan_standard_errors():
    X = np.eye(3)
    y = np.array([1.0, 2.0, 3.0])

    with pytest.warns(UserWarning, match="more observations than sources"):
        weights, std_errors = least_squares(X, y)

    assert np.allclose(weights, y)
    assert np.all(np.isnan(std_errors))


def test_least_squares_rank_deficient_does_not_raise():
    X = np.column_stack([np.arange(10.0), 2 * np.arange(10.0)])
    y = np.arange(10.0)

    with pytest.warns(UserWarning, match="rank deficient"):
        weights, std_errors = least_squares(X, y)

    assert np.all(np.isfinite(weights))
    assert np.allclose(X @ weights, y)
    assert np.all(np.isnan(std_errors))


def test_least_squares_window_applied_before_sample_size_checks():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([0.2, 0.3, 0.5]) + rng.normal(scale=0.1, size=50)

    # 3 observations for 3 sources once the window is applied
    with pytest.warns(UserWarning, match="more observations than sources"):
        _, std_errors = least_squares(X, y, window_size=3)

    assert np.all(np.isnan(std_errors))


def test_least_squares_empty_sample_raises():
    with pytest.raises(ValueError, match="estimation sample is empty"):
        least_squares(np.empty((0, 2)), np.empty(0))
