import numpy as np
import pytest

import forecast_combo.combinations as combos


def test_rmse_weights_raises_on_perfect_forecast():
    """RMSE weighting raises ``ValueError`` when a source has zero loss."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.column_stack([y.copy(), y + 1])

    with pytest.raises(ValueError, match="zero RMSE loss"):
        combos.rmse_weights(X, y, window_size=None, discount_param=1.0)


def test_mse_weights_raises_on_perfect_forecast():
    """MSE weighting raises ``ValueError`` when a source has zero loss."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.column_stack([y.copy(), y + 1])

    with pytest.raises(ValueError, match="zero MSE loss"):
        combos.mse_weights(X, y, window_size=None, discount_param=1.0)


def test_mae_weights_raises_on_perfect_forecast():
    """MAE weighting raises ``ValueError`` when a source has zero loss."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.column_stack([y.copy(), y + 1])

    with pytest.raises(ValueError, match="zero MAE loss"):
        combos.mae_weights(X, y, window_size=None, discount_param=1.0)


def test_huber_weights_raises_on_perfect_forecast():
    """Huber weighting raises ``ValueError`` when a source has zero loss."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.column_stack([y.copy(), y + 1])

    with pytest.raises(ValueError, match="zero Huber loss"):
        combos.huber_weights(X, y, window_size=None)


def test_rmse_weights_does_not_raise_without_perfect_forecast():
    """RMSE weighting accepts non-zero losses from every source."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.column_stack([y + 0.1, y + 0.5])

    weights, _ = combos.rmse_weights(X, y, window_size=None, discount_param=1.0)

    assert np.isclose(np.sum(weights), 1)
