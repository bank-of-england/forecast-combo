"""Focused tests for the private shared validation and window helpers.

These pin down the ``forecast_combo._validation`` helper contract directly,
before it is consumed by the public weighting functions and ``ForecastCombo``.
"""

import numpy as np
import pytest

from forecast_combo._validation import (
    apply_window,
    validate_discount_param,
    validate_forecast_matrix,
    validate_k,
    validate_nonempty_sample,
    validate_window_size,
)


def test_validate_forecast_matrix_accepts_x_only():
    validate_forecast_matrix(np.ones((2, 2)))


def test_validate_forecast_matrix_rejects_non_array_x():
    with pytest.raises(TypeError, match="X must be a numpy array"):
        validate_forecast_matrix([[1.0, 2.0]])


def test_validate_forecast_matrix_rejects_wrong_ndim_x():
    with pytest.raises(ValueError, match="X must be a 2D array"):
        validate_forecast_matrix(np.ones(2))


def test_validate_forecast_matrix_rejects_empty_sources():
    with pytest.raises(ValueError, match="at least one forecast source"):
        validate_forecast_matrix(np.empty((2, 0)))


def test_validate_forecast_matrix_rejects_non_numeric_x():
    with pytest.raises(TypeError, match="X must contain real numeric values"):
        validate_forecast_matrix(np.array([["a", "b"]]))


def test_validate_forecast_matrix_rejects_non_finite_x():
    with pytest.raises(ValueError, match="X must contain only finite values"):
        validate_forecast_matrix(np.array([[1.0, np.nan]]))


def test_validate_forecast_matrix_rejects_non_array_y():
    with pytest.raises(TypeError, match="y must be a numpy array"):
        validate_forecast_matrix(np.ones((2, 1)), [1.0, 2.0])


def test_validate_forecast_matrix_rejects_wrong_ndim_y():
    with pytest.raises(ValueError, match="y must be a 1D array"):
        validate_forecast_matrix(np.ones((2, 1)), np.ones((2, 1)))


def test_validate_forecast_matrix_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="must match length of y"):
        validate_forecast_matrix(np.ones((2, 1)), np.ones(3))


def test_validate_forecast_matrix_rejects_non_finite_y():
    with pytest.raises(ValueError, match="y must contain only finite values"):
        validate_forecast_matrix(np.ones((2, 1)), np.array([1.0, np.nan]))


def test_validate_forecast_matrix_does_not_check_emptiness():
    """Emptiness is a separate, data-dependent concern (validate_nonempty_sample)."""
    validate_forecast_matrix(np.empty((0, 2)), np.empty(0))


def test_validate_nonempty_sample_rejects_empty_target():
    with pytest.raises(ValueError, match="estimation sample is empty"):
        validate_nonempty_sample(np.empty(0))


def test_validate_nonempty_sample_accepts_nonempty_target():
    validate_nonempty_sample(np.ones(1))


@pytest.mark.parametrize(
    "window_size, error",
    [(0, ValueError), (-1, ValueError), (1.5, TypeError), ("1", TypeError)],
)
def test_validate_window_size_rejects_invalid_values(window_size, error):
    with pytest.raises(error):
        validate_window_size(window_size)


def test_validate_window_size_rejects_boolean():
    with pytest.raises(TypeError, match="window_size must be an integer"):
        validate_window_size(True)


def test_validate_window_size_accepts_none_and_positive_int():
    validate_window_size(None)
    validate_window_size(3)


def test_apply_window_with_none_returns_inputs_unchanged():
    X = np.ones((3, 2))
    y = np.ones(3)

    out_X, out_y = apply_window(X, y, None)

    assert out_X is X
    assert out_y is y


def test_apply_window_size_one_selects_final_aligned_row():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([10.0, 20.0, 30.0])

    out_X, out_y = apply_window(X, y, 1)

    assert out_X == pytest.approx(np.array([[5.0, 6.0]]))
    assert out_y == pytest.approx(np.array([30.0]))


def test_apply_window_equal_to_sample_size_is_a_no_op():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([10.0, 20.0, 30.0])

    out_X, out_y = apply_window(X, y, 3)

    assert out_X == pytest.approx(X)
    assert out_y == pytest.approx(y)


def test_apply_window_larger_than_sample_is_a_no_op():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([10.0, 20.0, 30.0])

    out_X, out_y = apply_window(X, y, 10)

    assert out_X == pytest.approx(X)
    assert out_y == pytest.approx(y)


def test_apply_window_rejects_non_positive_window_size():
    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        apply_window(np.ones((2, 1)), np.ones(2), 0)


def test_validate_discount_param_rejects_boolean():
    with pytest.raises(TypeError, match="discount_param must be a number"):
        validate_discount_param(True)


@pytest.mark.parametrize("discount_param", [-0.5, 0.0, 1.5, np.inf, np.nan])
def test_validate_discount_param_rejects_out_of_range_values(discount_param):
    with pytest.raises(ValueError, match="discount_param"):
        validate_discount_param(discount_param)


def test_validate_discount_param_accepts_valid_range():
    validate_discount_param(1.0)
    validate_discount_param(0.5)


def test_validate_k_rejects_boolean():
    with pytest.raises(TypeError, match="k must be an integer"):
        validate_k(True)


def test_validate_k_rejects_negative():
    with pytest.raises(ValueError, match="k must be >= 0"):
        validate_k(-1)


def test_validate_k_accepts_zero_and_positive():
    validate_k(0)
    validate_k(5)
