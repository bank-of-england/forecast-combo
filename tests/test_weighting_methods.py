"""Deterministic numerical tests for the fixed-weight combination methods.

Each case uses losses computed by hand, so the expected weights are exact
closed-form values rather than properties of a random sample.
"""

import numpy as np
import pytest

import forecast_combo.combinations as combos
from forecast_combo.combinations.static_combinations import average, least_squares


@pytest.fixture
def known_losses():
    """Errors of magnitude 1 for source 0 and 2 for source 1.

    This produces MAE = (1, 2), MSE = (1, 4), and RMSE = (1, 2). Both errors
    fall inside the Huber threshold, so Huber loss = (0.5, 2).
    """
    y = np.zeros(4)
    X = np.column_stack([np.array([1.0, -1.0, 1.0, -1.0]), np.array([2.0, -2.0, 2.0, -2.0])])
    return X, y


def test_average_returns_equal_weights():
    weights = average(np.zeros((5, 4)))
    assert weights == pytest.approx(np.full(4, 0.25))
    assert weights.sum() == pytest.approx(1.0)


def test_rmse_weights_are_inverse_rmse(known_losses):
    X, y = known_losses
    weights, std_errors = combos.rmse_weights(X, y, window_size=None, discount_param=1.0)

    # Inverse RMSE factors are (1, 1/2); normalisation gives these weights.
    assert weights == pytest.approx([2 / 3, 1 / 3])
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(std_errors))


def test_mse_weights_are_inverse_mse(known_losses):
    X, y = known_losses
    weights, std_errors = combos.mse_weights(X, y, window_size=None, discount_param=1.0)

    # Inverse MSE factors are (1, 1/4); normalisation gives these weights.
    assert weights == pytest.approx([0.8, 0.2])
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(std_errors))


def test_mae_weights_are_inverse_mae(known_losses):
    X, y = known_losses
    weights, std_errors = combos.mae_weights(X, y, window_size=None, discount_param=1.0)

    # Inverse MAE factors are (1, 1/2); normalisation gives these weights.
    assert weights == pytest.approx([2 / 3, 1 / 3])
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(std_errors))


def test_mse_penalises_large_errors_more_than_mae(known_losses):
    """The squared-error method must down-weight the worse source more strongly."""
    X, y = known_losses
    mse_weights, _ = combos.mse_weights(X, y, window_size=None, discount_param=1.0)
    mae_weights, _ = combos.mae_weights(X, y, window_size=None, discount_param=1.0)

    assert mse_weights[1] < mae_weights[1]


def test_huber_weights_are_inverse_huber_loss(known_losses):
    X, y = known_losses
    weights = combos.huber_weights(X, y, window_size=None)

    # Both errors fall below delta = 1.345 * sigma, so the loss is quadratic:
    # 0.5 * e^2 = (0.5, 2), with inverse factors (2, 0.5).
    assert weights == pytest.approx([0.8, 0.2])
    assert weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "weight_function",
    [
        lambda X, y, window: combos.rmse_weights(X, y, window_size=window, discount_param=1.0)[0],
        lambda X, y, window: combos.mse_weights(X, y, window_size=window, discount_param=1.0)[0],
        lambda X, y, window: combos.mae_weights(X, y, window_size=window, discount_param=1.0)[0],
        lambda X, y, window: combos.huber_weights(X, y, window_size=window),
    ],
    ids=["rmse", "mse", "mae", "huber"],
)
def test_rolling_window_only_uses_the_final_observations(weight_function):
    """The trailing window excludes older observations from the weights."""
    y = np.zeros(6)
    recent = np.column_stack([np.array([1.0, -1.0, 1.0, -1.0]), np.array([2.0, -2.0, 2.0, -2.0])])
    # Add two early observations that reverse the ranking before the recent block.
    early = np.array([[9.0, 0.5], [-9.0, -0.5]])
    X = np.vstack([early, recent])

    windowed = weight_function(X, y, 4)
    recent_only = weight_function(recent, np.zeros(4), None)

    assert windowed == pytest.approx(recent_only)


@pytest.mark.parametrize(
    "weight_function",
    [
        lambda X, y, window: combos.least_squares(X, y, window)[0],
        lambda X, y, window: combos.constrained_least_squares(X, y, window),
        lambda X, y, window: combos.rmse_weights(X, y, window, discount_param=1.0)[0],
        lambda X, y, window: combos.mse_weights(X, y, window, discount_param=1.0)[0],
        lambda X, y, window: combos.mae_weights(X, y, window, discount_param=1.0)[0],
        lambda X, y, window: combos.huber_weights(X, y, window),
    ],
    ids=["least_squares", "constrained_least_squares", "rmse", "mse", "mae", "huber"],
)
def test_all_weighting_methods_apply_the_trailing_window(weight_function):
    """A positive window must select exactly the final observations."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = np.column_stack(
        [
            np.array([0.0, 1.0, 2.0, 1.0, 3.0]),
            np.array([2.0, 4.0, 1.0, 0.0, 2.0]),
        ]
    )

    windowed = weight_function(X, y, 3)
    trailing = weight_function(X[-3:], y[-3:], None)

    assert windowed == pytest.approx(trailing)


@pytest.mark.parametrize(
    "weight_function",
    [
        lambda X, y, window: combos.least_squares(X, y, window)[0],
        lambda X, y, window: combos.rmse_weights(X, y, window, discount_param=1.0)[0],
        lambda X, y, window: combos.mse_weights(X, y, window, discount_param=1.0)[0],
        lambda X, y, window: combos.mae_weights(X, y, window, discount_param=1.0)[0],
    ],
    ids=["least_squares", "rmse", "mse", "mae"],
)
def test_trailing_window_of_size_one_selects_the_final_aligned_pair(weight_function):
    """window_size=1 must keep the last (X row, y value) pair together."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.column_stack(
        [
            np.array([0.0, 1.0, 2.0, 5.0]),
            np.array([2.0, 4.0, 1.0, 6.0]),
        ]
    )

    windowed = weight_function(X, y, 1)
    final_pair_only = weight_function(X[-1:], y[-1:], None)

    assert windowed == pytest.approx(final_pair_only)


@pytest.mark.parametrize(
    "weight_function",
    [
        lambda X, y, window: combos.least_squares(X, y, window)[0],
        lambda X, y, window: combos.constrained_least_squares(X, y, window),
        lambda X, y, window: combos.rmse_weights(X, y, window, discount_param=1.0)[0],
        lambda X, y, window: combos.mse_weights(X, y, window, discount_param=1.0)[0],
        lambda X, y, window: combos.mae_weights(X, y, window, discount_param=1.0)[0],
        lambda X, y, window: combos.huber_weights(X, y, window),
    ],
    ids=["least_squares", "constrained_least_squares", "rmse", "mse", "mae", "huber"],
)
def test_trailing_window_equal_to_sample_size_is_a_no_op(weight_function):
    """A window covering exactly the whole sample must equal no window at all."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = np.column_stack(
        [
            np.array([0.0, 1.0, 2.0, 1.0, 3.0]),
            np.array([2.0, 4.0, 1.0, 0.0, 2.0]),
        ]
    )

    windowed = weight_function(X, y, len(y))
    unwindowed = weight_function(X, y, None)

    assert windowed == pytest.approx(unwindowed)


@pytest.mark.parametrize(
    "weight_function",
    [
        lambda X, y, window: combos.least_squares(X, y, window)[0],
        lambda X, y, window: combos.constrained_least_squares(X, y, window),
        lambda X, y, window: combos.rmse_weights(X, y, window, discount_param=1.0)[0],
        lambda X, y, window: combos.mse_weights(X, y, window, discount_param=1.0)[0],
        lambda X, y, window: combos.mae_weights(X, y, window, discount_param=1.0)[0],
        lambda X, y, window: combos.huber_weights(X, y, window),
    ],
    ids=["least_squares", "constrained_least_squares", "rmse", "mse", "mae", "huber"],
)
def test_trailing_window_larger_than_sample_is_a_no_op(weight_function):
    """A window larger than the sample uses all observations."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = np.column_stack(
        [
            np.array([0.0, 1.0, 2.0, 1.0, 3.0]),
            np.array([2.0, 4.0, 1.0, 0.0, 2.0]),
        ]
    )

    windowed = weight_function(X, y, len(y) + 10)
    unwindowed = weight_function(X, y, None)

    assert windowed == pytest.approx(unwindowed)


@pytest.mark.parametrize(
    "weight_function",
    [combos.rmse_weights, combos.mse_weights, combos.mae_weights],
    ids=["rmse", "mse", "mae"],
)
def test_discount_param_rejects_boolean(weight_function, known_losses):
    X, y = known_losses

    with pytest.raises(TypeError, match="discount_param must be a number"):
        weight_function(X, y, window_size=None, discount_param=True)


def test_discounting_favours_the_source_with_smaller_recent_errors():
    """With discounting, recent accuracy must dominate older accuracy."""
    y = np.zeros(2)
    # Source 0 performs poorly recently; source 1 has the opposite pattern.
    X = np.column_stack([np.array([1.0, 3.0]), np.array([3.0, 1.0])])

    undiscounted, _ = combos.mse_weights(X, y, window_size=None, discount_param=1.0)
    discounted, _ = combos.mse_weights(X, y, window_size=None, discount_param=0.5)

    assert undiscounted == pytest.approx([0.5, 0.5])
    assert discounted[1] > discounted[0]
    assert discounted.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "weight_function",
    [combos.rmse_weights, combos.mse_weights, combos.mae_weights],
    ids=["rmse", "mse", "mae"],
)
def test_discount_param_defaults_to_one(weight_function, known_losses):
    X, y = known_losses

    default_weights, default_std_errors = weight_function(X, y, window_size=None)
    explicit_weights, explicit_std_errors = weight_function(X, y, window_size=None, discount_param=1.0)

    assert default_weights == pytest.approx(explicit_weights)
    assert default_std_errors == pytest.approx(explicit_std_errors)


@pytest.mark.parametrize(
    "weight_function",
    [combos.rmse_weights, combos.mse_weights, combos.mae_weights],
    ids=["rmse", "mse", "mae"],
)
@pytest.mark.parametrize("discount_param", [-0.5, 0.0, 1.5, np.inf, np.nan])
def test_invalid_discount_param_is_rejected(weight_function, discount_param, known_losses):
    X, y = known_losses

    with pytest.raises(ValueError, match="discount_param"):
        weight_function(X, y, window_size=None, discount_param=discount_param)


@pytest.mark.parametrize(
    "weight_function",
    [combos.rmse_weights, combos.mse_weights, combos.mae_weights],
    ids=["rmse", "mse", "mae"],
)
def test_inverse_loss_standard_errors_are_undefined_for_one_observation(weight_function):
    """One observation can produce weights, but not sampling standard errors."""
    X = np.array([[1.0, 2.0]])
    y = np.array([0.0])

    weights, std_errors = weight_function(X, y, window_size=None, discount_param=1.0)

    assert np.all(np.isfinite(weights))
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.isnan(std_errors))


def test_unconstrained_least_squares_recovers_exact_coefficients():
    """OLS must return the generating coefficients, including negative ones."""
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    true_weights = np.array([2.0, -0.5])
    y = X @ true_weights

    weights, std_errors = least_squares(X, y)

    assert weights == pytest.approx(true_weights)
    # OLS is unconstrained: weights need not be non-negative or sum to one.
    assert weights.sum() == pytest.approx(1.5)
    # The fit is exact, so the residual variance and standard errors are zero.
    assert std_errors == pytest.approx(np.zeros(2))


def test_unconstrained_least_squares_matches_normal_equations():
    """Weights must equal the closed-form solution on a noisy full-rank design."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3))
    y = X @ np.array([0.5, -0.2, 0.9]) + rng.normal(scale=0.1, size=40)

    weights, _ = least_squares(X, y)

    expected = np.linalg.solve(X.T @ X, X.T @ y)
    assert weights == pytest.approx(expected)


def test_least_squares_standard_errors_remain_finite_for_near_collinear_sources():
    """OLS uncertainty uses a stable covariance calculation."""
    rng = np.random.default_rng(23)
    base = rng.normal(size=80)
    X = np.column_stack([base, base * (1 + 1e-8) + rng.normal(scale=1e-10, size=80)])
    y = X @ np.array([0.7, -0.2]) + rng.normal(scale=0.05, size=80)

    _, std_errors = least_squares(X, y)

    assert np.all(np.isfinite(std_errors))
    assert np.all(std_errors > 0)


def test_constrained_least_squares_projects_onto_the_simplex():
    """The constrained solution must stay non-negative and sum to one."""
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    y = X @ np.array([2.0, -1.0])

    weights = combos.constrained_least_squares(X, y)

    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(weights >= -1e-8)
    # The unconstrained optimum puts negative weight on the second source, so
    # the constraint must bind there.
    assert weights[1] == pytest.approx(0.0, abs=1e-6)
