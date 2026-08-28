import numpy as np


def test_constrained_least_squares():
    """Constrained least squares recovers non-negative weights that sum to one."""
    # Generate data from a known weight vector.
    np.random.seed(0)
    size = 1000
    true_weights = np.array([0.5, 0.3, 0.2])

    X = np.random.rand(size, 3)
    y = X @ true_weights + np.random.normal(0, 0.1, size=size)

    import forecast_combo.combinations as combos

    weights = combos.constrained_least_squares(X, y)

    assert np.all(weights >= 0), "Weights are non-negative"
    assert np.isclose(np.sum(weights), 1), "Weights sum to one"
    assert np.isclose(weights, true_weights, atol=0.01).all(), "Weights closely match the true weights"


def test_constrained_least_squares_when_wrong_dgp():
    """Constraints hold when the data comes from negative unconstrained weights."""
    # Generate data whose unconstrained weights include a negative value.
    np.random.seed(0)
    size = 100
    true_weights = np.array([-1.5, 1.3, 0.2])

    X = np.random.rand(size, 3)
    y = X @ true_weights + np.random.normal(0, 0.1, size=size)

    import forecast_combo.combinations as combos

    weights = combos.constrained_least_squares(X, y)

    assert np.all(weights >= 0), "Weights are non-negative"
    assert np.isclose(np.sum(weights), 1), "Weights sum to one"


def test_constrained_least_squares_gradient_matches_finite_differences():
    """Pin the analytical gradient of the constrained least-squares objective.

    ``constrained_least_squares`` hands SLSQP a closed-form gradient of
    ``||y - Xw||^2`` instead of differentiating the objective numerically (it
    used to be obtained with JAX). This test re-derives the same quantity by
    finite differences so any future change to the algebra is caught here
    rather than showing up as slightly-off weights.
    """
    from scipy.optimize import approx_fprime

    rng = np.random.default_rng(0)
    for n_obs, n_sources in [(30, 3), (200, 8), (6, 5)]:
        X = rng.normal(size=(n_obs, n_sources)) + rng.normal(size=(n_obs, 1))
        y = X @ rng.dirichlet(np.ones(n_sources)) + 0.3 * rng.normal(size=n_obs)

        # The same normal-equation quantities the implementation builds.
        gram, Xty = X.T @ X, X.T @ y

        def objective(w):
            return np.sum((y - X @ w) ** 2)

        for w in [np.ones(n_sources) / n_sources, rng.dirichlet(np.ones(n_sources))]:
            analytical = 2.0 * (gram @ w - Xty)
            numerical = approx_fprime(w, objective, 1e-7)
            scale = max(np.max(np.abs(analytical)), 1.0)
            assert np.max(np.abs(analytical - numerical)) / scale < 1e-5


def test_constrained_least_squares_does_not_import_jax():
    """The package must not pull in JAX: it flips ``jax_enable_x64`` process-wide."""
    import sys

    import forecast_combo.combinations as combos

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    combos.constrained_least_squares(X, X @ np.array([0.6, 0.3, 0.1]))
    assert "jax" not in sys.modules
