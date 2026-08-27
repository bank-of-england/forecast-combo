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
