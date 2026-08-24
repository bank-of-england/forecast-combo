import numpy as np


def test_constrained_least_squares():
    """Test that constrained least squares weights are non-negative and sum to one."""
    # Create synthetic data
    np.random.seed(0)
    size = 1000
    true_weights = np.array([0.5, 0.3, 0.2])

    X = np.random.rand(size, 3)  # 100 time periods, 3 sources
    y = X @ true_weights + np.random.normal(0, 0.1, size=size)  # True weights are [0.5, 0.3, 0.2]

    # Calculate constrained least squares weights
    import forecast_combo.combinations as combos

    weights = combos.constrained_least_squares(X, y)

    # Assert that weights are non-negative and sum to one
    assert np.all(weights >= 0), "Weights should be non-negative"
    assert np.isclose(np.sum(weights), 1), "Weights should sum to one"
    assert np.isclose(weights, true_weights, atol=0.01).all(), "Weights should be close to true weights"


def test_constrained_least_squares_when_wrong_dgp():
    """Test that constrained least squares weights are non-negative and sum to one."""
    # Create synthetic data
    np.random.seed(0)
    size = 100
    true_weights = np.array([-1.5, 1.3, 0.2])

    X = np.random.rand(size, 3)  # 100 time periods, 3 sources
    y = X @ true_weights + np.random.normal(0, 0.1, size=size)  # True weights are [0.5, 0.3, 0.2]

    # Calculate constrained least squares weights
    import forecast_combo.combinations as combos

    weights = combos.constrained_least_squares(X, y)

    # Assert that weights are non-negative and sum to one
    assert np.all(weights >= 0), "Weights should be non-negative"
    assert np.isclose(np.sum(weights), 1), "Weights should sum to one"
