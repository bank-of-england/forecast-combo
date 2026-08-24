"""Forecast combination weighting methods."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .._validation import (
    apply_window,
    validate_discount_param,
    validate_forecast_matrix,
    validate_nonempty_sample,
)

jax.config.update("jax_enable_x64", True)


def _validate_forecast_matrix(X: np.ndarray, y: np.ndarray | None = None) -> None:
    """Validate arrays used by the weighting functions."""
    validate_forecast_matrix(X, y)
    if y is not None:
        validate_nonempty_sample(y)


def average(X: np.ndarray) -> np.ndarray:
    """Return equal weights for the sources in ``X``.

    Parameters
    ----------
    X : np.ndarray
        Forecast matrix. Only the number of sources is used.

    Returns
    -------
    np.ndarray
        Weight ``1 / n_sources`` for each source.
    """
    _validate_forecast_matrix(X)
    weights = np.ones(X.shape[1]) / X.shape[1]  # Equal weights for each source
    return weights


def _check_nonzero_loss(loss: np.ndarray, method_name: str) -> None:
    """Validate losses before inverse-loss weighting."""
    if not np.all(np.isfinite(loss)):
        raise ValueError(f"One or more sources have non-finite {method_name} loss due to numerical overflow.")
    if np.any(loss == 0):
        raise ValueError(
            f"One or more sources have exactly zero {method_name} loss. "
            "This is unexpected with real-valued forecast data and suggests "
            "a data or setup issue (e.g. the outturn was included as a source), "
            "or that no combination is needed for this source."
        )


def _inverse_loss_weights(loss: np.ndarray, method_name: str) -> np.ndarray:
    """Return normalised weights from a loss array."""
    _check_nonzero_loss(loss, method_name)
    scaled_inverse = loss.min() / loss
    weights = scaled_inverse / scaled_inverse.sum()
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{method_name} weight normalisation produced non-finite values")
    return weights


def least_squares(X: np.ndarray, y: np.ndarray, window_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return least-squares weights and standard errors.

    Parameters
    ----------
    X : np.ndarray
        Forecast matrix.
    y : np.ndarray
        Target values.
    window_size : int | None
        Number of trailing observations used for estimation. ``None`` uses
        all observations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Estimated weights and their standard errors. Standard errors are
        ``NaN`` when the design is rank deficient or has too few observations.

    Raises
    ------
    ValueError
        If the estimation sample is empty.
    """
    _validate_forecast_matrix(X, y)
    X, y = apply_window(X, y, window_size)

    n, k = X.shape

    if n == 0:
        raise ValueError("Cannot fit least squares: the estimation sample is empty.")

    # Solve for weights using least squares (minimum-norm solution if rank deficient)
    weights, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)

    if rank < k:
        warnings.warn(
            f"Least-squares design is rank deficient (rank {rank} < {k} sources); "
            "weights are the minimum-norm solution and standard errors are not identified."
        )
        return weights, np.full(k, np.nan)

    if n <= k:
        warnings.warn(
            f"Least-squares standard errors require more observations than sources "
            f"(have n={n}, k={k}); returning NaN standard errors."
        )
        return weights, np.full(k, np.nan)

    # Residuals
    residuals = y - X @ weights
    sigma2 = (residuals @ residuals) / (n - k)

    # Reconstruct the inverse Gram matrix from the SVD without forming X.T @ X.
    _, singular_values, vh = np.linalg.svd(X, full_matrices=False)
    scaled_v = vh.T / singular_values
    covariance = scaled_v @ scaled_v.T
    std_errors = np.sqrt(np.diag(sigma2 * covariance))

    return weights, std_errors


def constrained_least_squares(X: np.ndarray, y: np.ndarray, window_size: int | None = None) -> np.ndarray:
    """Return least-squares weights constrained to the unit simplex.

    Parameters
    ----------
    X : np.ndarray
        Forecast matrix.
    y : np.ndarray
        Target values.
    window_size : int | None
        Number of trailing observations used for estimation. ``None`` uses
        all observations.

    Returns
    -------
    np.ndarray
        Estimated weights. Each weight is non-negative and the weights sum to
        one.

    Raises
    ------
    ValueError
        If ``y`` has zero variance over the estimation sample.
    RuntimeError
        If the constrained optimisation fails or returns invalid weights.
    """
    _validate_forecast_matrix(X, y)

    # Apply rolling window if specified
    X, y = apply_window(X, y, window_size)

    # Scale in two stages so variance calculation cannot overflow for large,
    # finite targets while preserving the same least-squares objective.
    magnitude = np.max(np.abs(y))
    if magnitude == 0:
        raise ValueError("y has zero variance; cannot fit constrained least squares.")
    X = X / magnitude
    y = y / magnitude
    scaling_factor = np.std(y)
    if not np.isfinite(scaling_factor) or scaling_factor == 0:
        raise ValueError("y has zero or non-finite variance; cannot fit constrained least squares.")
    X = X / scaling_factor
    y = y / scaling_factor

    # Convert to JAX arrays
    X_jax = jnp.array(X)
    y_jax = jnp.array(y)
    n_sources = X.shape[1]

    def objective(w):
        return jnp.sum((y_jax - X_jax @ w) ** 2)

    # Compute the gradient with JAX
    grad_fn = jax.grad(objective)

    # Wrappers for SciPy (convert JAX arrays to numpy)
    def jac(w):
        return np.array(grad_fn(w))

    # Initial guess: equal weights
    w0 = np.ones(n_sources) / n_sources

    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Bounds: weights >= 0
    bounds = [(0, None) for _ in range(n_sources)]

    # Solve with analytical derivatives
    result = minimize(objective, w0, method="SLSQP", jac=jac, bounds=bounds, constraints=constraints)

    if not result.success:
        raise RuntimeError(f"Constrained least-squares optimisation did not converge: {result.message}")

    weights = np.asarray(result.x, dtype=float)
    if not np.all(np.isfinite(weights)):
        raise RuntimeError("Constrained least-squares optimisation returned non-finite weights")
    if np.any(weights < -1e-8) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise RuntimeError("Constrained least-squares optimisation returned weights outside the simplex")

    if np.allclose(weights, w0):
        warnings.warn("The constrained least-squares optimisation returned the initial guess.")

    return weights


def rmse_weights(
    X: np.ndarray, y: np.ndarray, window_size: int | None, discount_param: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return inverse-RMSE weights and standard errors.

    Parameters
    ----------
    X : np.ndarray
        Forecast matrix.
    y : np.ndarray
        Target values.
    window_size : int | None
        Number of trailing observations used for estimation.
    discount_param : float
        Exponential discount factor in ``(0, 1]``. Defaults to ``1.0``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Estimated weights and their standard errors.
    """

    _validate_forecast_matrix(X, y)
    validate_discount_param(discount_param)

    # Apply rolling window
    X, y = apply_window(X, y, window_size)

    T = len(y)

    discount = discount_param ** np.arange(T - 1, -1, -1)
    discount_array = np.repeat(discount.reshape(-1, 1), X.shape[1], axis=1)

    errors = y.reshape(-1, 1) - X
    N = errors.shape[0]
    sq_errors = errors**2

    # RMSE per model
    rmse = np.sqrt(np.mean(discount_array * sq_errors, axis=0))
    weights = _inverse_loss_weights(rmse, "RMSE")
    if N < 2:
        return weights, np.full(X.shape[1], np.nan)

    # SE of RMSE — delta method: SE(RMSE) = std(e²) / (2√N · RMSE)
    rmse_se = np.std(discount_array * sq_errors, axis=0, ddof=1) / (2 * np.sqrt(N) * rmse)
    var_rmse = rmse_se**2

    # Standard errors of weights via Delta method
    w_se = delta_method(rmse, var_rmse)

    return weights, w_se


def mse_weights(
    X: np.ndarray, y: np.ndarray, window_size: int | None, discount_param: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return inverse-MSE weights and standard errors.

    Parameters
    ----------
    X : np.ndarray
        Forecast matrix.
    y : np.ndarray
        Target values.
    window_size : int | None
        Number of trailing observations used for estimation.
    discount_param : float
        Exponential discount factor in ``(0, 1]``. Defaults to ``1.0``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Estimated weights and their standard errors.
    """

    _validate_forecast_matrix(X, y)
    validate_discount_param(discount_param)

    # Apply rolling window
    X, y = apply_window(X, y, window_size)

    T = len(y)

    discount = discount_param ** np.arange(T - 1, -1, -1)
    discount_array = np.repeat(discount.reshape(-1, 1), X.shape[1], axis=1)

    errors = y.reshape(-1, 1) - X
    N = errors.shape[0]
    sq_errors = errors**2

    # Discounted MSE per model
    mse = np.mean(discount_array * sq_errors, axis=0)
    weights = _inverse_loss_weights(mse, "MSE")
    if N < 2:
        return weights, np.full(X.shape[1], np.nan)

    # SE of MSE: std(e²) / √N
    mse_se = np.std(discount_array * sq_errors, axis=0, ddof=1) / np.sqrt(N)
    var_mse = mse_se**2

    # Standard errors of weights via Delta method
    w_se = delta_method(mse, var_mse)

    return weights, w_se


def mae_weights(
    X: np.ndarray, y: np.ndarray, window_size: int | None, discount_param: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return inverse-MAE weights and standard errors.

    Parameters
    ----------
    X : np.ndarray
        Forecast matrix.
    y : np.ndarray
        Target values.
    window_size : int | None
        Number of trailing observations used for estimation.
    discount_param : float
        Exponential discount factor in ``(0, 1]``. Defaults to ``1.0``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Estimated weights and their standard errors.
    """
    _validate_forecast_matrix(X, y)
    validate_discount_param(discount_param)

    # Apply rolling window
    X, y = apply_window(X, y, window_size)

    T = len(y)
    discount = discount_param ** np.arange(T - 1, -1, -1)
    discount_array = np.repeat(discount.reshape(-1, 1), X.shape[1], axis=1)

    # Absolute errors

    abs_errors = np.abs(y.reshape(-1, 1) - X)
    N = abs_errors.shape[0]

    # Discounted MAE per model
    dmae = np.mean(discount_array * abs_errors, axis=0)
    weights = _inverse_loss_weights(dmae, "MAE")
    if N < 2:
        return weights, np.full(X.shape[1], np.nan)

    mae_se = np.std(discount_array * abs_errors, axis=0, ddof=1) / np.sqrt(N)
    var_mae = mae_se**2

    # Standard errors of weights via Delta method
    w_se = delta_method(dmae, var_mae)

    return weights, w_se


def huber_weights(X: np.ndarray, y: np.ndarray, window_size: int | None) -> np.ndarray:
    """Return inverse-Huber-loss weights.

    Parameters
    ----------
    X : np.ndarray
        Forecast matrix.
    y : np.ndarray
        Target values.
    window_size : int | None
        Number of trailing observations used for estimation.

    Returns
    -------
    np.ndarray
        Estimated weights.

    Raises
    ------
    ValueError
        If fewer than two observations are available.
    """
    _validate_forecast_matrix(X, y)

    # Apply rolling window
    X, y = apply_window(X, y, window_size)

    if len(y) < 2:
        raise ValueError("Huber weights require at least two observations")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        errors = y.reshape(-1, 1) - X
        abs_err = np.abs(errors)

        # Robust scale estimate (standard deviation of residuals)
        sigma = errors.std(axis=0, ddof=1)
        delta = 1.345 * sigma

        # Huber loss
        quad = 0.5 * errors**2
        lin = delta * (abs_err - 0.5 * delta)
        huber = np.where(abs_err <= delta, quad, lin).mean(axis=0)
    weights = _inverse_loss_weights(huber, "Huber")

    return weights


def delta_method(rmse: np.ndarray, var_rmse: np.ndarray) -> np.ndarray:
    """Return standard errors for inverse-RMSE weights.

    Parameters
    ----------
    rmse : np.ndarray
        RMSE values for each model.
    var_rmse : np.ndarray
        Variance of RMSE estimates (SE^2).

    Returns
    -------
    np.ndarray
        Standard errors of the normalised inverse-RMSE weights.

    Raises
    ------
    TypeError
        If either input is not a real numeric NumPy array.
    ValueError
        If the inputs have invalid dimensions, shapes, or values.
    """
    if not isinstance(rmse, np.ndarray):
        raise TypeError(f"rmse must be a numpy array, got {type(rmse).__name__}")
    if not isinstance(var_rmse, np.ndarray):
        raise TypeError(f"var_rmse must be a numpy array, got {type(var_rmse).__name__}")
    if rmse.ndim != 1 or var_rmse.ndim != 1:
        raise ValueError("rmse and var_rmse must be 1D arrays")
    if rmse.size == 0:
        raise ValueError("rmse and var_rmse must not be empty")
    if rmse.shape != var_rmse.shape:
        raise ValueError("rmse and var_rmse must have the same shape")
    if not np.issubdtype(rmse.dtype, np.number) or np.issubdtype(rmse.dtype, np.complexfloating):
        raise TypeError("rmse must contain real numeric values")
    if not np.issubdtype(var_rmse.dtype, np.number) or np.issubdtype(var_rmse.dtype, np.complexfloating):
        raise TypeError("var_rmse must contain real numeric values")
    if not np.all(np.isfinite(rmse)) or np.any(rmse <= 0):
        raise ValueError("rmse must contain only finite positive values")
    if not np.all(np.isfinite(var_rmse)) or np.any(var_rmse < 0):
        raise ValueError("var_rmse must contain only finite non-negative values")

    weights = _inverse_loss_weights(rmse, "inverse-loss")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative_se = np.sqrt(var_rmse) / rmse
    if not np.all(np.isfinite(relative_se)):
        raise ValueError("Delta-method inputs produce non-finite relative standard errors")

    m = len(rmse)
    w_se = np.zeros(m)
    for j in range(m):
        contributions = weights[j] * weights * relative_se
        contributions[j] = weights[j] * (1 - weights[j]) * relative_se[j]
        w_se[j] = np.linalg.norm(contributions)

    if not np.all(np.isfinite(w_se)):
        raise ValueError("Delta-method calculation produced non-finite standard errors")

    return w_se
