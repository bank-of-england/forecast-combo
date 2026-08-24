"""Validation and trailing-window helpers."""

import numpy as np


def validate_forecast_matrix(X: np.ndarray, y: np.ndarray | None = None) -> None:
    """Validate forecast-array shape, dtype, and finite values."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be a numpy array, got {type(X).__name__}")
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got {X.ndim} dimensions")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one forecast source")
    if not np.issubdtype(X.dtype, np.number) or np.issubdtype(X.dtype, np.complexfloating):
        raise TypeError("X must contain real numeric values")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")

    if y is None:
        return
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be a numpy array, got {type(y).__name__}")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, got {y.ndim} dimensions")
    if not np.issubdtype(y.dtype, np.number) or np.issubdtype(y.dtype, np.complexfloating):
        raise TypeError("y must contain real numeric values")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Number of rows in X ({X.shape[0]}) must match length of y ({y.shape[0]})")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values")


def validate_nonempty_sample(y: np.ndarray) -> None:
    """Raise if the estimation target ``y`` is empty."""
    if y.size == 0:
        raise ValueError("The estimation sample is empty; at least one observation is required")


def validate_window_size(window_size: int | None) -> None:
    """Validate an optional rolling-window length."""
    if window_size is None:
        return
    if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)):
        raise TypeError(f"window_size must be an integer or None, got {type(window_size).__name__}")
    if window_size < 1:
        raise ValueError(f"window_size must be a positive integer or None, got {window_size}")


def apply_window(X: np.ndarray, y: np.ndarray, window_size: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Validate and apply an optional trailing estimation window."""
    validate_window_size(window_size)
    if window_size is None or len(y) <= window_size:
        return X, y
    return X[-window_size:], y[-window_size:]


def validate_discount_param(discount_param: float) -> None:
    """Validate the exponential discount factor."""
    if isinstance(discount_param, bool) or not isinstance(discount_param, (int, float, np.number)):
        raise TypeError(f"discount_param must be a number, got {type(discount_param).__name__}")
    if not np.isfinite(discount_param):
        raise ValueError(f"discount_param must be finite, got {discount_param}")
    if not 0 < discount_param <= 1:
        raise ValueError(f"discount_param must be in the interval (0, 1], got {discount_param}")


def validate_k(k: int) -> None:
    """Validate the outturn maturity parameter ``k``."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
