"""Weighting algorithms used to estimate forecast combination weights."""

from .static_combinations import (
    average,
    constrained_least_squares,
    huber_weights,
    least_squares,
    mae_weights,
    mse_weights,
    rmse_weights,
)

__all__ = [
    "average",
    "least_squares",
    "constrained_least_squares",
    "rmse_weights",
    "mse_weights",
    "mae_weights",
    "huber_weights",
]
