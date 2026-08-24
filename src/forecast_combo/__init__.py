"""Forecast combination toolkit: combine forecasts from multiple sources."""

from . import combinations, visualisations
from .forecast_combo import (
    SUPPORTED_METHODS,
    ComboSpec,
    ForecastCombo,
    get_weights,
    validate_spec_graph,
)
from .utils import create_period_filter

_PLOT_FUNCTIONS = frozenset(
    {
        "heatmap_by_vintage",
        "heatmap_by_horizon",
        "line_plot_by_vintage",
        "line_plot_by_horizon",
        "bar_plot_by_vintage",
        "bar_plot_by_horizon",
    }
)


def __getattr__(name):
    """Load plotting functions only when they are requested."""
    if name in _PLOT_FUNCTIONS:
        function = getattr(visualisations, name)
        globals()[name] = function
        return function
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ForecastCombo",
    "ComboSpec",
    "get_weights",
    "validate_spec_graph",
    "SUPPORTED_METHODS",
    "create_period_filter",
    "heatmap_by_vintage",
    "heatmap_by_horizon",
    "line_plot_by_vintage",
    "line_plot_by_horizon",
    "bar_plot_by_vintage",
    "bar_plot_by_horizon",
]
