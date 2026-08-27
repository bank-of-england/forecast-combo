"""Matplotlib plots of forecast combination weights."""

from importlib import import_module

_PLOT_MODULES = {
    "heatmap_by_vintage": "heatmap",
    "heatmap_by_horizon": "heatmap",
    "line_plot_by_vintage": "lineplot",
    "line_plot_by_horizon": "lineplot",
    "bar_plot_by_vintage": "barplot",
    "bar_plot_by_horizon": "barplot",
}


def __getattr__(name):
    """Load the module that defines a requested plotting function."""
    if name in _PLOT_MODULES:
        module = import_module(f".{_PLOT_MODULES[name]}", __name__)
        function = getattr(module, name)
        globals()[name] = function
        return function
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "heatmap_by_vintage",
    "heatmap_by_horizon",
    "line_plot_by_vintage",
    "line_plot_by_horizon",
    "bar_plot_by_vintage",
    "bar_plot_by_horizon",
]
