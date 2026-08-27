"""Server handlers for the Across Horizon tab (line, heatmap, and bar)."""

from typing import Any

import pandas as pd
from shiny import render

from forecast_combo.visualisations.barplot import bar_plot_by_horizon
from forecast_combo.visualisations.heatmap import heatmap_by_horizon
from forecast_combo.visualisations.lineplot import line_plot_by_horizon

_PLOT_FN = {"line": line_plot_by_horizon, "heatmap": heatmap_by_horizon, "bar": bar_plot_by_horizon}


def horizon_tab(input: Any, output: Any, session: Any, combo_df: pd.DataFrame) -> None:
    """Register plot renderers for the "Across Horizon" tab.

    Parameters
    ----------
    input : Any
        The Shiny input object.
    output : Any
        The Shiny output object.
    session : Any
        The Shiny session object.
    combo_df : pd.DataFrame
        Combination weights to plot.
    """

    def _kwargs():
        """Get the selected plot filters."""
        return {
            "model": list(input.model()) or None,
            "method": list(input.method()) or None,
            "variable": list(input.variable()) or None,
        }

    @output
    @render.plot
    def plot_horizon_model():
        fig, _ = _PLOT_FN[input.plot_type()](combo_df, y_axis="model", **_kwargs())
        return fig

    @output
    @render.plot
    def plot_horizon_method():
        fig, _ = _PLOT_FN[input.plot_type()](combo_df, y_axis="method", **_kwargs())
        return fig

    @output
    @render.plot
    def plot_horizon_variable():
        fig, _ = _PLOT_FN[input.plot_type()](combo_df, y_axis="variable", **_kwargs())
        return fig
