"""Server handlers for the Across Vintage tab (line, heatmap, and bar)."""

from typing import Any

import pandas as pd
from shiny import render

from forecast_combo.visualisations.barplot import bar_plot_by_vintage
from forecast_combo.visualisations.heatmap import heatmap_by_vintage
from forecast_combo.visualisations.lineplot import line_plot_by_vintage

_PLOT_FN = {"line": line_plot_by_vintage, "heatmap": heatmap_by_vintage, "bar": bar_plot_by_vintage}


def vintage_tab(input: Any, output: Any, session: Any, combo_df: pd.DataFrame) -> None:
    """Register the render handlers for the "Across Vintage" tab.

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
        kw = {
            "model": list(input.model()) or None,
            "method": list(input.method()) or None,
            "variable": list(input.variable()) or None,
        }
        selected_horizons = list(input.horizon())
        kw["horizon"] = [int(h) for h in selected_horizons] if selected_horizons else None
        return kw

    @output
    @render.plot
    def plot_vintage_model():
        fig, _ = _PLOT_FN[input.plot_type()](combo_df, y_axis="model", **_kwargs())
        return fig

    @output
    @render.plot
    def plot_vintage_method():
        fig, _ = _PLOT_FN[input.plot_type()](combo_df, y_axis="method", **_kwargs())
        return fig

    @output
    @render.plot
    def plot_vintage_variable():
        fig, _ = _PLOT_FN[input.plot_type()](combo_df, y_axis="variable", **_kwargs())
        return fig
