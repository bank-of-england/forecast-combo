"""Line plots of combination weights across vintages and horizons."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from forecast_combo.visualisations._plot_utils import (
    facet_dimensions,
    facet_title,
    filter_plot_data,
    prepare_combo_data,
    render_facet_grid,
)


def line_plot_by_vintage(
    weights_df: pd.DataFrame,
    y_axis: str = "model",
    model: str | list[str] | None = None,
    method: str | list[str] | None = None,
    variable: str | list[str] | None = None,
    horizon: int | str | list[int | str] | None = None,
    combo_label: str | list[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot average weights by vintage date.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Combination weights with the required plotting columns.
    y_axis : str
        Dimension represented by lines: ``"model"``, ``"method"``, or
        ``"variable"``.
    model : str | list[str] | None
        Models used to filter the data.
    method : str | list[str] | None
        Methods used to filter the data.
    variable : str | list[str] | None
        Variables used to filter the data.
    horizon : int | str | list[int | str] | None
        Horizons to plot as separate facets. If omitted, weights are averaged
        over horizons.
    combo_label : str | list[str] | None
        Combination labels used to filter the data.

    Returns
    -------
    tuple[Figure, np.ndarray]
        Figure and two-dimensional array of axes.

    Raises
    ------
    ValueError
        If ``y_axis`` is not a supported dimension.
    """
    if y_axis not in ("model", "method", "variable"):
        raise ValueError("y_axis must be 'model', 'method', or 'variable'")

    # Apply filters
    df = prepare_combo_data(weights_df, combo_label)
    df = filter_plot_data(df, model=model, method=method, variable=variable, horizon=horizon)

    # Determine faceting dimensions (all dims except y_axis, plus horizon if specified)
    facet_dims = facet_dimensions(df, y_axis)
    if horizon is not None:
        facet_dims = ["horizon"] + facet_dims

    def render_facet(ax, df_subset, facet_combo, row, col, n_rows, n_cols):
        # Average over any remaining horizons (or show exact values if already faceted by horizon)
        for val in sorted(df_subset[y_axis].unique()):
            d = (
                df_subset[df_subset[y_axis] == val]
                .groupby("vintage_date")["weight"]
                .mean()
                .reset_index()
                .sort_values("vintage_date")
            )
            ax.plot(
                d["vintage_date"],
                d["weight"],
                label=val,
            )

        title = facet_title(facet_dims, facet_combo, df_subset)
        ax.set_title(title, fontsize=11)

        if row == n_rows - 1:
            ax.set_xlabel("Vintage date", fontsize=9)
            ax.tick_params(axis="x", rotation=45)
        if col == 0:
            ax.set_ylabel("Average weight", fontsize=9)

        ax.legend(fontsize=8)

    fig, axes_2d = render_facet_grid(
        df,
        facet_dims,
        render_facet,
        figsize=lambda n_rows, n_cols: (6 * n_cols, 4 * n_rows),
        constrained_layout=True,
    )

    fig.suptitle("Weights Across Time", fontsize=14)
    return fig, axes_2d


def line_plot_by_horizon(
    weights_df: pd.DataFrame,
    y_axis: str = "model",
    model: str | list[str] | None = None,
    method: str | list[str] | None = None,
    variable: str | list[str] | None = None,
    combo_label: str | list[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot average weights by forecast horizon.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Combination weights with the required plotting columns.
    y_axis : str
        Dimension represented by lines: ``"model"``, ``"method"``, or
        ``"variable"``.
    model : str | list[str] | None
        Models used to filter the data.
    method : str | list[str] | None
        Methods used to filter the data.
    variable : str | list[str] | None
        Variables used to filter the data.
    combo_label : str | list[str] | None
        Combination labels used to filter the data.

    Returns
    -------
    tuple[Figure, np.ndarray]
        Figure and two-dimensional array of axes.

    Raises
    ------
    ValueError
        If ``y_axis`` is not a supported dimension.
    """
    if y_axis not in ("model", "method", "variable"):
        raise ValueError("y_axis must be 'model', 'method', or 'variable'")

    # Apply filters
    df = prepare_combo_data(weights_df, combo_label)
    df = filter_plot_data(df, model=model, method=method, variable=variable)

    # Determine faceting dimensions (all dims except y_axis)
    facet_dims = facet_dimensions(df, y_axis)

    def render_facet(ax, df_subset, facet_combo, row, col, n_rows, n_cols):
        # Average over vintage dates
        for val in sorted(df_subset[y_axis].unique()):
            d = df_subset[df_subset[y_axis] == val].groupby("horizon")["weight"].mean()
            ax.plot(d.index, d.values, label=val)

        title = facet_title(facet_dims, facet_combo, df_subset)
        ax.set_title(title, fontsize=11)

        if row == n_rows - 1:
            ax.set_xlabel("Forecast horizon", fontsize=9)
        if col == 0:
            ax.set_ylabel("Average weight", fontsize=9)

        ax.legend(fontsize=8)

    fig, axes_2d = render_facet_grid(
        df,
        facet_dims,
        render_facet,
        figsize=lambda n_rows, n_cols: (6 * n_cols, 4 * n_rows),
        constrained_layout=True,
    )

    fig.suptitle("Average Weights by Horizon", fontsize=14)
    return fig, axes_2d
