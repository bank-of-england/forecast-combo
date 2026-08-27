"""Heatmaps of combination weights across vintages and horizons."""

import numpy as np
import pandas as pd

from forecast_combo._optional import require_optional_dependency

require_optional_dependency("matplotlib", "plots", "Plotting")

from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from forecast_combo.visualisations._plot_utils import (  # noqa: E402
    add_colourbar,
    facet_dimensions,
    facet_title,
    filter_plot_data,
    prepare_combo_data,
    render_facet_grid,
    validate_frequency_column,
)


def _shared_colour_mappable(weights: pd.Series) -> ScalarMappable:
    """Create a finite colour scale covering all weights in a heatmap."""
    values = weights.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(finite_values.min()), float(finite_values.max())
        if vmin == vmax:
            padding = max(abs(vmin) * 0.05, 0.5)
            vmin -= padding
            vmax += padding

    return ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis")


def _format_date_label(date, frequency):
    """Return a display label for a date and frequency."""
    date = pd.to_datetime(date)
    if frequency == "Q":
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year} Q{quarter}"
    elif frequency == "M":
        return f"{date.year} M{date.month:02d}"
    else:
        return str(date.date())


def heatmap_by_vintage(
    weights_df: pd.DataFrame,
    y_axis: str = "model",
    model: str | list[str] | None = None,
    method: str | list[str] | None = None,
    variable: str | list[str] | None = None,
    horizon: int | str | list[int | str] | None = None,
    combo_label: str | list[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot average weights by vintage date as a heatmap.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Combination weights with the required plotting columns.
    y_axis : str
        Dimension represented on the y-axis: ``"model"``, ``"method"``, or
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
    if y_axis not in ["model", "method", "variable"]:
        raise ValueError("y_axis must be 'model', 'method', or 'variable'")

    # Filter the plotting data.
    df_filtered = prepare_combo_data(weights_df, combo_label)
    df_filtered = filter_plot_data(
        df_filtered,
        model=model,
        method=method,
        variable=variable,
        horizon=horizon,
    )
    validate_frequency_column(df_filtered)

    # Use every dimension except y_axis, and add horizon when requested.
    facet_dims = facet_dimensions(df_filtered, y_axis)
    if horizon is not None:
        facet_dims = ["horizon"] + facet_dims

    colour_mappable = _shared_colour_mappable(df_filtered["weight"])

    def render_facet(ax, df_subset, facet_combo, row, col, n_rows, n_cols):
        df_subset = df_subset.sort_values("vintage_date")

        # Arrange values with y_axis on the rows and vintage_date on the columns.
        pivot_data = df_subset.pivot_table(values="weight", index=y_axis, columns="vintage_date", aggfunc="mean")

        if pivot_data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            title = facet_title(facet_dims, facet_combo, df_subset)
            ax.set_title(title)
            return

        frequency = df_filtered["frequency"].iloc[0]
        pivot_data.columns = [_format_date_label(col, frequency) for col in pivot_data.columns]

        ax.imshow(pivot_data.values, aspect="auto", cmap=colour_mappable.cmap, norm=colour_mappable.norm)
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns, fontsize=8)
        ax.set_yticklabels(pivot_data.index, fontsize=8)

        # Show x-axis labels on the bottom row only.
        if row < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # Show y-axis labels in the leftmost column only.
        if col > 0:
            ax.set_yticklabels([])

        title = facet_title(facet_dims, facet_combo, df_subset)
        ax.set_title(title, fontsize=11)

        # Label the x-axis on the bottom row only.
        if row == n_rows - 1:
            ax.set_xlabel("Vintage date", fontsize=9)

        # Label the y-axis in the leftmost column only.
        if col == 0:
            ax.set_ylabel(y_axis.capitalize(), fontsize=9)

    fig, axes_2d = render_facet_grid(
        df_filtered,
        facet_dims,
        render_facet,
        figsize=lambda n_rows, n_cols: (5 * n_cols, 4 * n_rows),
        hide_unused_axes=lambda ax: ax.axis("off"),
    )

    fig.suptitle("Weights Across Time", fontsize=14)

    visible_axes = [axis for axis in axes_2d.flat if axis.get_visible()]
    add_colourbar(fig, colour_mappable, visible_axes, "Weight")

    return fig, axes_2d


def heatmap_by_horizon(
    weights_df: pd.DataFrame,
    y_axis: str = "model",
    model: str | list[str] | None = None,
    method: str | list[str] | None = None,
    variable: str | list[str] | None = None,
    combo_label: str | list[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot average weights by forecast horizon as a heatmap.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Combination weights with the required plotting columns.
    y_axis : str
        Dimension represented on the y-axis: ``"model"``, ``"method"``, or
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
    if y_axis not in ["model", "method", "variable"]:
        raise ValueError("y_axis must be 'model', 'method', or 'variable'")

    # Filter the plotting data.
    df_filtered = prepare_combo_data(weights_df, combo_label)
    df_filtered = filter_plot_data(df_filtered, model=model, method=method, variable=variable)

    # Use every dimension except y_axis for the facets.
    facet_dims = facet_dimensions(df_filtered, y_axis)

    colour_mappable = _shared_colour_mappable(df_filtered["weight"])

    def render_facet(ax, df_subset, facet_combo, row, col, n_rows, n_cols):
        # Arrange values with y_axis on the rows and horizon on the columns.
        pivot_data = df_subset.pivot_table(values="weight", index=y_axis, columns="horizon", aggfunc="mean")

        if pivot_data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            title = facet_title(facet_dims, facet_combo, df_subset)
            ax.set_title(title)
            return

        ax.imshow(pivot_data.values, aspect="auto", cmap=colour_mappable.cmap, norm=colour_mappable.norm)
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns, fontsize=8)
        ax.set_yticklabels(pivot_data.index, fontsize=8)

        # Show x-axis labels on the bottom row only.
        if row < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # Show y-axis labels in the leftmost column only.
        if col > 0:
            ax.set_yticklabels([])

        title = facet_title(facet_dims, facet_combo, df_subset)
        ax.set_title(title, fontsize=11)

        # Label the x-axis on the bottom row only.
        if row == n_rows - 1:
            ax.set_xlabel("Forecast horizon", fontsize=9)

        # Label the y-axis in the leftmost column only.
        if col == 0:
            ax.set_ylabel(y_axis.capitalize(), fontsize=9)

    fig, axes_2d = render_facet_grid(
        df_filtered,
        facet_dims,
        render_facet,
        # Leave room for the colour bar at the bottom.
        figsize=lambda n_rows, n_cols: (5 * n_cols, 4 * n_rows + 1.5),
        hide_unused_axes=lambda ax: ax.axis("off"),
    )

    fig.suptitle("Average Weights by Horizon", fontsize=14)

    visible_axes = [axis for axis in axes_2d.flat if axis.get_visible()]
    add_colourbar(fig, colour_mappable, visible_axes, "Weight")

    return fig, axes_2d
