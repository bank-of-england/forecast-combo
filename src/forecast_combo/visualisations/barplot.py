"""Bar charts of combination weights across vintages and horizons."""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from forecast_combo.visualisations._plot_utils import (
    facet_dimensions,
    facet_title,
    filter_plot_data,
    prepare_combo_data,
    render_facet_grid,
)


def _use_stacked_bars(pivot, y_axis: str) -> bool:
    """Return whether the values in ``pivot`` can be stacked."""
    if y_axis != "model":
        return False
    if pivot.isna().any().any():
        return False
    values = pivot.to_numpy()
    return not bool((values < 0).any()) and bool(np.allclose(values.sum(axis=1), 1))


def _draw_bars(ax: Axes, pivot: pd.DataFrame, stacked: bool) -> tuple[np.ndarray, bool]:
    """Draw stacked or grouped bars for ``pivot`` and report negative values.

    Parameters
    ----------
    ax : Axes
        Axes on which to draw the bars.
    pivot : pd.DataFrame
        Values to draw, indexed by group with one column per segment.
    stacked : bool
        Whether to stack the segment bars.

    Returns
    -------
    tuple[np.ndarray, bool]
        The x positions of the groups, and whether any drawn value (or stacked
        total) is negative.
    """
    segments = sorted(pivot.columns)
    x = np.arange(len(pivot))
    has_negative = False

    if stacked:
        bottoms = np.zeros(len(pivot))
        for segment in segments:
            values = pivot[segment].to_numpy()
            has_negative = has_negative or bool((values < 0).any())
            ax.bar(x, values, bottom=bottoms, label=segment)
            bottoms += values
        has_negative = has_negative or bool((bottoms < 0).any())
    else:
        width = 0.8 / max(len(segments), 1)
        offsets = (np.arange(len(segments)) - (len(segments) - 1) / 2) * width
        for index, segment in enumerate(segments):
            values = pivot[segment].to_numpy()
            has_negative = has_negative or bool((values < 0).any())
            positions = x + offsets[index]
            ax.bar(positions, values, width=width, label=segment)
            for position in positions[np.isnan(values)]:
                ax.text(position, 0, "NA", ha="center", va="bottom")

    return x, has_negative


def _set_weight_ylim(ax, has_negative: bool) -> None:
    """Pin the axis at zero only when nothing negative needs to be shown."""
    if has_negative:
        ax.axhline(0, linewidth=0.8)
    else:
        ax.set_ylim(0, None)


def bar_plot_by_vintage(
    weights_df: pd.DataFrame,
    y_axis: str = "model",
    model: str | list[str] | None = None,
    method: str | list[str] | None = None,
    variable: str | list[str] | None = None,
    horizon: int | str | list[int | str] | None = None,
    combo_label: str | list[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot average weights by vintage date as bars.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Combination weights with the required plotting columns.
    y_axis : str
        Dimension represented by bar segments: ``"model"``, ``"method"``,
        or ``"variable"``.
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
        # Pivot to (y_axis x vintage_date); average over horizons if not already faceted
        pivot = df_subset.groupby(["vintage_date", y_axis])["weight"].mean().unstack(y_axis).sort_index()

        x, has_negative = _draw_bars(ax, pivot, _use_stacked_bars(pivot, y_axis))

        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(d)[:10] for d in pivot.index],
            rotation=45,
            ha="right",
            fontsize=7,
        )

        title = facet_title(facet_dims, facet_combo, df_subset)
        ax.set_title(title, fontsize=11)

        if row == n_rows - 1:
            ax.set_xlabel("Vintage date", fontsize=9)
        if col == 0:
            ax.set_ylabel("Average weight", fontsize=9)

        ax.legend(fontsize=8)
        _set_weight_ylim(ax, has_negative)

    fig, axes_2d = render_facet_grid(
        df,
        facet_dims,
        render_facet,
        figsize=lambda n_rows, n_cols: (6 * n_cols, 4 * n_rows),
        constrained_layout=True,
    )

    fig.suptitle("Weights Across Time", fontsize=14)
    return fig, axes_2d


def bar_plot_by_horizon(
    weights_df: pd.DataFrame,
    y_axis: str = "model",
    model: str | list[str] | None = None,
    method: str | list[str] | None = None,
    variable: str | list[str] | None = None,
    combo_label: str | list[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot average weights by forecast horizon as bars.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Combination weights with the required plotting columns.
    y_axis : str
        Dimension represented by bar segments: ``"model"``, ``"method"``,
        or ``"variable"``.
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
        # Average over vintage dates, then pivot to (y_axis x horizon)
        pivot = df_subset.groupby(["horizon", y_axis])["weight"].mean().unstack(y_axis).sort_index()

        x, has_negative = _draw_bars(ax, pivot, _use_stacked_bars(pivot, y_axis))

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index.tolist(), fontsize=8)

        title = facet_title(facet_dims, facet_combo, df_subset)
        ax.set_title(title, fontsize=11)

        if row == n_rows - 1:
            ax.set_xlabel("Forecast horizon", fontsize=9)
        if col == 0:
            ax.set_ylabel("Average weight", fontsize=9)

        ax.legend(fontsize=8)
        _set_weight_ylim(ax, has_negative)

    fig, axes_2d = render_facet_grid(
        df,
        facet_dims,
        render_facet,
        figsize=lambda n_rows, n_cols: (6 * n_cols, 4 * n_rows),
        constrained_layout=True,
    )

    fig.suptitle("Average Weights by Horizon", fontsize=14)
    return fig, axes_2d
