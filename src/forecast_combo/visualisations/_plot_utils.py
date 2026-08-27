"""Shared data-handling helpers for weight visualisations."""

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from forecast_combo._optional import require_optional_dependency

matplotlib = require_optional_dependency("matplotlib", "plots", "Plotting")

from matplotlib.axes import Axes  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

matplotlib.use("Agg")

_REQUIRED_WEIGHT_COLUMNS = {
    "model",
    "method",
    "variable",
    "horizon",
    "vintage_date",
    "weight",
}


def validate_plot_data(df: pd.DataFrame) -> None:
    """Check the columns and values required by weight plots."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"weights_df must be a pandas DataFrame, got {type(df).__name__}")
    missing = sorted(_REQUIRED_WEIGHT_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"weights_df is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("weights_df contains no rows to plot")
    for column in ("model", "method", "variable"):
        valid_strings = df[column].map(lambda value: isinstance(value, str) and bool(value.strip()))
        if df[column].isna().any() or not valid_strings.all():
            raise ValueError(f"weights_df column '{column}' must contain only non-empty strings")
    if "combo_label" in df.columns:
        labels = df["combo_label"].dropna()
        if not labels.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("weights_df column 'combo_label' must contain only non-empty strings or missing values")
    if not pd.api.types.is_numeric_dtype(df["horizon"]):
        raise TypeError("weights_df column 'horizon' must be numeric")
    if pd.api.types.is_bool_dtype(df["horizon"]):
        raise TypeError("weights_df column 'horizon' must not contain booleans")
    horizons = df["horizon"].to_numpy(dtype=float)
    if not np.all(np.isfinite(horizons)) or not np.all(horizons == np.floor(horizons)):
        raise ValueError("weights_df column 'horizon' must contain only finite integers")
    if not pd.api.types.is_numeric_dtype(df["weight"]):
        raise TypeError("weights_df column 'weight' must be numeric")
    weights = df["weight"].to_numpy(dtype=float)
    if np.isinf(weights).any():
        raise ValueError("weights_df column 'weight' must not contain infinite values")
    if not np.isfinite(weights).any():
        raise ValueError("weights_df column 'weight' must contain at least one finite value")
    try:
        vintage_dates = pd.to_datetime(df["vintage_date"])
    except (TypeError, ValueError) as exc:
        raise ValueError("weights_df column 'vintage_date' must contain valid dates") from exc
    if vintage_dates.isna().any():
        raise ValueError("weights_df column 'vintage_date' must not contain missing dates")


def require_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    """Raise if filtering leaves no rows to plot."""
    if df.empty:
        raise ValueError("No weight data remains after applying the selected filters")
    return df


def prepare_combo_data(df: pd.DataFrame, combo_label=None) -> pd.DataFrame:
    """Copy ``df`` and optionally filter it to selected ``combo_label`` values."""
    validate_plot_data(df)
    df = df.copy()
    if combo_label is not None:
        if "combo_label" not in df.columns:
            raise ValueError("weights_df is missing required column 'combo_label' for the requested filter")
        labels = _normalise_filter(combo_label, "combo_label", (str,))
        df = df[df["combo_label"].isin(labels)]
    return df


def filter_plot_data(df: pd.DataFrame, *, model=None, method=None, variable=None, horizon=None) -> pd.DataFrame:
    """Apply validated plot filters and require at least one matching row."""
    for column, value in (("model", model), ("method", method), ("variable", variable)):
        if value is not None:
            selected = _normalise_filter(value, column, (str,))
            df = df[df[column].isin(selected)]
    if horizon is not None:
        selected = _normalise_horizon_filter(horizon)
        df = df[df["horizon"].isin(selected)]
    return require_plot_data(df)


def validate_frequency_column(df: pd.DataFrame) -> None:
    """Check the frequency column used to format vintage labels."""
    if "frequency" not in df.columns:
        raise ValueError("weights_df is missing required column: frequency")
    if (
        df["frequency"].isna().any()
        or not df["frequency"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    ):
        raise ValueError("weights_df column 'frequency' must contain only non-empty strings")
    if df["frequency"].nunique() != 1:
        raise ValueError("weights_df must contain exactly one frequency when plotting by vintage")


def _normalise_filter(value, name: str, scalar_types: tuple[type, ...]) -> list:
    """Normalise one scalar or iterable filter and validate its members."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must not be a bool")
    if isinstance(value, scalar_types):
        values = [value]
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            expected = "string" if scalar_types == (str,) else "string or integer"
            raise TypeError(f"{name} must be a {expected} or an iterable of those values")
        values = list(value)
    if not values:
        raise ValueError(f"{name} filter must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, scalar_types) for item in values):
        expected = "strings" if scalar_types == (str,) else "strings or integers"
        raise TypeError(f"{name} filter must contain only {expected}")
    if str in scalar_types and any(isinstance(item, str) and not item.strip() for item in values):
        raise ValueError(f"{name} filter values must not be empty")
    return values


def _normalise_horizon_filter(value) -> list[int]:
    """Normalise integer-like horizon values, including numeric strings."""
    scalar_types = (str, int, float, np.integer, np.floating)
    values = _normalise_filter(value, "horizon", scalar_types)
    horizons = []
    for item in values:
        try:
            numeric = float(item)
        except ValueError as exc:
            raise ValueError(f"horizon filter value {item!r} is not numeric") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"horizon filter value {item!r} must be a finite integer")
        horizons.append(int(numeric))
    return horizons


def facet_dimensions(df: pd.DataFrame, y_axis: str) -> list[str]:
    """Return facet dimensions other than ``y_axis``."""
    dimensions = sorted({"model", "method", "variable"} - {y_axis})
    if "combo_label" in df.columns and df["combo_label"].nunique() > 1:
        dimensions = ["combo_label", *dimensions]
    return dimensions


def facet_title(facet_dims: list[str], facet_values: tuple, df_subset: pd.DataFrame) -> str:
    """Format a facet title from its dimension values."""
    return " | ".join(f"{dimension}: {value}" for dimension, value in zip(facet_dims, facet_values))


def _facet_grid_shape(n_plots: int) -> tuple[int, int]:
    """Return ``(n_rows, n_cols)`` for ``n_plots`` facets, prioritising rows."""
    n_cols = int(np.ceil(np.sqrt(n_plots / 2)))
    n_rows = int(np.ceil(n_plots / n_cols))
    return n_rows, n_cols


def _normalise_axes_grid(axes, n_plots: int, n_rows: int, n_cols: int) -> np.ndarray:
    """Reshape whatever ``plt.subplots`` returned into a consistent 2D array."""
    if n_plots == 1:
        return np.array([[axes]])
    if n_rows == 1:
        return axes.reshape(1, -1)
    if n_cols == 1:
        return axes.reshape(-1, 1)
    return axes


def render_facet_grid(
    df: pd.DataFrame,
    facet_dims: list[str],
    render_facet: Callable[[Axes, pd.DataFrame, tuple, int, int, int, int], None],
    *,
    figsize: Callable[[int, int], tuple[float, float]],
    constrained_layout: bool = False,
    hide_unused_axes: Callable[[Axes], None] = lambda ax: ax.set_visible(False),
) -> tuple[Figure, np.ndarray]:
    """Build and render a grid of filtered facets.

    The function visits facets in sorted order and passes each facet's axes,
    matching rows, values, and grid position to ``render_facet``.
    """
    facet_combinations = df[facet_dims].drop_duplicates().sort_values(facet_dims)
    facet_values_list = [tuple(row) for row in facet_combinations.values]

    n_plots = len(facet_values_list)
    n_rows, n_cols = _facet_grid_shape(n_plots)

    fig = Figure(figsize=figsize(n_rows, n_cols), constrained_layout=constrained_layout)
    axes = fig.subplots(n_rows, n_cols)
    axes_2d = _normalise_axes_grid(axes, n_plots, n_rows, n_cols)

    for idx, facet_combo in enumerate(facet_values_list):
        row, col = divmod(idx, n_cols)
        df_subset = df.copy()
        for dim, value in zip(facet_dims, facet_combo):
            df_subset = df_subset[df_subset[dim] == value]
        render_facet(axes_2d[row, col], df_subset, facet_combo, row, col, n_rows, n_cols)

    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        hide_unused_axes(axes_2d[row, col])

    return fig, axes_2d


def add_colourbar(fig: Figure, mappable: ScalarMappable, axes: list[Axes], label: str):
    """Add a Shiny-compatible colourbar to a faceted figure."""
    colourbar = fig.colorbar(mappable, ax=axes, label=label)
    colourbar.ax.set_subplotspec(axes[0].get_subplotspec())
    return colourbar
