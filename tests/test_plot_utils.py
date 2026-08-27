# Select a non-interactive backend before importing pyplot.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecast_combo.visualisations._plot_utils import render_facet_grid  # noqa: E402


def _facet_df(*facet_tuples, dims=("model", "method")):
    return pd.DataFrame(list(facet_tuples), columns=list(dims))


def _default_figsize(n_rows, n_cols):
    return (n_cols, n_rows)


def test_single_facet_produces_1x1_grid_and_renders_once():
    df = _facet_df(("model_a", "rmse"))
    calls = []

    fig, axes = render_facet_grid(
        df,
        ["model", "method"],
        lambda ax, subset, combo, row, col, n_rows, n_cols: calls.append((subset, combo, row, col, n_rows, n_cols)),
        figsize=_default_figsize,
    )
    try:
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.ndim == 2
        assert axes.shape == (1, 1)
        assert len(calls) == 1
        subset, combo, row, col, n_rows, n_cols = calls[0]
        assert combo == ("model_a", "rmse")
        assert (row, col, n_rows, n_cols) == (0, 0, 1, 1)
        assert len(subset) == 1
    finally:
        plt.close(fig)


def test_two_facets_layout_is_a_single_column_of_two_rows():
    """The grid formula prioritises rows over columns, so 2 facets stack vertically."""
    df = _facet_df(("model_a", "rmse"), ("model_b", "rmse"))

    fig, axes = render_facet_grid(df, ["model", "method"], lambda *args: None, figsize=_default_figsize)
    try:
        assert axes.ndim == 2
        assert axes.shape == (2, 1)
    finally:
        plt.close(fig)


def test_multi_facet_layout_matches_grid_formula_with_unused_axes():
    dims = ("model", "method")
    facets = [(f"model_{i}", "rmse") for i in range(3)]
    df = _facet_df(*facets, dims=dims)

    fig, axes = render_facet_grid(df, list(dims), lambda *args: None, figsize=_default_figsize)
    try:
        n_plots = len(facets)
        expected_cols = int(np.ceil(np.sqrt(n_plots / 2)))
        expected_rows = int(np.ceil(n_plots / expected_cols))
        assert axes.shape == (expected_rows, expected_cols)
        # Three plots in a 2x2 grid leave one hidden axis.
        hidden = [ax for ax in axes.flat if not ax.get_visible()]
        assert len(hidden) == 1
    finally:
        plt.close(fig)


def test_facets_are_visited_in_deterministic_sorted_order():
    df = _facet_df(("model_b", "x"), ("model_a", "y"), ("model_a", "x"))
    calls = []

    fig, axes = render_facet_grid(
        df,
        ["model", "method"],
        lambda ax, subset, combo, row, col, n_rows, n_cols: calls.append(combo),
        figsize=_default_figsize,
    )
    try:
        assert calls == [("model_a", "x"), ("model_a", "y"), ("model_b", "x")]
    finally:
        plt.close(fig)


def test_render_facet_receives_only_matching_rows():
    df = pd.DataFrame(
        {
            "model": ["model_a", "model_a", "model_b"],
            "method": ["rmse", "rmse", "rmse"],
            "value": [1, 2, 3],
        }
    )
    subsets_by_combo = {}

    def render(ax, subset, combo, row, col, n_rows, n_cols):
        subsets_by_combo[combo] = subset["value"].tolist()

    fig, axes = render_facet_grid(df, ["model", "method"], render, figsize=_default_figsize)
    try:
        assert subsets_by_combo[("model_a", "rmse")] == [1, 2]
        assert subsets_by_combo[("model_b", "rmse")] == [3]
    finally:
        plt.close(fig)


def test_default_hide_unused_axes_sets_visibility_false():
    df = _facet_df(*[(f"model_{i}", "rmse") for i in range(3)])

    fig, axes = render_facet_grid(df, ["model", "method"], lambda *args: None, figsize=_default_figsize)
    try:
        used = axes.flat[:3]
        unused = list(axes.flat)[3:]
        assert all(ax.get_visible() for ax in used)
        assert all(not ax.get_visible() for ax in unused)
    finally:
        plt.close(fig)


def test_custom_hide_unused_axes_callback_is_used_instead_of_default():
    df = _facet_df(*[(f"model_{i}", "rmse") for i in range(3)])
    turned_off = []

    fig, axes = render_facet_grid(
        df,
        ["model", "method"],
        lambda *args: None,
        figsize=_default_figsize,
        hide_unused_axes=lambda ax: turned_off.append(ax),
    )
    try:
        assert len(turned_off) == 1
        # The custom callback leaves every axis visible.
        assert all(ax.get_visible() for ax in axes.flat)
    finally:
        plt.close(fig)


def test_figsize_callback_receives_grid_dimensions():
    df = _facet_df(*[(f"model_{i}", "rmse") for i in range(3)])
    seen = []

    def figsize(n_rows, n_cols):
        seen.append((n_rows, n_cols))
        return (2 * n_cols, 3 * n_rows)

    fig, axes = render_facet_grid(df, ["model", "method"], lambda *args: None, figsize=figsize)
    try:
        n_rows, n_cols = seen[0]
        assert axes.shape == (n_rows, n_cols)
        assert tuple(fig.get_size_inches()) == (2 * n_cols, 3 * n_rows)
    finally:
        plt.close(fig)


def test_constrained_layout_option_is_applied_to_the_figure():
    df = _facet_df(("model_a", "rmse"))

    fig_true, _ = render_facet_grid(
        df, ["model", "method"], lambda *args: None, figsize=_default_figsize, constrained_layout=True
    )
    fig_false, _ = render_facet_grid(
        df, ["model", "method"], lambda *args: None, figsize=_default_figsize, constrained_layout=False
    )
    try:
        assert fig_true.get_constrained_layout() is True
        assert fig_false.get_constrained_layout() is False
    finally:
        plt.close(fig_true)
        plt.close(fig_false)


def test_returns_figure_and_2d_ndarray_of_axes():
    df = _facet_df(("model_a", "rmse"), ("model_b", "rmse"))

    fig, axes = render_facet_grid(df, ["model", "method"], lambda *args: None, figsize=_default_figsize)
    try:
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.ndim == 2
        assert all(isinstance(ax, plt.Axes) for ax in axes.flat)
    finally:
        plt.close(fig)
