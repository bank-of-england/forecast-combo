# Select a non-interactive backend before importing pyplot.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from shiny.render._coordmap import get_coordmap  # noqa: E402

import forecast_combo as fc  # noqa: E402
from forecast_combo.dashboard.ui import create_ui  # noqa: E402
from forecast_combo.visualisations.barplot import bar_plot_by_horizon, bar_plot_by_vintage  # noqa: E402
from forecast_combo.visualisations.heatmap import heatmap_by_horizon, heatmap_by_vintage  # noqa: E402
from forecast_combo.visualisations.lineplot import line_plot_by_horizon, line_plot_by_vintage  # noqa: E402


def plotted_line_values(axes, label):
    """Collect every plotted (x, y) pair for a given legend label."""
    values = []
    for ax in np.ravel(axes):
        if not ax.get_visible():
            continue
        for line in ax.get_lines():
            if line.get_label() == label:
                values.extend(zip(line.get_xdata(), line.get_ydata(), strict=False))
    return values


def plotted_bar_values(axes):
    """Collect every drawn bar height across all visible axes."""
    heights = []
    for ax in np.ravel(axes):
        if not ax.get_visible():
            continue
        for container in ax.containers:
            heights.extend(patch.get_height() for patch in container.patches)
    return heights


def assert_no_vertical_clipping(axes, values):
    """Every plotted value must lie inside the drawn y-limits."""
    finite = [value for value in values if np.isfinite(value)]
    assert finite, "No finite values were plotted"
    for ax in np.ravel(axes):
        if not ax.get_visible() or not (ax.get_lines() or ax.containers):
            continue
        bottom, top = ax.get_ylim()
        assert bottom <= min(finite) + 1e-9, f"Lowest value {min(finite)} is clipped by y-limit {bottom}"
        assert top >= max(finite) - 1e-9, f"Highest value {max(finite)} is clipped by y-limit {top}"


@pytest.fixture(scope="module")
def combo_df(fer_data):
    """Load and fit a simple combo model for testing."""
    forecast_data = fer_data.copy()
    combo = fc.ForecastCombo(
        forecast_data=forecast_data,
    )
    combo.fit(
        training_start="2024-01-01",
        sources=["bvar unconditional", "compass unconditional"],
        variables=["cpisa"],
        method=["rmse", "average"],
        window_size=10,
    )
    return pd.DataFrame(combo.weights)


def test_line_plot_by_horizon_y_axis_model(combo_df):
    """The model view draws a faceted line plot."""
    fig, axes = line_plot_by_horizon(combo_df, y_axis="model")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_line_plot_by_horizon_y_axis_method(combo_df):
    """The method view draws a faceted line plot."""
    fig, axes = line_plot_by_horizon(combo_df, y_axis="method")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_line_plot_by_horizon_y_axis_variable(combo_df):
    """The variable view draws a faceted line plot."""
    fig, axes = line_plot_by_horizon(combo_df, y_axis="variable")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_line_plot_by_horizon_with_filters(combo_df):
    """The horizon plot applies dimension filters."""
    fig, axes = line_plot_by_horizon(combo_df, y_axis="model", method=["rmse"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_line_plot_by_horizon_does_not_average_across_fit_configurations():
    """Weights from different source sets must remain separate."""
    weights = pd.DataFrame(
        [
            {
                "vintage_date": pd.Timestamp("2025-03-31"),
                "horizon": 1,
                "variable": "gdp",
                "method": "rmse",
                "model": "model_a",
                "weight": 0.2,
                "combo_label": "model_a, model_b",
            },
            {
                "vintage_date": pd.Timestamp("2025-03-31"),
                "horizon": 1,
                "variable": "gdp",
                "method": "rmse",
                "model": "model_b",
                "weight": 0.8,
                "combo_label": "model_a, model_b",
            },
            {
                "vintage_date": pd.Timestamp("2025-03-31"),
                "horizon": 1,
                "variable": "gdp",
                "method": "rmse",
                "model": "model_a",
                "weight": 0.8,
                "combo_label": "model_a, model_c",
            },
            {
                "vintage_date": pd.Timestamp("2025-03-31"),
                "horizon": 1,
                "variable": "gdp",
                "method": "rmse",
                "model": "model_c",
                "weight": 0.2,
                "combo_label": "model_a, model_c",
            },
        ]
    )

    fig, axes = line_plot_by_horizon(weights, y_axis="model")
    try:
        model_a_weights = [
            line.get_ydata()[0]
            for ax in axes.flat
            if ax.get_visible()
            for line in ax.get_lines()
            if line.get_label() == "model_a"
        ]

        assert sorted(model_a_weights) == pytest.approx([0.2, 0.8])
    finally:
        plt.close(fig)


def test_repeated_combo_fits_are_not_merged_in_plot():
    """Plotting accumulated weights must preserve each fitted combination.

    ``model_a`` has weight 1/2 when two sources are averaged and weight 1/3
    when three sources are averaged. The plot preserves those fitted values
    instead of replacing them with their average, 5/12.
    """
    import forecast_evaluation as fe

    date = pd.Timestamp("2025-03-31")
    outturns = pd.DataFrame(
        {
            "date": [date],
            "variable": ["gdp"],
            "frequency": ["Q"],
            "value": [1.0],
            "metric": ["levels"],
        }
    )
    forecasts = pd.DataFrame(
        {
            "date": [date] * 3,
            "vintage_date": [date] * 3,
            "variable": ["gdp"] * 3,
            "frequency": ["Q"] * 3,
            "forecast_horizon": [0] * 3,
            "source": ["model_a", "model_b", "model_c"],
            "value": [0.8, 1.1, 0.9],
            "metric": ["levels"] * 3,
        }
    )
    forecast_data = fe.ForecastData(
        outturns_data=outturns,
        forecasts_data=forecasts,
        outturn_vintages=False,
        compute_levels=False,
        data_check=False,
    )
    combo = fc.ForecastCombo(forecast_data)

    combo.fit(
        sources=["model_a", "model_b"],
        variables=["gdp"],
        method="average",
        metric="levels",
        automatic_labelling=True,
        training_start=date,
        training_end=date,
    )
    combo.fit(
        sources=["model_a", "model_b", "model_c"],
        variables=["gdp"],
        method="average",
        metric="levels",
        automatic_labelling=True,
        training_start=date,
        training_end=date,
    )

    assert set(combo.weights["combo_sources"]) == {
        "model_a, model_b",
        "model_a, model_b, model_c",
    }
    assert combo.weights["combo_label"].nunique() == 2
    fitted_model_a_weights = combo.weights.loc[combo.weights["model"] == "model_a", "weight"]
    assert sorted(fitted_model_a_weights) == pytest.approx([1 / 3, 1 / 2])

    fig, axes = line_plot_by_horizon(combo.weights, y_axis="model")
    try:
        plotted_model_a_weights = [
            line.get_ydata()[0]
            for ax in axes.flat
            if ax.get_visible()
            for line in ax.get_lines()
            if line.get_label() == "model_a"
        ]

        assert sorted(plotted_model_a_weights) == pytest.approx([1 / 3, 1 / 2])
    finally:
        plt.close(fig)


def test_reused_label_for_a_different_configuration_raises():
    """Reusing a label for a different fit configuration raises an error."""
    import forecast_evaluation as fe

    date = pd.Timestamp("2025-03-31")
    outturns = pd.DataFrame(
        {
            "date": [date],
            "variable": ["gdp"],
            "frequency": ["Q"],
            "value": [1.0],
            "metric": ["levels"],
        }
    )
    forecasts = pd.DataFrame(
        {
            "date": [date] * 3,
            "vintage_date": [date] * 3,
            "variable": ["gdp"] * 3,
            "frequency": ["Q"] * 3,
            "forecast_horizon": [0] * 3,
            "source": ["model_a", "model_b", "model_c"],
            "value": [0.8, 1.1, 0.9],
            "metric": ["levels"] * 3,
        }
    )
    forecast_data = fe.ForecastData(
        outturns_data=outturns,
        forecasts_data=forecasts,
        outturn_vintages=False,
        compute_levels=False,
        data_check=False,
    )
    combo = fc.ForecastCombo(forecast_data)

    combo.fit(
        sources=["model_a", "model_b"],
        variables=["gdp"],
        method="average",
        metric="levels",
        label="my_combo",
        training_start=date,
        training_end=date,
    )

    with pytest.raises(ValueError, match="already used by a different fit configuration"):
        combo.fit(
            sources=["model_a", "model_b", "model_c"],
            variables=["gdp"],
            method="average",
            metric="levels",
            label="my_combo",
            training_start=date,
            training_end=date,
        )


def test_line_plot_by_horizon_invalid_y_axis(combo_df):
    with pytest.raises(ValueError, match="y_axis must be"):
        line_plot_by_horizon(combo_df, y_axis="horizon")


@pytest.fixture
def filter_weights_df():
    rows = []
    values = {
        ("gdp", 1, "first"): [0.2, 0.3],
        ("cpi", 1, "first"): [0.4, 0.5],
        ("gdp", 2, "first"): [0.6, 0.7],
        ("gdp", 1, "second"): [0.8, 0.9],
    }
    vintages = pd.to_datetime(["2025-03-31", "2025-06-30"])
    for vintage_index, vintage in enumerate(vintages):
        for (variable, horizon, combo_label), weights in values.items():
            rows.append(
                {
                    "vintage_date": vintage,
                    "horizon": horizon,
                    "variable": variable,
                    "method": "rmse",
                    "model": "model_a",
                    "weight": weights[vintage_index],
                    "combo_label": combo_label,
                    "frequency": "Q",
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("filters", "expected_weights"),
    [
        ({"variable": "cpi"}, [0.4, 0.5]),
        ({"horizon": 2}, [0.6, 0.7]),
        ({"combo_label": "second"}, [0.8, 0.9]),
    ],
)
def test_line_plot_by_vintage_applies_dimension_filters(filter_weights_df, filters, expected_weights):
    fig, axes = line_plot_by_vintage(filter_weights_df, y_axis="model", **filters)
    try:
        plotted = plotted_line_values(axes, "model_a")
        assert [value for _, value in plotted] == pytest.approx(expected_weights)
    finally:
        plt.close(fig)


@pytest.fixture
def minimal_weights():
    return pd.DataFrame(
        {
            "vintage_date": [pd.Timestamp("2025-03-31")],
            "horizon": [1],
            "variable": ["gdp"],
            "method": ["average"],
            "model": ["model_a"],
            "weight": [1.0],
        }
    )


@pytest.mark.parametrize(
    "plot_function, kwargs",
    [
        (line_plot_by_horizon, {"model": "missing"}),
        (line_plot_by_vintage, {"model": "missing"}),
        (heatmap_by_horizon, {"model": "missing"}),
        (heatmap_by_vintage, {"model": "missing"}),
        (bar_plot_by_horizon, {"model": "missing"}),
        (bar_plot_by_vintage, {"model": "missing"}),
    ],
)
def test_plot_functions_raise_clear_error_for_empty_filtered_data(minimal_weights, plot_function, kwargs):
    with pytest.raises(ValueError, match="No weight data remains after applying the selected filters"):
        plot_function(minimal_weights, **kwargs)


def test_plot_functions_raise_clear_error_for_missing_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        line_plot_by_horizon(pd.DataFrame())


def test_plot_validation_rejects_missing_horizon_column(minimal_weights):
    """Weights validated for plotting must key on 'horizon', not 'forecast_horizon'."""
    df = minimal_weights.drop(columns=["horizon"])
    with pytest.raises(ValueError, match="missing required columns.*horizon"):
        line_plot_by_horizon(df)


def test_dashboard_ui_rejects_empty_weights():
    with pytest.raises(ValueError, match="no fitted combination weights are available"):
        create_ui(pd.DataFrame())


def test_dashboard_ui_rejects_missing_weight_columns():
    with pytest.raises(ValueError, match="weights are missing columns"):
        create_ui(pd.DataFrame({"model": ["model_a"]}))


@pytest.mark.parametrize("missing_column", ["vintage_date", "weight", "frequency"])
def test_dashboard_ui_rejects_missing_plot_columns(missing_column):
    weights = pd.DataFrame(
        {
            "method": ["average"],
            "model": ["model_a"],
            "variable": ["gdp"],
            "horizon": [1],
            "vintage_date": [pd.Timestamp("2025-03-31")],
            "weight": [1.0],
            "frequency": ["Q"],
        }
    ).drop(columns=missing_column)

    with pytest.raises(ValueError, match="weights are missing columns"):
        create_ui(weights)


def test_line_plot_uses_matplotlib_default_colour_cycle(minimal_weights):
    fig, axes = line_plot_by_horizon(minimal_weights, y_axis="model")
    try:
        expected_colour = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        line = next(line for ax in axes.flat if ax.get_visible() for line in ax.get_lines())
        assert line.get_color() == expected_colour
    finally:
        plt.close(fig)


def test_line_plot_by_vintage_y_axis_model(combo_df):
    """The model view draws a faceted vintage plot."""
    fig, axes = line_plot_by_vintage(combo_df, y_axis="model")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_line_plot_by_vintage_y_axis_method(combo_df):
    """The method view draws a faceted vintage plot."""
    fig, axes = line_plot_by_vintage(combo_df, y_axis="method")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_line_plot_by_vintage_y_axis_variable(combo_df):
    """The variable view draws a faceted vintage plot."""
    fig, axes = line_plot_by_vintage(combo_df, y_axis="variable")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_line_plot_by_vintage_with_filters(combo_df):
    """The vintage plot applies dimension filters."""
    fig, axes = line_plot_by_vintage(combo_df, y_axis="model", method=["rmse"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_line_plot_by_vintage_invalid_y_axis(combo_df):
    with pytest.raises(ValueError, match="y_axis must be"):
        line_plot_by_vintage(combo_df, y_axis="horizon")


def test_heatmap_by_vintage_y_axis_model(combo_df):
    fig, axes = heatmap_by_vintage(combo_df, y_axis="model", method="rmse")
    assert isinstance(fig, plt.Figure)
    assert axes is not None
    plt.close(fig)


def test_heatmap_by_vintage_y_axis_method(combo_df):
    fig, axes = heatmap_by_vintage(combo_df, y_axis="method")
    assert isinstance(fig, plt.Figure)
    assert axes is not None
    plt.close(fig)


def test_heatmap_by_vintage_y_axis_variable(combo_df):
    fig, axes = heatmap_by_vintage(combo_df, y_axis="variable")
    assert isinstance(fig, plt.Figure)
    assert axes is not None
    plt.close(fig)


def test_heatmap_by_vintage_invalid_y_axis(combo_df):
    with pytest.raises(ValueError, match="y_axis must be"):
        heatmap_by_vintage(combo_df, y_axis="horizon")


@pytest.mark.parametrize("plot_func", [heatmap_by_vintage, heatmap_by_horizon])
def test_heatmap_uses_shared_colour_limits_across_facets(plot_func):
    weights = pd.DataFrame(
        [
            {
                "vintage_date": vintage,
                "horizon": horizon,
                "variable": "gdp",
                "method": method,
                "model": "model_a",
                "weight": weight,
                "frequency": "Q",
            }
            for method, values in [("low", [1.0, 2.0]), ("high", [100.0, 200.0])]
            for index, (vintage, horizon) in enumerate(
                zip(pd.to_datetime(["2025-03-31", "2025-06-30"]), [1, 2], strict=True)
            )
            for weight in [values[index]]
        ]
    )

    fig, axes = plot_func(weights, y_axis="model")
    try:
        images = [image for ax in axes.flat if ax.get_visible() for image in ax.get_images()]
        assert len(images) == 2
        assert {image.get_clim() for image in images} == {(1.0, 200.0)}

        colourbar = next(axis._colorbar for axis in fig.axes if axis not in axes.flat)
        assert colourbar.mappable.get_clim() == pytest.approx((1.0, 200.0))
    finally:
        plt.close(fig)


def test_heatmap_by_horizon_y_axis_model(combo_df):
    fig, axes = heatmap_by_horizon(combo_df, y_axis="model")
    assert isinstance(fig, plt.Figure)
    assert axes is not None
    plt.close(fig)


def test_heatmap_by_horizon_y_axis_method(combo_df):
    fig, axes = heatmap_by_horizon(combo_df, y_axis="method")
    assert isinstance(fig, plt.Figure)
    assert axes is not None
    plt.close(fig)


def test_heatmap_by_horizon_y_axis_variable(combo_df):
    fig, axes = heatmap_by_horizon(combo_df, y_axis="variable")
    assert isinstance(fig, plt.Figure)
    assert axes is not None
    plt.close(fig)


def test_heatmap_by_horizon_invalid_y_axis(combo_df):
    with pytest.raises(ValueError, match="y_axis must be"):
        heatmap_by_horizon(combo_df, y_axis="horizon")


@pytest.mark.parametrize("plot_func", [heatmap_by_vintage, heatmap_by_horizon])
def test_heatmap_colourbar_resizes_all_visible_subplots(plot_func):
    weights = pd.DataFrame(
        [
            {
                "vintage_date": pd.Timestamp("2025-03-31"),
                "horizon": 1,
                "variable": "gdp",
                "method": method,
                "model": "model_a",
                "weight": weight,
                "frequency": "Q",
            }
            for method, weight in [("average", 0.4), ("rmse", 0.6)]
        ]
    )

    fig, axes = plot_func(weights, y_axis="model")
    try:
        visible_axes = [ax for ax in axes.flat if ax.get_visible()]
        widths = [ax.get_position().width for ax in visible_axes]
        assert max(widths) == pytest.approx(min(widths))
    finally:
        plt.close(fig)


@pytest.mark.parametrize("plot_func", [heatmap_by_vintage, heatmap_by_horizon])
def test_heatmap_is_compatible_with_shiny_plot_rendering(plot_func):
    weights = pd.DataFrame(
        {
            "vintage_date": pd.to_datetime(["2025-03-31", "2025-06-30"]),
            "horizon": [1, 2],
            "variable": ["gdp", "gdp"],
            "method": ["average", "average"],
            "model": ["model_a", "model_a"],
            "weight": [0.4, 0.6],
            "frequency": ["Q", "Q"],
        }
    )

    fig, _ = plot_func(weights, y_axis="model")
    try:
        coordmap = get_coordmap(fig)
        assert coordmap is not None
        assert len(coordmap["panels"]) == len(fig.axes)
    finally:
        plt.close(fig)


def test_bar_plot_by_horizon_y_axis_model(combo_df):
    fig, axes = bar_plot_by_horizon(combo_df, y_axis="model")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_bar_plot_by_horizon_y_axis_method(combo_df):
    fig, axes = bar_plot_by_horizon(combo_df, y_axis="method")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_bar_plot_by_horizon_y_axis_variable(combo_df):
    fig, axes = bar_plot_by_horizon(combo_df, y_axis="variable")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_bar_plot_by_horizon_with_filters(combo_df):
    fig, axes = bar_plot_by_horizon(combo_df, y_axis="model", method=["rmse"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_bar_plot_by_horizon_invalid_y_axis(combo_df):
    with pytest.raises(ValueError, match="y_axis must be"):
        bar_plot_by_horizon(combo_df, y_axis="horizon")


def test_bar_plot_by_vintage_y_axis_model(combo_df):
    fig, axes = bar_plot_by_vintage(combo_df, y_axis="model")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_bar_plot_by_vintage_y_axis_method(combo_df):
    fig, axes = bar_plot_by_vintage(combo_df, y_axis="method")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_bar_plot_by_vintage_y_axis_variable(combo_df):
    fig, axes = bar_plot_by_vintage(combo_df, y_axis="variable")
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_bar_plot_by_vintage_with_filters(combo_df):
    fig, axes = bar_plot_by_vintage(combo_df, y_axis="model", method=["rmse"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_bar_plot_by_vintage_invalid_y_axis(combo_df):
    with pytest.raises(ValueError, match="y_axis must be"):
        bar_plot_by_vintage(combo_df, y_axis="horizon")


def test_top_level_imports():
    assert hasattr(fc, "line_plot_by_vintage")
    assert hasattr(fc, "line_plot_by_horizon")
    assert hasattr(fc, "heatmap_by_vintage")
    assert hasattr(fc, "heatmap_by_horizon")
    assert hasattr(fc, "bar_plot_by_vintage")
    assert hasattr(fc, "bar_plot_by_horizon")


def test_top_level_bar_plot_by_horizon(combo_df):
    fig, axes = fc.bar_plot_by_horizon(combo_df, y_axis="model")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_top_level_bar_plot_by_vintage(combo_df):
    fig, axes = fc.bar_plot_by_vintage(combo_df, y_axis="model")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_top_level_line_plot_by_horizon(combo_df):
    fig, axes = fc.line_plot_by_horizon(combo_df, y_axis="model")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_top_level_line_plot_by_vintage(combo_df):
    fig, axes = fc.line_plot_by_vintage(combo_df, y_axis="model")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_top_level_heatmap_by_vintage(combo_df):
    fig, axes = fc.heatmap_by_vintage(combo_df, y_axis="model")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_top_level_heatmap_by_horizon(combo_df):
    fig, axes = fc.heatmap_by_horizon(combo_df, y_axis="model")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.fixture
def negative_weights_df():
    """Synthetic weights containing negative and greater-than-one values."""
    rows = []
    for vintage in pd.to_datetime(["2024-01-01", "2024-04-01"]):
        for horizon in (1, 2):
            for model, weight in [("a", -0.4), ("b", 1.4)]:
                rows.append(
                    {
                        "combo_label": "ols",
                        "model": model,
                        "method": "least_squares",
                        "variable": "cpisa",
                        "horizon": horizon,
                        "vintage_date": vintage,
                        "weight": weight,
                        "frequency": "Q",
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("plot_func", [heatmap_by_vintage, heatmap_by_horizon])
def test_heatmap_colour_scale_covers_negative_weights(negative_weights_df, plot_func):
    """Colour limits must span the data instead of being pinned to [0, 1]."""
    fig, _ = plot_func(negative_weights_df, y_axis="model")
    images = [im for ax in fig.axes for im in ax.get_images()]
    assert images
    for im in images:
        vmin, vmax = im.get_clim()
        assert vmin <= -0.4
        assert vmax >= 1.4
    plt.close(fig)


@pytest.mark.parametrize("plot_func", [bar_plot_by_vintage, bar_plot_by_horizon])
def test_bar_plot_axis_shows_negative_weights(negative_weights_df, plot_func):
    """The y-axis must extend below zero so negative bars remain visible."""
    fig, axes = plot_func(negative_weights_df, y_axis="model")
    visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
    assert visible_axes
    for ax in visible_axes:
        assert ax.get_ylim()[0] < 0
    plt.close(fig)


@pytest.mark.parametrize("plot_func", [bar_plot_by_vintage, bar_plot_by_horizon])
def test_bar_plot_axis_still_starts_at_zero_for_nonnegative_weights(negative_weights_df, plot_func):
    """Non-negative weights keep the original zero-based axis."""
    df = negative_weights_df.copy()
    df["weight"] = df["weight"].abs()
    fig, axes = plot_func(df, y_axis="model")
    visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
    for ax in visible_axes:
        assert ax.get_ylim()[0] == 0
    plt.close(fig)


@pytest.mark.parametrize("plot_func", [bar_plot_by_vintage, bar_plot_by_horizon])
def test_bar_plot_preserves_missing_weights(plot_func):
    weights = pd.DataFrame(
        [
            {
                "vintage_date": vintage,
                "horizon": horizon,
                "variable": "gdp",
                "method": "average",
                "model": model,
                "weight": weight,
                "frequency": "Q",
            }
            for vintage, horizon, model, weight in [
                (pd.Timestamp("2025-03-31"), 1, "model_a", 0.4),
                (pd.Timestamp("2025-03-31"), 1, "model_b", 0.6),
                (pd.Timestamp("2025-06-30"), 2, "model_a", 0.5),
                (pd.Timestamp("2025-06-30"), 2, "model_b", np.nan),
            ]
        ]
    )

    fig, axes = plot_func(weights, y_axis="model")
    try:
        heights = [
            patch.get_height()
            for ax in axes.flat
            if ax.get_visible()
            for container in ax.containers
            for patch in container.patches
        ]
        assert any(np.isnan(height) for height in heights)
        assert any(text.get_text() == "NA" for ax in axes.flat if ax.get_visible() for text in ax.texts)
    finally:
        plt.close(fig)


# ===========================================================================
# Bars are only stacked where the segments form a meaningful total
# ===========================================================================


def _bar_x_positions(ax):
    """Group each bar's x centre by container (one container per segment)."""
    return [tuple(patch.get_x() + patch.get_width() / 2 for patch in container) for container in ax.containers]


@pytest.mark.parametrize("plot_func", [bar_plot_by_vintage, bar_plot_by_horizon])
def test_bar_plot_stacks_model_weights(combo_df, plot_func):
    """Model weights within a facet sum to one, so they are stacked."""
    fig, axes = plot_func(combo_df, y_axis="model")
    ax = next(ax for ax in axes.ravel() if ax.get_visible() and ax.containers)
    positions = _bar_x_positions(ax)
    assert len(positions) > 1
    assert all(pos == positions[0] for pos in positions)  # segments share x centres
    plt.close(fig)


@pytest.mark.parametrize("plot_func", [bar_plot_by_vintage, bar_plot_by_horizon])
def test_bar_plot_groups_method_segments(combo_df, plot_func):
    """Methods are unrelated weight sets, so they are drawn side by side."""
    fig, axes = plot_func(combo_df, y_axis="method")
    ax = next(ax for ax in axes.ravel() if ax.get_visible() and ax.containers)
    positions = _bar_x_positions(ax)
    assert len(positions) > 1
    assert all(pos != positions[0] for pos in positions[1:])  # segments are offset
    plt.close(fig)


@pytest.mark.parametrize("plot_func", [bar_plot_by_vintage, bar_plot_by_horizon])
def test_bar_plot_groups_negative_weights(negative_weights_df, plot_func):
    """Mixed-sign weights are not stacked, since the totals would be unreadable."""
    fig, axes = plot_func(negative_weights_df, y_axis="model")
    ax = next(ax for ax in axes.ravel() if ax.get_visible() and ax.containers)
    positions = _bar_x_positions(ax)
    assert len(positions) > 1
    assert all(pos != positions[0] for pos in positions[1:])
    plt.close(fig)


@pytest.mark.parametrize("plot_func", [bar_plot_by_vintage, bar_plot_by_horizon])
def test_bar_plot_groups_incomplete_model_weights(negative_weights_df, plot_func):
    """A filtered or non-simplex model set appears as grouped bars."""
    df = negative_weights_df.copy()
    df["weight"] = df["weight"].abs() / 2

    fig, axes = plot_func(df, y_axis="model")
    ax = next(ax for ax in axes.ravel() if ax.get_visible() and ax.containers)
    positions = _bar_x_positions(ax)
    assert len(positions) > 1
    assert all(pos != positions[0] for pos in positions[1:])
    plt.close(fig)


# ===========================================================================
# Plotted values must equal the aggregated input weights
# ===========================================================================


@pytest.fixture
def known_weights_df():
    """Weights whose per-horizon and per-vintage averages are known exactly.

    Model 'a' takes weights 0.2 and 0.4 at horizon 1 (mean 0.3) and 0.6 and
    0.8 at horizon 2 (mean 0.7); model 'b' is its complement.
    """
    vintages = pd.to_datetime(["2024-03-31", "2024-06-30"])
    weights_a = {(1, vintages[0]): 0.2, (1, vintages[1]): 0.4, (2, vintages[0]): 0.6, (2, vintages[1]): 0.8}
    rows = []
    for (horizon, vintage), weight_a in weights_a.items():
        for model, weight in [("a", weight_a), ("b", 1 - weight_a)]:
            rows.append(
                {
                    "combo_label": "combo",
                    "model": model,
                    "method": "rmse",
                    "variable": "cpisa",
                    "horizon": horizon,
                    "vintage_date": vintage,
                    "weight": weight,
                    "frequency": "Q",
                }
            )
    return pd.DataFrame(rows)


def test_line_plot_by_horizon_plots_vintage_averages(known_weights_df):
    """Each horizon shows the mean weight across vintages, without clipping."""
    fig, axes = line_plot_by_horizon(known_weights_df, y_axis="model")
    try:
        assert dict(plotted_line_values(axes, "a")) == pytest.approx({1: 0.3, 2: 0.7})
        assert dict(plotted_line_values(axes, "b")) == pytest.approx({1: 0.7, 2: 0.3})
        assert_no_vertical_clipping(axes, [0.3, 0.7])
    finally:
        plt.close(fig)


def test_line_plot_by_vintage_plots_horizon_averages(known_weights_df):
    """Each vintage shows the mean weight across horizons, without clipping."""
    fig, axes = line_plot_by_vintage(known_weights_df, y_axis="model")
    try:
        plotted = plotted_line_values(axes, "a")
        assert len(plotted) == 2
        assert sorted(value for _, value in plotted) == pytest.approx([0.4, 0.6])
        assert_no_vertical_clipping(axes, [0.4, 0.6])
    finally:
        plt.close(fig)


def test_line_plot_by_horizon_respects_model_filter(known_weights_df):
    """Filtering to one model must remove the other model's line entirely."""
    fig, axes = line_plot_by_horizon(known_weights_df, y_axis="model", model=["a"])
    try:
        assert dict(plotted_line_values(axes, "a")) == pytest.approx({1: 0.3, 2: 0.7})
        assert plotted_line_values(axes, "b") == []
    finally:
        plt.close(fig)


def test_bar_plot_by_horizon_bar_heights_match_averages(known_weights_df):
    """Stacked model segments carry the aggregated weights and total one."""
    fig, axes = bar_plot_by_horizon(known_weights_df, y_axis="model")
    try:
        heights = plotted_bar_values(axes)
        assert sorted(heights) == pytest.approx([0.3, 0.3, 0.7, 0.7])

        ax = next(ax for ax in np.ravel(axes) if ax.get_visible() and ax.containers)
        totals = np.sum([[patch.get_height() for patch in c.patches] for c in ax.containers], axis=0)
        assert totals == pytest.approx(np.ones(len(totals)))
    finally:
        plt.close(fig)


def test_heatmap_by_vintage_cells_match_input_weights(known_weights_df):
    """Heatmap cells contain the horizon-averaged weights, in sorted row order."""
    fig, _ = heatmap_by_vintage(known_weights_df, y_axis="model")
    try:
        images = [im for ax in fig.axes for im in ax.get_images()]
        assert len(images) == 1
        # Rows are models 'a' then 'b'; columns are the two vintages in order.
        np.testing.assert_allclose(images[0].get_array(), np.array([[0.4, 0.6], [0.6, 0.4]]))
    finally:
        plt.close(fig)
