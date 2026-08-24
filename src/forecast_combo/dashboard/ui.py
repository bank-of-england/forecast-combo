"""UI components for the dashboard."""

import pandas as pd
from shiny import ui

from forecast_combo.visualisations._plot_utils import validate_frequency_column, validate_plot_data


def create_ui(combo_df: pd.DataFrame) -> ui.Tag:
    """Build the dashboard UI from a weights DataFrame.

    Parameters
    ----------
    combo_df : pd.DataFrame
        Combination weights, which must satisfy the plotting data contract.
        Their unique values populate the sidebar filters.

    Returns
    -------
    ui.Tag
        The page layout for the dashboard.

    Raises
    ------
    ValueError
        If ``combo_df`` is empty or is missing any required column.
    TypeError
        If ``combo_df`` is not a pandas DataFrame.
    """
    if not isinstance(combo_df, pd.DataFrame):
        raise TypeError(f"combo_df must be a pandas DataFrame, got {type(combo_df).__name__}")
    if combo_df.empty:
        raise ValueError("Cannot create dashboard: no fitted combination weights are available")

    required_columns = {
        "method",
        "model",
        "variable",
        "horizon",
        "vintage_date",
        "weight",
        "frequency",
    }
    missing = sorted(required_columns - set(combo_df.columns))
    if missing:
        raise ValueError(f"Cannot create dashboard: weights are missing columns: {', '.join(missing)}")

    validate_plot_data(combo_df)
    validate_frequency_column(combo_df)

    horizon_values = combo_df["horizon"].to_numpy(dtype=int)

    method_choices = sorted(combo_df["method"].unique())
    models = sorted(combo_df["model"].unique())
    variables = sorted(combo_df["variable"].unique())
    horizons = sorted(set(horizon_values))

    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_radio_buttons(
                    "plot_type",
                    "Plot type",
                    {"line": "Line", "heatmap": "Heatmap", "bar": "Bar"},
                    selected="line",
                ),
                ui.input_selectize(
                    "variable",
                    "Variables",
                    choices=variables,
                    selected=variables,
                    multiple=True,
                ),
                ui.input_selectize(
                    "model",
                    "Models",
                    choices=models,
                    selected=models,
                    multiple=True,
                ),
                ui.input_selectize(
                    "method",
                    "Methods",
                    choices=method_choices,
                    selected=method_choices,
                    multiple=True,
                ),
            ),
            ui.navset_tab(
                ui.nav_panel(
                    "Across Horizon",
                    ui.navset_tab(
                        ui.nav_panel("Model", ui.output_plot("plot_horizon_model", height="700px")),
                        ui.nav_panel("Method", ui.output_plot("plot_horizon_method", height="700px")),
                        ui.nav_panel("Variable", ui.output_plot("plot_horizon_variable", height="700px")),
                    ),
                ),
                ui.nav_panel(
                    "Across Vintage",
                    ui.input_selectize(
                        "horizon",
                        "Forecast horizons",
                        choices=[str(h) for h in horizons],
                        selected=[str(horizons[0])],
                        multiple=True,
                    ),
                    ui.navset_tab(
                        ui.nav_panel("Model", ui.output_plot("plot_vintage_model", height="700px")),
                        ui.nav_panel("Method", ui.output_plot("plot_vintage_method", height="700px")),
                        ui.nav_panel("Variable", ui.output_plot("plot_vintage_variable", height="700px")),
                    ),
                ),
            ),
        )
    )
