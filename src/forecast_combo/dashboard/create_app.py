"""Main dashboard application."""

from shiny import App

from forecast_combo.dashboard.tabs.by_horizon import horizon_tab
from forecast_combo.dashboard.tabs.by_vintage import vintage_tab
from forecast_combo.dashboard.ui import create_ui
from forecast_combo.forecast_combo import ForecastCombo


def dashboard_app(combo_data: ForecastCombo) -> App:
    """Build the Shiny combo dashboard for a fitted combination.

    Parameters
    ----------
    combo_data : ForecastCombo
        A fitted ``ForecastCombo`` whose ``weights`` DataFrame is displayed.

    Returns
    -------
    App
        The Shiny application, ready to be run or embedded.

    Raises
    ------
    TypeError
        If ``combo_data`` is not a :class:`ForecastCombo`.
    """
    if not isinstance(combo_data, ForecastCombo):
        raise TypeError(f"combo_data must be a ForecastCombo, got {type(combo_data).__name__}")

    weights = combo_data.weights
    app_ui = create_ui(weights)

    def server(input, output, session):
        horizon_tab(input, output, session, weights)
        vintage_tab(input, output, session, weights)

    return App(app_ui, server)
