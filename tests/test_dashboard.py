from unittest.mock import MagicMock

import pandas as pd

from forecast_combo import ForecastCombo
from forecast_combo.dashboard.create_app import dashboard_app
from forecast_combo.dashboard.tabs import by_horizon, by_vintage
from forecast_combo.dashboard.ui import create_ui


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["average", "rmse"],
            "model": ["model_a", "model_b"],
            "variable": ["gdp", "gdp"],
            "horizon": [1, 2],
            "vintage_date": [pd.Timestamp("2025-03-31")] * 2,
            "frequency": ["Q", "Q"],
            "weight": [0.5, 0.5],
        }
    )


def _combo_with_weights(weights: pd.DataFrame | None = None) -> ForecastCombo:
    combo = ForecastCombo.__new__(ForecastCombo)
    combo.weights = _weights() if weights is None else weights
    return combo


def _capture_plot_handlers(module, register_fn, input, combo_df, invoke):
    captured = {}

    def fake_plot_decorator(fn):
        return fn

    def capture_output(fn):
        captured[fn.__name__] = fn

    output = MagicMock(side_effect=capture_output)
    original_plot_functions = module._PLOT_FN
    calls = []

    def fake_plot(*args, **kwargs):
        calls.append((args, kwargs))
        return "figure", None

    module._PLOT_FN = {plot_type: fake_plot for plot_type in ("line", "heatmap", "bar")}
    try:
        with _patched_plot_decorator(module, fake_plot_decorator):
            register_fn(input, output, MagicMock(), combo_df)
            captured[invoke]()
    finally:
        module._PLOT_FN = original_plot_functions

    return captured, calls


class _patched_plot_decorator:
    def __init__(self, module, decorator):
        self.module = module
        self.decorator = decorator

    def __enter__(self):
        self.original = self.module.render.plot
        self.module.render.plot = self.decorator

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.render.plot = self.original


def test_dashboard_ui_contains_filters_tabs_and_plot_outputs():
    html = str(create_ui(_weights()))

    for input_id in ("plot_type", "variable", "model", "method", "horizon"):
        assert f'id="{input_id}"' in html
    for tab_name in ("Across Horizon", "Across Vintage"):
        assert f'data-value="{tab_name}"' in html
    for output_id in (
        "plot_horizon_model",
        "plot_horizon_method",
        "plot_horizon_variable",
        "plot_vintage_model",
        "plot_vintage_method",
        "plot_vintage_variable",
    ):
        assert f'id="{output_id}"' in html


def test_dashboard_app_registers_horizon_and_vintage_tabs(monkeypatch):
    combo = _combo_with_weights()
    app = dashboard_app(combo)
    horizon_handler = MagicMock()
    vintage_handler = MagicMock()
    monkeypatch.setattr("forecast_combo.dashboard.create_app.horizon_tab", horizon_handler)
    monkeypatch.setattr("forecast_combo.dashboard.create_app.vintage_tab", vintage_handler)

    app.server(MagicMock(), MagicMock(), MagicMock())

    horizon_handler.assert_called_once()
    vintage_handler.assert_called_once()
    assert horizon_handler.call_args.args[3] is combo.weights
    assert vintage_handler.call_args.args[3] is combo.weights


def test_horizon_tab_registers_all_outputs_and_passes_selected_filters():
    input = MagicMock()
    input.plot_type.return_value = "line"
    input.model.return_value = ["model_a"]
    input.method.return_value = ["average"]
    input.variable.return_value = ["gdp"]

    captured, calls = _capture_plot_handlers(
        by_horizon,
        by_horizon.horizon_tab,
        input,
        _weights(),
        "plot_horizon_model",
    )

    assert set(captured) == {"plot_horizon_model", "plot_horizon_method", "plot_horizon_variable"}

    assert len(calls) == 1
    pd.testing.assert_frame_equal(calls[0][0][0], _weights())
    assert calls[0][1] == {
        "y_axis": "model",
        "model": ["model_a"],
        "method": ["average"],
        "variable": ["gdp"],
    }


def test_vintage_tab_converts_selected_horizons_and_passes_filters():
    input = MagicMock()
    input.plot_type.return_value = "heatmap"
    input.model.return_value = []
    input.method.return_value = ["rmse"]
    input.variable.return_value = ["gdp"]
    input.horizon.return_value = ["1", "2"]

    captured, calls = _capture_plot_handlers(
        by_vintage,
        by_vintage.vintage_tab,
        input,
        _weights(),
        "plot_vintage_method",
    )

    assert set(captured) == {"plot_vintage_model", "plot_vintage_method", "plot_vintage_variable"}

    assert len(calls) == 1
    pd.testing.assert_frame_equal(calls[0][0][0], _weights())
    assert calls[0][1] == {
        "y_axis": "method",
        "model": None,
        "method": ["rmse"],
        "variable": ["gdp"],
        "horizon": [1, 2],
    }


def test_run_combo_dashboard_delegates_to_shiny_app(monkeypatch):
    combo = _combo_with_weights()
    app = MagicMock()
    dashboard_factory = MagicMock(return_value=app)
    monkeypatch.setattr("forecast_combo.dashboard.create_app.dashboard_app", dashboard_factory)

    combo.run_combo_dashboard(host="localhost", port=8123)

    dashboard_factory.assert_called_once_with(combo)
    app.run.assert_called_once_with(host="localhost", port=8123)


def test_run_forecast_dashboard_delegates_to_existing_data():
    forecast_data = MagicMock()
    combo = ForecastCombo.__new__(ForecastCombo)
    combo.forecast_data = forecast_data

    combo.run_forecast_dashboard(host="localhost", port=8125)

    forecast_data.run_dashboard.assert_called_once_with(host="localhost", port=8125)
