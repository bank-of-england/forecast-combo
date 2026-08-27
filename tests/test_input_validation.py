import numpy as np
import pandas as pd
import pytest

import forecast_combo
import forecast_combo.combinations as combos
from forecast_combo import ComboSpec, ForecastCombo
from forecast_combo.combinations.static_combinations import delta_method
from forecast_combo.dashboard.create_app import dashboard_app
from forecast_combo.dashboard.ui import create_ui
from forecast_combo.forecast_combo import SUPPORTED_METHODS, get_weights, validate_spec_graph
from forecast_combo.utils import create_period_filter

WEIGHT_FUNCTIONS = [
    lambda X, y, window: combos.least_squares(X, y, window),
    lambda X, y, window: combos.constrained_least_squares(X, y, window),
    lambda X, y, window: combos.rmse_weights(X, y, window),
    lambda X, y, window: combos.mse_weights(X, y, window),
    lambda X, y, window: combos.mae_weights(X, y, window),
    lambda X, y, window: combos.huber_weights(X, y, window),
]


def test_supported_methods_share_one_immutable_source():
    assert ForecastCombo.supported_methods is SUPPORTED_METHODS
    with pytest.raises(AttributeError):
        SUPPORTED_METHODS.add("new_method")


def test_public_helpers_are_exported_from_package_root():
    assert forecast_combo.get_weights is get_weights
    assert forecast_combo.validate_spec_graph is validate_spec_graph
    assert forecast_combo.SUPPORTED_METHODS is SUPPORTED_METHODS
    assert {"get_weights", "validate_spec_graph", "SUPPORTED_METHODS"}.issubset(forecast_combo.__all__)


@pytest.mark.parametrize("weight_function", WEIGHT_FUNCTIONS)
def test_weight_functions_reject_non_finite_arrays(weight_function):
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    y = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="X must contain only finite values"):
        weight_function(X, y, None)


@pytest.mark.parametrize("weight_function", WEIGHT_FUNCTIONS)
@pytest.mark.parametrize("window_size", [0, -1])
def test_weight_functions_reject_non_positive_window(weight_function, window_size):
    X = np.array([[1.0, 3.0], [2.0, 4.0]])
    y = np.array([1.5, 2.5])

    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        weight_function(X, y, window_size)


@pytest.mark.parametrize("weight_function", WEIGHT_FUNCTIONS)
def test_weight_functions_reject_mismatched_shapes(weight_function):
    with pytest.raises(ValueError, match="must match length of y"):
        weight_function(np.ones((2, 2)), np.ones(3), None)


@pytest.mark.parametrize("weight_function", WEIGHT_FUNCTIONS)
def test_weight_functions_reject_broadcastable_target_shape(weight_function):
    with pytest.raises(ValueError, match="must match length of y"):
        weight_function(np.ones((2, 2)), np.ones(1), None)


@pytest.mark.parametrize("weight_function", WEIGHT_FUNCTIONS)
def test_weight_functions_reject_non_finite_targets(weight_function):
    with pytest.raises(ValueError, match="y must contain only finite values"):
        weight_function(np.ones((2, 2)), np.array([1.0, np.nan]), None)


def test_average_rejects_invalid_shape_and_empty_sources():
    with pytest.raises(ValueError, match="X must be a 2D array"):
        combos.average(np.ones(2))
    with pytest.raises(ValueError, match="at least one forecast source"):
        combos.average(np.empty((2, 0)))


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_average_rejects_non_finite_forecasts(invalid_value):
    with pytest.raises(ValueError, match="X must contain only finite values"):
        combos.average(np.array([[1.0, invalid_value]]))


def test_single_source_weight_is_one():
    weights, _ = get_weights(np.ones((2, 1)), np.ones(2), "average", None, 1.0)

    assert weights == pytest.approx([1.0])


def test_get_weights_defaults_match_documentation():
    X = np.array([[1.0, 2.0], [2.0, 4.0]])
    y = np.array([1.5, 3.0])

    default_weights, default_std_errors = get_weights(X, y, "rmse")
    explicit_weights, explicit_std_errors = get_weights(X, y, "rmse", None, 1.0)

    assert default_weights == pytest.approx(explicit_weights)
    assert default_std_errors == pytest.approx(explicit_std_errors)


@pytest.mark.parametrize(
    "weight_function",
    [
        lambda X, y: combos.constrained_least_squares(X, y),
        lambda X, y: combos.rmse_weights(X, y, window_size=None),
        lambda X, y: combos.mse_weights(X, y, window_size=None),
        lambda X, y: combos.mae_weights(X, y, window_size=None),
        lambda X, y: combos.huber_weights(X, y, window_size=None),
    ],
    ids=["constrained_least_squares", "rmse", "mse", "mae", "huber"],
)
def test_non_ols_weight_functions_reject_empty_samples(weight_function):
    with pytest.raises(ValueError, match="estimation sample is empty"):
        weight_function(np.empty((0, 2)), np.empty(0))


@pytest.mark.parametrize("weight_function", WEIGHT_FUNCTIONS)
def test_weight_functions_reject_boolean_window(weight_function):
    X = np.array([[1.0, 3.0], [2.0, 4.0]])
    y = np.array([1.5, 2.5])

    with pytest.raises(TypeError, match="window_size must be an integer"):
        weight_function(X, y, True)


def test_get_weights_average_tolerates_empty_target():
    """``average`` ignores ``y`` and accepts an empty target."""
    weights, std_errors = get_weights(np.empty((0, 2)), np.empty(0), "average", None, 1.0)

    assert weights == pytest.approx([0.5, 0.5])
    assert np.all(np.isnan(std_errors))


@pytest.mark.parametrize("method", ["least_squares", "constrained_least_squares", "rmse", "mse", "mae", "huber"])
def test_get_weights_rejects_empty_target_for_data_dependent_methods(method):
    with pytest.raises(ValueError, match="estimation sample is empty"):
        get_weights(np.empty((0, 2)), np.empty(0), method, None, 1.0)


@pytest.mark.parametrize(
    "X, y",
    [
        (np.ones(2), np.ones(2)),
        (np.ones((2, 1)), np.ones((2, 1))),
        (np.ones((2, 1)), np.ones(3)),
    ],
    ids=["x_not_2d", "y_not_1d", "length_mismatch"],
)
def test_forecast_matrix_validation_parity_between_get_weights_and_combos(X, y):
    """``get_weights`` and the static combination functions must reject the
    same malformed arrays with the same exception type and message."""
    with pytest.raises(ValueError) as combo_exc:
        combos.least_squares(X, y)
    with pytest.raises(ValueError) as gw_exc:
        get_weights(X, y, "least_squares", None, 1.0)

    assert str(combo_exc.value) == str(gw_exc.value)


def test_get_weights_rejects_non_finite_arrays():
    with pytest.raises(ValueError, match="X must contain only finite values"):
        get_weights(
            np.array([[1.0, np.nan], [2.0, 3.0]]),
            np.array([1.0, 2.0]),
            "rmse",
            None,
            1.0,
        )


def test_get_weights_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown combination method"):
        get_weights(np.ones((2, 1)), np.ones(2), "unknown", None, 1.0)


@pytest.mark.parametrize(
    "X, y, match",
    [
        (np.ones(2), np.ones(2), "X must be a 2D array"),
        (np.ones((2, 1)), np.ones((2, 1)), "y must be a 1D array"),
        (np.ones((2, 1)), np.ones(3), "must match length of y"),
    ],
)
def test_get_weights_rejects_invalid_shapes(X, y, match):
    with pytest.raises(ValueError, match=match):
        get_weights(X, y, "average", None, 1.0)


def test_huber_rejects_sample_too_small_for_scale_estimate():
    with pytest.raises(ValueError, match="at least two observations"):
        combos.huber_weights(np.array([[1.0, 2.0]]), np.array([1.5]), None)


def test_huber_rejects_overflowing_intermediate_loss():
    with pytest.raises(ValueError, match="non-finite Huber loss"):
        combos.huber_weights(np.zeros((2, 2)), np.array([1e308, -1e308]), None)


def test_mae_uses_scale_safe_inverse_loss_normalisation():
    tiny = np.nextafter(0.0, 1.0)
    X = np.array([[tiny, 2 * tiny], [-tiny, -2 * tiny]])

    weights, std_errors = combos.mae_weights(X, np.zeros(2), None)

    assert weights == pytest.approx([2 / 3, 1 / 3])
    assert np.all(np.isfinite(std_errors))


def test_delta_method_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="must not be empty"):
        delta_method(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="finite positive"):
        delta_method(np.array([0.0, 1.0]), np.array([0.1, 0.1]))
    with pytest.raises(ValueError, match="same shape"):
        delta_method(np.array([1.0]), np.array([0.1, 0.2]))


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"name": "", "sources": ["a"]}, ValueError, "name must not be empty"),
        ({"name": "x", "sources": []}, ValueError, "sources must not be empty"),
        ({"name": "x", "sources": "a"}, TypeError, "sources must be a list"),
        ({"name": "x", "sources": ["a"], "method": "unknown"}, ValueError, "invalid method"),
        ({"name": "x", "sources": ["a"], "training_start": "not-a-date"}, ValueError, "valid date"),
        ({"name": "x", "sources": ["a"], "allow_partial_sources": "yes"}, TypeError, "must be a bool"),
    ],
)
def test_combo_spec_rejects_invalid_fields(kwargs, error, match):
    with pytest.raises(error, match=match):
        ComboSpec(**kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"k": True}, "k must be an integer"),
        ({"window_size": True}, "window_size must be an integer"),
        ({"discount_param": True}, "discount_param must be a number"),
    ],
)
def test_combo_spec_rejects_boolean_numeric_fields(kwargs, match):
    with pytest.raises(TypeError, match=match):
        ComboSpec(name="x", sources=["a"], **kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"k": True}, "k must be an integer"),
        ({"window_size": True}, "window_size must be an integer"),
        ({"discount_param": True}, "discount_param must be a number"),
    ],
)
def test_fit_rejects_boolean_numeric_options_before_accessing_data(kwargs, match):
    combo = ForecastCombo.__new__(ForecastCombo)

    with pytest.raises(TypeError, match=match):
        combo.fit(sources=["a"], variables=["x"], **kwargs)


def test_validate_spec_graph_revalidates_mutated_specs():
    spec = ComboSpec("valid", ["a"])
    spec.sources = []

    with pytest.raises(ValueError, match="sources must not be empty"):
        validate_spec_graph([spec])


def test_validate_spec_graph_rejects_invalid_roots():
    with pytest.raises(TypeError, match="only ComboSpec objects"):
        validate_spec_graph(["not-a-spec"])


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"label": 1}, TypeError, "label must be a string"),
        ({"label": " "}, ValueError, "label must not be empty"),
        ({"automatic_labelling": "yes"}, TypeError, "automatic_labelling must be a bool"),
        ({"allow_partial_sources": None}, TypeError, "allow_partial_sources must be a bool"),
        ({"print_warning": "no"}, TypeError, "print_warning must be a bool"),
        ({"metric": None}, TypeError, "metric must be a string"),
    ],
)
def test_fit_rejects_invalid_options_before_accessing_data(kwargs, error, match):
    combo = ForecastCombo.__new__(ForecastCombo)

    with pytest.raises(error, match=match):
        combo.fit(sources=["a"], variables=["x"], **kwargs)


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"method": 1}, TypeError, "method must be a string"),
        ({"window_size": 0}, ValueError, "window_size must be a positive integer"),
        ({"discount_param": 2}, ValueError, "discount_param must be in the interval"),
    ],
)
def test_get_weights_rejects_invalid_scalar_arguments(kwargs, error, match):
    call_kwargs = {"method": "average", "window_size": None, "discount_param": 1.0}
    call_kwargs.update(kwargs)

    with pytest.raises(error, match=match):
        get_weights(np.ones((2, 1)), np.ones(2), **call_kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"window_size": True}, "window_size must be an integer"),
        ({"discount_param": True}, "discount_param must be a number"),
    ],
)
def test_get_weights_rejects_boolean_scalar_arguments(kwargs, match):
    call_kwargs = {"method": "average", "window_size": None, "discount_param": 1.0}
    call_kwargs.update(kwargs)

    with pytest.raises(TypeError, match=match):
        get_weights(np.ones((2, 1)), np.ones(2), **call_kwargs)


def test_create_period_filter_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="must not be after"):
        create_period_filter("2021Q4", "2020Q1", "Q")
    with pytest.raises(ValueError, match="must not be missing"):
        create_period_filter(np.datetime64("NaT"), "2020Q1", "Q")
    with pytest.raises(TypeError, match="scalar period-like"):
        create_period_filter(["2020Q1"], "2020Q2", "Q")


def test_create_period_filter_includes_both_bounds():
    assert create_period_filter("2020Q1", "2020Q1", "Q") == [pd.Period("2020Q1", freq="Q")]
    assert create_period_filter("2020Q1", "2020Q3", "Q") == [
        pd.Period("2020Q1", freq="Q"),
        pd.Period("2020Q2", freq="Q"),
        pd.Period("2020Q3", freq="Q"),
    ]


def test_dashboard_app_requires_forecast_combo():
    with pytest.raises(TypeError, match="must be a ForecastCombo"):
        dashboard_app(object())


def test_dashboard_ui_rejects_invalid_horizons():
    weights = pd.DataFrame(
        {
            "method": ["average"],
            "model": ["a"],
            "variable": ["gdp"],
            "horizon": [np.nan],
            "vintage_date": [pd.Timestamp("2025-03-31")],
            "weight": [1.0],
            "frequency": ["Q"],
        }
    )
    with pytest.raises(ValueError, match="finite integers"):
        create_ui(weights)


def test_dashboard_ui_normalises_integral_float_horizons():
    weights = pd.DataFrame(
        {
            "method": ["average"],
            "model": ["a"],
            "variable": ["gdp"],
            "horizon": [1.0],
            "vintage_date": [pd.Timestamp("2025-03-31")],
            "weight": [1.0],
            "frequency": ["Q"],
        }
    )

    assert 'option value="1"' in str(create_ui(weights))


def test_dashboard_ui_rejects_boolean_horizons():
    weights = pd.DataFrame(
        {
            "method": ["average"],
            "model": ["a"],
            "variable": ["gdp"],
            "horizon": [True],
            "vintage_date": [pd.Timestamp("2025-03-31")],
            "weight": [1.0],
            "frequency": ["Q"],
        }
    )

    with pytest.raises(TypeError, match="must not contain booleans"):
        create_ui(weights)


def test_dashboard_ui_accepts_minimal_valid_weights_with_horizon():
    weights = pd.DataFrame(
        {
            "method": ["average"],
            "model": ["a"],
            "variable": ["gdp"],
            "horizon": [1],
            "vintage_date": [pd.Timestamp("2025-03-31")],
            "weight": [1.0],
            "frequency": ["Q"],
        }
    )

    assert 'option value="1"' in str(create_ui(weights))


def test_dashboard_ui_rejects_missing_horizon_column():
    weights = pd.DataFrame(
        {
            "method": ["average"],
            "model": ["a"],
            "variable": ["gdp"],
            "vintage_date": [pd.Timestamp("2025-03-31")],
            "weight": [1.0],
            "frequency": ["Q"],
        }
    )

    with pytest.raises(ValueError, match="missing columns.*horizon"):
        create_ui(weights)


def test_failed_mixed_hierarchy_restores_state(monkeypatch):
    combo = ForecastCombo.__new__(ForecastCombo)
    combo.forecast_data = type("Data", (), {"forecasts": pd.DataFrame({"unique_id": ["raw"]})})()
    combo.forecast_data.copy = lambda: combo.forecast_data
    combo.weights = pd.DataFrame({"weight": [1.0]})
    combo._combo_unique_ids = {}
    combo._combo_labels = {}
    child = ComboSpec("child", ["raw"])

    def fail_after_child(spec, variables):
        combo.weights = pd.DataFrame({"weight": [2.0]})
        combo._combo_unique_ids[spec.name] = "child-id"
        raise ValueError("parent failed")

    monkeypatch.setattr(combo, "_fit_single_spec", fail_after_child)
    with pytest.raises(ValueError, match="parent failed"):
        combo._fit_from_spec_list([child, "raw"], ["x"])

    assert combo.weights["weight"].tolist() == [1.0]
    assert combo._combo_unique_ids == {}


def test_legacy_hierarchy_resolves_string_combo_references(monkeypatch):
    combo = ForecastCombo.__new__(ForecastCombo)
    combo._combo_unique_ids = {"child": "child-id"}
    parent = ComboSpec("parent", ["child", "raw"])
    captured = {}

    def capture_fit(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(combo, "fit", capture_fit)
    monkeypatch.setattr(combo, "_register_combo_unique_id", lambda name: None)
    combo._fit_single_spec(parent, ["gdp"])

    assert captured["sources"] == ["child-id", "raw"]


@pytest.mark.parametrize(
    "method_name, kwargs, error, match",
    [
        ("run_combo_dashboard", {"host": " "}, ValueError, "host must not be empty"),
        ("run_forecast_dashboard", {"port": 0}, ValueError, "port must be in the range"),
        ("run_forecast_dashboard", {"port": "8000"}, TypeError, "port must be an integer"),
    ],
)
def test_dashboard_methods_reject_invalid_server_arguments(method_name, kwargs, error, match):
    combo = ForecastCombo.__new__(ForecastCombo)

    with pytest.raises(error, match=match):
        getattr(combo, method_name)(**kwargs)
