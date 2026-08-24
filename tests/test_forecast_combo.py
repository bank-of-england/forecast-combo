"""
Tests for ForecastCombo class.
"""

import importlib

import forecast_evaluation as fe
import numpy as np
import pandas as pd
import pytest

from forecast_combo import ComboSpec, ForecastCombo

forecast_combo_module = importlib.import_module("forecast_combo.forecast_combo")


def _fit_unfiltered_baseline(fer_data):
    """Fit the reference combination used by the period-filter comparison."""
    forecast_data = fer_data.copy()
    combo = ForecastCombo(forecast_data=forecast_data)

    # just one vintage in this test for speed
    last_vintage = np.sort(forecast_data.outturns["vintage_date"].unique())[-1]
    combo.fit(
        sources=["mpr", "baseline ar(p) model"],
        variables=["gdpkp", "cpisa"],
        method=["average", "least_squares", "constrained_least_squares"],
        training_start=last_vintage,
        metric="pop",
    )
    return combo


@pytest.fixture(scope="module")
def unfiltered_baseline(fer_data):
    """Read-only baseline results fitted without a period filter.

    Copies are returned so that a consuming test cannot mutate the shared
    fixture and change the behaviour of the other tests in this module.
    """
    combo = _fit_unfiltered_baseline(fer_data)
    return {
        "weights": combo.weights.copy(),
        "combined_forecasts": combo._combined_forecasts.copy(),
    }


def test_forecast_combo_fit_runs(fer_data):
    """Test that ForecastCombo.fit() runs without errors."""
    combo = _fit_unfiltered_baseline(fer_data)

    # Assert that combined forecasts were created
    assert not combo._combined_forecasts.empty

    # Assert that weights were stored
    assert len(combo.weights) > 0


def test_fit_raises_clear_error_when_no_combined_forecasts_are_produced(monkeypatch, fer_data):
    """An empty estimation result should raise the explanatory fit error."""
    monkeypatch.setattr(forecast_combo_module, "_estimation_loop", lambda **kwargs: ([], []))
    combo = ForecastCombo(forecast_data=fer_data.copy())

    with pytest.raises(ValueError, match="No combined forecasts were fitted successfully"):
        combo.fit(sources=["mpr", "baseline ar(p) model"], variables=["gdpkp"], training_start="2010-03-31")


def test_fit_raises_clear_error_when_combined_forecasts_are_all_nan(monkeypatch, fer_data):
    """Rows with only NaN combined values should be treated as an empty result."""
    monkeypatch.setattr(
        forecast_combo_module,
        "_estimation_loop",
        lambda **kwargs: ([], [{"value": np.nan}]),
    )
    combo = ForecastCombo(forecast_data=fer_data.copy())

    with pytest.raises(ValueError, match="No combined forecasts were fitted successfully"):
        combo.fit(sources=["mpr", "baseline ar(p) model"], variables=["gdpkp"], training_start="2010-03-31")


def test_forecast_combo_fit_with_period_filter(fer_data, unfiltered_baseline):
    """Test that ForecastCombo.fit() works with period_filter for COVID periods and compare with baseline."""
    forecast_data = fer_data.copy()

    # Initialise ForecastCombo with multiple sources and variables
    combo = ForecastCombo(
        forecast_data=forecast_data,
    )

    # Get the frequency from the data to ensure period compatibility
    freq = forecast_data.outturns["frequency"].iloc[0]

    # Create COVID periods (Q2 2020 through Q4 2022) with quarter-end frequency
    from forecast_combo.utils import create_period_filter

    covid_periods = create_period_filter("2020Q1", "2021Q4", freq=freq)

    # Fit the combination model with COVID periods filtered
    last_vintage = np.sort(forecast_data.outturns["vintage_date"].unique())[-1]
    combo.fit(
        sources=["mpr", "baseline ar(p) model"],
        variables=["gdpkp", "cpisa"],
        method=["average", "least_squares", "constrained_least_squares"],
        training_start=last_vintage,
        metric="pop",
        period_filter=covid_periods,
    )

    # Assert that combined forecasts were created
    assert not combo._combined_forecasts.empty

    # Assert that weights were stored
    assert len(combo.weights) > 0

    # Compare with the unfiltered baseline, which this test fits independently
    baseline_weights = unfiltered_baseline["weights"]
    baseline_forecasts = unfiltered_baseline["combined_forecasts"]

    assert not baseline_weights.empty
    assert not baseline_forecasts.empty

    # Compare weights and forecasts
    assert len(combo.weights) == len(baseline_weights), "Number of weight records differs"
    assert len(combo._combined_forecasts) == len(baseline_forecasts), "Number of forecast records differs"

    # check that the estimated weights are different
    assert not combo.weights.equals(baseline_weights), "Weights should differ when COVID periods are filtered"


def test_automatic_labelling_with_period_filter_is_consistent(fer_data):
    """automatic_labelling creates every declared id column, also with a period filter."""
    from forecast_combo.utils import create_period_filter

    forecast_data = fer_data.copy()
    combo = ForecastCombo(forecast_data=forecast_data)

    freq = forecast_data.outturns["frequency"].iloc[0]
    covid_periods = create_period_filter("2020Q1", "2021Q4", freq=freq)
    last_vintage = np.sort(forecast_data.outturns["vintage_date"].unique())[-1]

    sources = ["mpr", "baseline ar(p) model"]
    combo.fit(
        sources=sources,
        variables=["gdpkp"],
        method=["average"],
        training_start=last_vintage,
        metric="pop",
        period_filter=covid_periods,
        window_size=5,
        discount_param=0.9,
        automatic_labelling=True,
    )

    metadata_columns = ["combo_sources", "discount_param", "estimation_window_size", "period_filter"]
    forecasts = combo.forecast_data.forecasts
    combined = forecasts[forecasts["period_filter"].notna()]
    assert not combined.empty

    for column in metadata_columns:
        # Every declared id column exists and carries the same value in both tables.
        # ForecastData stores id columns as strings, so compare on that basis.
        assert column in combo.weights.columns
        assert column in forecasts.columns
        assert set(combined[column].astype(str)) == set(combo.weights[column].astype(str))

    assert combo.weights["combo_sources"].unique().tolist() == [", ".join(sources)]


def test_estimation_loop_uses_available_outturn_maturity_without_lookahead(monkeypatch):
    """Use exact k where available, otherwise the most mature published release."""
    dates = pd.to_datetime(["2020-12-31", "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31"])
    rows = []
    for date_index, date in enumerate(dates):
        for source, scale in (("a", 1.0), ("b", 10.0)):
            maturities = range(4) if date < dates[-1] else [None]
            for maturity in maturities:
                rows.append(
                    {
                        "date": date,
                        "variable": "x",
                        "vintage_date_forecast": date,
                        "horizon": 0,
                        "unique_id": source,
                        "value_forecast": (date_index + 1) * scale,
                        "vintage_date_outturn": (
                            date + pd.offsets.QuarterEnd(maturity + 1) if maturity is not None else pd.NaT
                        ),
                        "outturn_target_minus_vintage": -(maturity + 1) if maturity is not None else np.nan,
                        "value_outturn": date_index * 100 + maturity if maturity is not None else np.nan,
                    }
                )
    merged_data = pd.DataFrame(rows)
    captured = {}

    def fake_get_weights(X, y, method, window_size, discount_param):
        captured["X"] = X.copy()
        captured["y"] = y.copy()
        return np.full(X.shape[1], 1 / X.shape[1]), np.zeros(X.shape[1])

    monkeypatch.setattr(forecast_combo_module, "get_weights", fake_get_weights)

    forecast_combo_module._estimation_loop(
        merged_data=merged_data,
        training_vintages=np.array([pd.Timestamp("2021-12-31")]),
        freq="Q",
        k=3,
        metric="levels",
        period_filter=None,
        sources=["a", "b"],
        methods=["rmse"],
        window_size=None,
        discount_param=1.0,
        print_warning=False,
    )

    # At 2021Q4, 2020Q4 has reached exact k=3. More recent targets fall
    # back to the most mature release published by 2021Q4: k=2, 1, and 0.
    np.testing.assert_array_equal(
        captured["X"],
        np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]),
    )
    np.testing.assert_array_equal(captured["y"], np.array([3.0, 102.0, 201.0, 300.0]))


def test_estimation_loop_supports_final_outturns_without_vintages(monkeypatch):
    """When outturn vintages are unavailable, k remains a no-op."""
    dates = pd.to_datetime(["2021-03-31", "2021-06-30", "2021-09-30"])
    rows = []
    for date_index, date in enumerate(dates, start=1):
        for source, scale in (("a", 1.0), ("b", 10.0)):
            rows.append(
                {
                    "date": date,
                    "variable": "x",
                    "vintage_date_forecast": date,
                    "horizon": 0,
                    "unique_id": source,
                    "value_forecast": date_index * scale,
                    "vintage_date_outturn": pd.NaT,
                    "outturn_target_minus_vintage": -1,
                    "value_outturn": float(date_index),
                }
            )
    captured = {}

    def fake_get_weights(X, y, method, window_size, discount_param):
        captured["y"] = y.copy()
        return np.full(X.shape[1], 1 / X.shape[1]), np.zeros(X.shape[1])

    monkeypatch.setattr(forecast_combo_module, "get_weights", fake_get_weights)

    forecast_combo_module._estimation_loop(
        merged_data=pd.DataFrame(rows),
        training_vintages=np.array([pd.Timestamp("2021-09-30")]),
        freq="Q",
        k=3,
        metric="levels",
        period_filter=None,
        sources=["a", "b"],
        methods=["rmse"],
        window_size=None,
        discount_param=1.0,
        print_warning=False,
    )

    np.testing.assert_array_equal(captured["y"], np.array([1.0, 2.0]))


def test_fit_uses_target_minus_vintage_for_grouping(fer_data):
    """fit() must not require a 'forecast_horizon' column on outturns.

    Outturns only carry the derived 'target_minus_vintage' column; forecasts
    separately carry a forecaster-supplied 'forecast_horizon'. fit() must
    source vintage-distance from 'target_minus_vintage' on both sides.
    """
    forecast_data = fer_data.copy()

    assert "forecast_horizon" not in forecast_data.outturns.columns
    assert "forecast_horizon" in forecast_data.forecasts.columns

    combo = ForecastCombo(forecast_data=forecast_data)
    last_vintage = np.sort(forecast_data.outturns["vintage_date"].unique())[-1]

    combo.fit(
        sources=["mpr", "baseline ar(p) model"],
        variables=["gdpkp"],
        method="average",
        training_start=last_vintage,
        metric="pop",
    )

    assert not combo._combined_forecasts.empty


def test_combined_forecasts_have_horizon_column_not_forecast_horizon(fer_data):
    """The combined-forecasts table exposes 'horizon' (vintage-distance), not 'forecast_horizon'."""
    forecast_data = fer_data.copy()
    combo = ForecastCombo(forecast_data=forecast_data)
    last_vintage = np.sort(forecast_data.outturns["vintage_date"].unique())[-1]

    combo.fit(
        sources=["mpr", "baseline ar(p) model"],
        variables=["gdpkp"],
        method="average",
        training_start=last_vintage,
        metric="pop",
    )

    combined = combo._combined_forecasts
    assert "horizon" in combined.columns
    assert "forecast_horizon" not in combined.columns

    freq = forecast_data.forecasts["frequency"].iloc[0]
    for _, row in combined.iterrows():
        vintage_period = pd.Period(pd.Timestamp(row["vintage_date"]), freq=freq)
        target_period = pd.Period(pd.Timestamp(row["date"]), freq=freq)
        assert (target_period - vintage_period).n == row["horizon"]


def test_estimation_loop_computes_correct_target_period_from_horizon(fer_data):
    """target_period = vintage_period + horizon must reconstruct the forecast date."""
    forecast_data = fer_data.copy()
    combo = ForecastCombo(forecast_data=forecast_data)
    last_vintage = np.sort(forecast_data.outturns["vintage_date"].unique())[-1]

    combo.fit(
        sources=["mpr", "baseline ar(p) model"],
        variables=["gdpkp"],
        method="average",
        training_start=last_vintage,
        metric="pop",
    )

    combined = combo._combined_forecasts
    freq = forecast_data.forecasts["frequency"].iloc[0]
    assert not combined.empty

    for _, row in combined.iterrows():
        vintage_period = pd.Period(pd.Timestamp(row["vintage_date"]), freq=freq)
        expected_date = (vintage_period + row["horizon"]).to_timestamp(how="end").normalize()
        assert pd.Timestamp(row["date"]) == expected_date


def _combo_rows(forecast_data, metric="levels"):
    """Return the combo-written rows from a ForecastData's stored forecasts.

    Filters to a single metric because add_forecasts(compute_levels=True) also
    derives other metrics (e.g. 'pop') from the written 'levels' rows.
    """
    forecasts = forecast_data.forecasts
    return forecasts[(forecasts["type"] == "combo") & (forecasts["metric"] == metric)]


def test_write_back_forecast_horizon_is_information_horizon(monkeypatch):
    """The 'forecast_horizon' persisted via fit()->add_forecasts() is the information horizon.

    Both input forecasts carry a forecaster-supplied 'forecast_horizon' that is
    deliberately different from both the vintage-distance ('horizon') and the
    expected information horizon, so the test cannot pass by accident if the
    write-back regresses to reusing an input source's 'forecast_horizon'.
    """

    def fake_get_weights(X, y, method, window_size, discount_param):
        return np.full(X.shape[1], 1 / X.shape[1]), np.zeros(X.shape[1])

    monkeypatch.setattr(forecast_combo_module, "get_weights", fake_get_weights)

    outturns = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-03-31"),
                "vintage_date": pd.Timestamp("2020-03-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 100.0,
            },
        ]
    )
    # Two forecasts sharing vintage-distance horizon=1: a 2020Q1 forecast (with
    # a matching, already-published outturn usable as training data) and the
    # 2020Q2 forecast being produced now. Both carry forecaster-supplied
    # 'forecast_horizon' values (42, 7) that differ from horizon (1) and from
    # the expected information horizon (0).
    forecasts = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-03-31"),
                "vintage_date": pd.Timestamp("2019-12-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 99.0,
                "forecast_horizon": 42,
                "source": "a",
            },
            {
                "date": pd.Timestamp("2020-03-31"),
                "vintage_date": pd.Timestamp("2019-12-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 101.0,
                "forecast_horizon": 42,
                "source": "b",
            },
            {
                "date": pd.Timestamp("2020-06-30"),
                "vintage_date": pd.Timestamp("2020-03-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 110.0,
                "forecast_horizon": 7,
                "source": "a",
            },
            {
                "date": pd.Timestamp("2020-06-30"),
                "vintage_date": pd.Timestamp("2020-03-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 112.0,
                "forecast_horizon": 7,
                "source": "b",
            },
        ]
    )
    forecast_data = fe.ForecastData(outturns_data=outturns, forecasts_data=forecasts, metric="levels")

    combo = ForecastCombo(forecast_data=forecast_data)
    combo.fit(
        sources=["a", "b"],
        variables=["gdp"],
        method="average",
        training_start="2020-03-31",
        training_end="2020-03-31",
        metric="levels",
        k=0,
    )

    combo_rows = _combo_rows(combo.forecast_data)
    assert len(combo_rows) == 1
    row = combo_rows.iloc[0]
    assert row["date"] == pd.Timestamp("2020-06-30")
    # last training period used is 2020Q1; target period is 2020Q2.
    assert row["forecast_horizon"] == 0
    assert isinstance(row["forecast_horizon"], (int, np.integer))

    combined = combo._combined_forecasts
    assert "forecast_horizon" not in combined.columns
    assert (combined["horizon"] == 1).all()


def test_fit_matches_forecasts_and_outturns_by_date_not_dataframe_index(monkeypatch):
    """Different input DataFrame indices must not affect date-based fitting."""

    def fake_get_weights(X, y, method, window_size, discount_param):
        return np.full(X.shape[1], 1 / X.shape[1]), np.zeros(X.shape[1])

    monkeypatch.setattr(forecast_combo_module, "get_weights", fake_get_weights)

    outturns = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-03-31"),
                "vintage_date": pd.Timestamp("2020-03-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 100.0,
            },
        ],
        index=[101],
    )
    forecasts = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-03-31"),
                "vintage_date": pd.Timestamp("2019-12-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 99.0,
                "forecast_horizon": 42,
                "source": "a",
            },
            {
                "date": pd.Timestamp("2020-03-31"),
                "vintage_date": pd.Timestamp("2019-12-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 101.0,
                "forecast_horizon": 42,
                "source": "b",
            },
            {
                "date": pd.Timestamp("2020-06-30"),
                "vintage_date": pd.Timestamp("2020-03-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 110.0,
                "forecast_horizon": 7,
                "source": "a",
            },
            {
                "date": pd.Timestamp("2020-06-30"),
                "vintage_date": pd.Timestamp("2020-03-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 112.0,
                "forecast_horizon": 7,
                "source": "b",
            },
        ],
        index=[201, 202, 203, 204],
    )
    forecast_data = fe.ForecastData(outturns_data=outturns, forecasts_data=forecasts, metric="levels")

    combo = ForecastCombo(forecast_data=forecast_data)
    combo.fit(
        sources=["a", "b"],
        variables=["gdp"],
        method="average",
        training_start="2020-03-31",
        training_end="2020-03-31",
        metric="levels",
        print_warning=False,
    )

    combined = combo._combined_forecasts
    assert len(combined) == 1
    assert combined.iloc[0]["date"] == pd.Timestamp("2020-06-30")
    assert combined.iloc[0]["value"] == pytest.approx(111.0)


def test_write_back_forecast_horizon_falls_back_to_horizon_when_no_training_data(monkeypatch):
    """Persisted 'forecast_horizon' equals 'horizon' when a group has zero training observations."""

    def fake_get_weights(X, y, method, window_size, discount_param):
        return np.full(X.shape[1], 1 / X.shape[1]), np.zeros(X.shape[1])

    monkeypatch.setattr(forecast_combo_module, "get_weights", fake_get_weights)

    # The only outturn available is for the target itself, so it can't be used
    # as training data (training requires dates strictly before the target).
    outturns = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-06-30"),
                "vintage_date": pd.Timestamp("2020-03-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 99.0,
            },
        ]
    )
    forecasts = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-06-30"),
                "vintage_date": pd.Timestamp("2020-03-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 110.0,
                "forecast_horizon": 77,  # deliberately different from horizon (1)
                "source": "a",
            },
            {
                "date": pd.Timestamp("2020-06-30"),
                "vintage_date": pd.Timestamp("2020-03-31"),
                "variable": "gdp",
                "frequency": "Q",
                "metric": "levels",
                "value": 112.0,
                "forecast_horizon": 77,
                "source": "b",
            },
        ]
    )
    forecast_data = fe.ForecastData(outturns_data=outturns, forecasts_data=forecasts, metric="levels")

    combo = ForecastCombo(forecast_data=forecast_data)
    combo.fit(
        sources=["a", "b"],
        variables=["gdp"],
        method="average",
        training_start="2020-03-31",
        training_end="2020-03-31",
        metric="levels",
        k=0,
    )

    combo_rows = _combo_rows(combo.forecast_data)
    assert len(combo_rows) == 1
    row = combo_rows.iloc[0]
    assert isinstance(row["forecast_horizon"], (int, np.integer))

    combined = combo._combined_forecasts
    assert "forecast_horizon" not in combined.columns
    assert row["forecast_horizon"] == combined["horizon"].iloc[0]


def _partial_source_merged_data():
    """Merged data where source 'b' is never available for variable 'x'."""
    dates = pd.to_datetime(["2021-03-31", "2021-06-30", "2021-09-30"])
    rows = []
    for date_index, date in enumerate(dates, start=1):
        rows.append(
            {
                "date": date,
                "variable": "x",
                "vintage_date_forecast": date,
                "horizon": 0,
                "unique_id": "a",
                "value_forecast": float(date_index),
                "vintage_date_outturn": pd.NaT,
                "outturn_target_minus_vintage": -1,
                "value_outturn": float(date_index),
            }
        )
    return pd.DataFrame(rows)


def _partial_source_missing_at_target_merged_data():
    """Merged data where source 'b' has history but misses the target date."""
    dates = pd.to_datetime(["2021-03-31", "2021-06-30", "2021-09-30"])
    rows = []
    for date_index, date in enumerate(dates, start=1):
        rows.append(
            {
                "date": date,
                "variable": "x",
                "vintage_date_forecast": date,
                "horizon": 0,
                "unique_id": "a",
                "value_forecast": float(date_index),
                "vintage_date_outturn": pd.NaT,
                "outturn_target_minus_vintage": -1,
                "value_outturn": float(date_index),
            }
        )
        if date != dates[-1]:
            rows.append(
                {
                    "date": date,
                    "variable": "x",
                    "vintage_date_forecast": date,
                    "horizon": 0,
                    "unique_id": "b",
                    "value_forecast": float(10 * date_index),
                    "vintage_date_outturn": pd.NaT,
                    "outturn_target_minus_vintage": -1,
                    "value_outturn": float(date_index),
                }
            )
    return pd.DataFrame(rows)


def _run_partial_source_loop(sources=("a", "b"), merged_data=None, **overrides):
    return forecast_combo_module._estimation_loop(
        merged_data=merged_data if merged_data is not None else _partial_source_merged_data(),
        training_vintages=np.array([pd.Timestamp("2021-09-30")]),
        freq="Q",
        k=0,
        metric="levels",
        period_filter=None,
        sources=list(sources),
        methods=["average"],
        window_size=None,
        discount_param=1.0,
        print_warning=False,
        **overrides,
    )


def test_allow_partial_sources_defaults_to_fitting_available_sources():
    """By default a missing source is dropped and the fit proceeds."""
    list_weights, list_combined_forecasts = _run_partial_source_loop()

    assert list_combined_forecasts, "Expected a combined forecast from the available source"
    assert {row["model"] for row in list_weights} == {"a"}


def test_allow_partial_sources_uses_sources_available_at_target():
    """A historical source missing at the target is omitted in partial mode."""
    list_weights, list_combined_forecasts = _run_partial_source_loop(
        merged_data=_partial_source_missing_at_target_merged_data()
    )

    assert len(list_combined_forecasts) == 1
    assert list_combined_forecasts[0]["value"] == pytest.approx(3.0)
    assert {row["model"] for row in list_weights} == {"a"}
    assert list_weights[0]["weight"] == pytest.approx(1.0)


def test_allow_partial_sources_false_raises_for_target_source_gap():
    """Strict mode rejects a source missing only at the target date."""
    with pytest.raises(ValueError, match=r"\['b'\] not available"):
        _run_partial_source_loop(
            merged_data=_partial_source_missing_at_target_merged_data(),
            allow_partial_sources=False,
        )


def test_allow_partial_sources_false_raises_on_missing_source():
    """With allow_partial_sources=False a missing source is an error."""
    with pytest.raises(ValueError, match=r"\['b'\] not available"):
        _run_partial_source_loop(allow_partial_sources=False)


def test_allow_partial_sources_false_allows_complete_fit():
    """Strict mode does not interfere when every requested source is present."""
    _, list_combined_forecasts = _run_partial_source_loop(sources=("a",), allow_partial_sources=False)

    assert list_combined_forecasts


# ---------------------------------------------------------------------------
# ComboSpec tests
# ---------------------------------------------------------------------------


def test_combo_spec_defaults():
    """ComboSpec should have sensible defaults for all optional fields."""
    spec = ComboSpec(name="my_combo", sources=["a", "b"])

    assert spec.name == "my_combo"
    assert spec.sources == ["a", "b"]
    assert spec.method == "average"
    assert spec.training_start is None
    assert spec.training_end is None
    assert spec.metric == "pop"
    assert spec.k == 0
    assert spec.period_filter is None
    assert spec.window_size is None
    assert spec.discount_param == 1.0
    assert spec.allow_partial_sources is True
    assert spec.print_warning is True


def test_combo_spec_custom_values():
    """ComboSpec should store any explicitly supplied values."""
    spec = ComboSpec(
        name="top",
        sources=["stage1", "stage2"],
        method="rmse",
        training_start="2015-01-01",
        window_size=20,
        discount_param=0.9,
    )

    assert spec.method == "rmse"
    assert spec.training_start == "2015-01-01"
    assert spec.window_size == 20
    assert spec.discount_param == 0.9


def test_combo_spec_source_names():
    """source_names should resolve nested ComboSpec objects to their names."""
    child = ComboSpec(name="child", sources=["a", "b"])
    parent = ComboSpec(name="parent", sources=[child, "c"])

    assert parent.source_names == ["child", "c"]


def test_combo_spec_flatten_simple():
    """flatten should return a single spec when there are no nested specs."""
    spec = ComboSpec(name="leaf", sources=["a", "b"])
    flat = spec.flatten_and_validate()
    assert len(flat) == 1
    assert flat[0].name == "leaf"


def test_combo_spec_flatten_nested():
    """flatten should return all nodes in dependency order (leaves first)."""
    child_a = ComboSpec(name="child_a", sources=["x"])
    child_b = ComboSpec(name="child_b", sources=["y"])
    parent = ComboSpec(name="parent", sources=[child_a, child_b, "z"])

    flat = parent.flatten_and_validate()
    names = [s.name for s in flat]
    assert names == ["child_a", "child_b", "parent"]


def test_combo_spec_flatten_deep():
    """flatten should handle three levels of nesting."""
    grandchild = ComboSpec(name="gc", sources=["x"])
    child = ComboSpec(name="child", sources=[grandchild, "y"])
    root = ComboSpec(name="root", sources=[child, "z"])

    flat = root.flatten_and_validate()
    names = [s.name for s in flat]
    assert names == ["gc", "child", "root"]


def test_combo_spec_flatten_deduplicates():
    """flatten should not repeat specs that appear in multiple branches."""
    shared = ComboSpec(name="shared", sources=["x"])
    branch_a = ComboSpec(name="branch_a", sources=[shared, "a"])
    branch_b = ComboSpec(name="branch_b", sources=[shared, "b"])
    root = ComboSpec(name="root", sources=[branch_a, branch_b])

    flat = root.flatten_and_validate()
    names = [s.name for s in flat]
    assert names == ["shared", "branch_a", "branch_b", "root"]


def test_combo_spec_flatten_shared_object_visited_once():
    """A shared child object should be fitted-order listed exactly once."""
    shared = ComboSpec(name="shared", sources=["x"])
    root = ComboSpec(name="root", sources=[shared, shared, "a"])

    names = [s.name for s in root.flatten_and_validate()]
    assert names == ["shared", "root"]


def test_combo_spec_flatten_detects_self_cycle():
    """A spec listing itself as a source should raise a cycle error."""
    node = ComboSpec(name="loop", sources=["x"])
    node.sources.append(node)

    with pytest.raises(ValueError, match="Cycle detected"):
        node.flatten_and_validate()


def test_combo_spec_flatten_detects_mutual_cycle():
    """Two specs referencing each other should raise a cycle error."""
    a = ComboSpec(name="a", sources=["x"])
    b = ComboSpec(name="b", sources=[a])
    a.sources.append(b)

    with pytest.raises(ValueError, match="Cycle detected"):
        a.flatten_and_validate()


def test_combo_spec_flatten_detects_duplicate_names():
    """Distinct specs sharing a name should raise a duplicate-name error."""
    dup_a = ComboSpec(name="dup", sources=["x"])
    dup_b = ComboSpec(name="dup", sources=["y"])
    root = ComboSpec(name="root", sources=[dup_a, dup_b])

    with pytest.raises(ValueError, match="Duplicate ComboSpec name"):
        root.flatten_and_validate()


def test_combo_spec_flatten_rejects_list_method():
    """A list-valued method should raise because output naming is ambiguous."""
    with pytest.raises(TypeError, match="single string"):
        ComboSpec(name="multi", sources=["x", "y"], method=["average", "rmse"])


def test_combo_spec_flatten_detects_raw_source_collision():
    """A node name colliding with a raw source should raise when checked."""
    node = ComboSpec(name="collision", sources=["x", "y"])

    with pytest.raises(ValueError, match="collides with a raw forecast source"):
        node.flatten_and_validate(raw_sources={"collision", "x", "y"})


def test_validate_spec_graph_across_roots():
    """validate_spec_graph should dedupe shared children across roots."""
    shared = ComboSpec(name="shared", sources=["x"])
    root_a = ComboSpec(name="root_a", sources=[shared, "a"])
    root_b = ComboSpec(name="root_b", sources=[shared, "b"])

    names = [s.name for s in forecast_combo_module.validate_spec_graph([root_a, root_b])]
    assert names == ["shared", "root_a", "root_b"]


# ---------------------------------------------------------------------------
# Hierarchical fit tests (new API: fit(sources=ComboSpec))
# ---------------------------------------------------------------------------


def test_fit_drops_nan_forecast_rows_from_the_estimation_panel(fer_data):
    """A NaN introduced after data validation is excluded from fitting."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    forecasts = combo.forecast_data.forecasts
    nan_row = forecasts.index[(forecasts["unique_id"] == "mpr") & (forecasts["variable"] == "gdpkp")][0]
    forecasts.loc[nan_row, "value"] = np.nan

    combo.fit(
        sources=["mpr", "baseline ar(p) model"],
        variables=["gdpkp"],
        method="average",
        training_start=_penultimate_vintage(fer_data),
        print_warning=False,
    )

    assert not combo._combined_forecasts.empty
    assert combo._combined_forecasts["value"].notna().all()


def _penultimate_vintage(forecast_data):
    """Return the single second most-recent vintage date as a string."""
    return str(pd.Timestamp(np.sort(forecast_data.outturns["vintage_date"].unique())[-2]).date())


def test_fit_with_nested_combo_spec(fer_data):
    """fit(sources=ComboSpec) with nested specs should produce all combo sources."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    vintage = _penultimate_vintage(fer_data)

    stage_mpr = ComboSpec(
        name="stage_mpr",
        sources=["mpr", "baseline ar(p) model"],
        method="average",
        training_start=vintage,
    )
    stage_ar = ComboSpec(
        name="stage_ar",
        sources=["mpr", "baseline random walk model"],
        method="average",
        training_start=vintage,
    )
    top = ComboSpec(
        name="top_combo",
        sources=[stage_mpr, stage_ar],
        method="average",
        training_start=vintage,
    )

    combo.fit(sources=top, variables=["gdpkp", "cpisa"])

    sources_in_data = combo.forecast_data.forecasts["source"].unique()

    assert "stage_mpr" in sources_in_data
    assert "stage_ar" in sources_in_data
    assert "top_combo" in sources_in_data
    assert "combo" not in combo.forecast_data.id_columns
    assert "combo_label" not in combo.forecast_data.id_columns

    assert set(combo.weights["combo"].unique()) == {"stage_mpr", "stage_ar", "top_combo"}

    child_weights = combo.weights[combo.weights["combo"].isin(["stage_mpr", "stage_ar"])]
    assert child_weights.groupby("combo")["model"].nunique().to_dict() == {
        "stage_mpr": 2,
        "stage_ar": 2,
    }

    parent_weights = combo.weights[combo.weights["combo"] == "top_combo"]
    assert parent_weights["model"].nunique() == 2

    top_forecasts = combo.forecast_data.forecasts[combo.forecast_data.forecasts["source"] == "top_combo"]
    assert not top_forecasts.empty
    assert top_forecasts["value"].notna().any()


def test_fit_with_combo_spec_ignores_outer_options(fer_data):
    """A single ComboSpec should ignore fit options supplied outside the spec."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    vintage = _penultimate_vintage(fer_data)
    spec = ComboSpec(
        name="single_combo",
        sources=["mpr", "baseline ar(p) model"],
        method="average",
        training_start=vintage,
    )

    combo.fit(sources=spec, variables=["gdpkp"], method="not-a-method")

    assert "single_combo" in combo.forecast_data.forecasts["source"].unique()


def test_fit_with_mixed_combo_spec_validates_outer_options(fer_data):
    """A mixed ComboSpec call should validate options for its top-level fit."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    vintage = _penultimate_vintage(fer_data)
    child = ComboSpec(
        name="mixed_child",
        sources=["mpr", "baseline ar(p) model"],
        method="average",
        training_start=vintage,
    )

    with pytest.raises(ValueError, match="Invalid method"):
        combo.fit(
            sources=[child, "baseline ar(p) model"],
            variables=["gdpkp"],
            method="not-a-method",
        )


def test_fit_with_mixed_sources(fer_data):
    """fit(sources=[ComboSpec, str]) should fit nested specs then combine."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    vintage = _penultimate_vintage(fer_data)

    child = ComboSpec(
        name="child_combo",
        sources=["mpr", "baseline ar(p) model"],
        method="average",
        training_start=vintage,
    )

    combo.fit(
        sources=[child, "baseline ar(p) model"],
        variables=["gdpkp"],
        method="average",
        training_start=vintage,
    )

    sources_in_data = combo.forecast_data.forecasts["source"].unique()
    assert "child_combo" in sources_in_data


def test_fit_combo_spec_uses_name_as_source(fer_data):
    """fit(sources=ComboSpec) must store combos with source=name."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    vintage = _penultimate_vintage(fer_data)

    spec = ComboSpec(
        name="my_named_combo",
        sources=["mpr", "baseline ar(p) model"],
        method="average",
        training_start=vintage,
    )
    combo.fit(sources=spec, variables=["gdpkp"])

    sources = combo.forecast_data.forecasts["source"].unique()
    assert "my_named_combo" in sources
    assert "average" not in sources


def test_multi_method_fit_with_label_sets_source_to_label(fer_data):
    """fit() with multiple methods and a label must always set source=label."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    vintage = _penultimate_vintage(fer_data)

    combo.fit(
        sources=["mpr", "baseline ar(p) model"],
        variables=["gdpkp"],
        method=["average", "rmse"],
        training_start=vintage,
        label="my_combo",
    )

    sources = combo.forecast_data.forecasts["source"].unique()
    assert "my_combo" in sources
    assert "average" not in sources
    assert "rmse" not in sources


def test_fit_combo_spec_returns_self(fer_data):
    """fit(sources=ComboSpec) should return the ForecastCombo instance for chaining."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    vintage = _penultimate_vintage(fer_data)

    spec = ComboSpec(
        name="chained",
        sources=["mpr", "baseline ar(p) model"],
        method="average",
        training_start=vintage,
    )
    result = combo.fit(sources=spec, variables=["gdpkp"])
    assert result is combo


def test_combined_forecasts_carry_method_column(fer_data):
    """Combined forecasts must carry their own 'method' column, disambiguating multi-method fits."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    vintage = _penultimate_vintage(fer_data)

    combo.fit(
        sources=["mpr", "baseline ar(p) model"],
        variables=["gdpkp"],
        method=["average", "rmse"],
        training_start=vintage,
    )

    forecasts = combo.forecast_data.forecasts
    assert "method" in forecasts.columns

    combo_forecasts = forecasts[forecasts["type"] == "combo"]
    assert {"average", "rmse"}.issubset(set(combo_forecasts["method"].unique()))


# ---------------------------------------------------------------------------
# Deprecated fit_hierarchical tests (kept for backwards-compatibility)
# ---------------------------------------------------------------------------


def test_fit_hierarchical_emits_deprecation_warning(fer_data):
    """fit_hierarchical should emit a DeprecationWarning."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    vintage = _penultimate_vintage(fer_data)

    spec = ComboSpec(
        name="dep_test",
        sources=["mpr", "baseline ar(p) model"],
        method="average",
        training_start=vintage,
    )
    with pytest.warns(DeprecationWarning, match="deprecated"):
        combo.fit_hierarchical(specs=[spec], variables=["gdpkp"])


def test_fit_hierarchical_invalid_specs_type(fer_data):
    """Passing something other than a list of ComboSpecs should raise TypeError."""
    combo = ForecastCombo(forecast_data=fer_data.copy())

    with pytest.warns(DeprecationWarning):
        with pytest.raises(TypeError, match="list of ComboSpec"):
            combo.fit_hierarchical(specs={"name": "bad"}, variables=["gdpkp"])

    with pytest.warns(DeprecationWarning):
        with pytest.raises(TypeError, match="list of ComboSpec"):
            combo.fit_hierarchical(specs=["not_a_spec"], variables=["gdpkp"])


# ---------------------------------------------------------------------------
# Core parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"k": -1}, ValueError, "k must be >= 0"),
        ({"k": 1.5}, TypeError, "k must be an integer"),
        ({"k": True}, TypeError, "k must be an integer"),
        ({"window_size": 0}, ValueError, "window_size must be a positive integer"),
        ({"window_size": -5}, ValueError, "window_size must be a positive integer"),
        ({"window_size": 2.5}, TypeError, "window_size must be an integer"),
        ({"discount_param": 0.0}, ValueError, r"discount_param must be in the interval"),
        ({"discount_param": 1.5}, ValueError, r"discount_param must be in the interval"),
        ({"discount_param": -0.5}, ValueError, r"discount_param must be in the interval"),
        ({"discount_param": float("nan")}, ValueError, "discount_param must be finite"),
        ({"discount_param": "1"}, TypeError, "discount_param must be a number"),
    ],
)
def test_fit_rejects_invalid_numeric_params(fer_data, kwargs, exc, match):
    """Invalid k, window_size and discount_param values are rejected."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    with pytest.raises(exc, match=match):
        combo.fit(
            sources=["mpr", "baseline ar(p) model"],
            variables=["gdpkp"],
            training_start=_penultimate_vintage(fer_data),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sources": []}, "sources must not be empty"),
        ({"sources": ["mpr", "mpr"]}, "sources must not contain duplicates"),
        ({"variables": []}, "variables must not be empty"),
        ({"variables": ["gdpkp", "gdpkp"]}, "variables must not contain duplicates"),
        ({"method": []}, "method must not be empty"),
        ({"method": ["average", "average"]}, "method must not contain duplicates"),
    ],
)
def test_fit_rejects_empty_or_duplicate_collections(fer_data, kwargs, match):
    """Empty or duplicated sources, variables and methods are rejected."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    call_kwargs = {
        "sources": ["mpr", "baseline ar(p) model"],
        "variables": ["gdpkp"],
        "training_start": _penultimate_vintage(fer_data),
    }
    call_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=match):
        combo.fit(**call_kwargs)


def test_fit_rejects_period_filter_with_mismatched_frequency(fer_data):
    """A pd.Period whose frequency differs from the data frequency is rejected."""
    combo = ForecastCombo(forecast_data=fer_data.copy())
    with pytest.raises(ValueError, match="does not match the data frequency"):
        combo.fit(
            sources=["mpr", "baseline ar(p) model"],
            variables=["gdpkp"],
            training_start=_penultimate_vintage(fer_data),
            period_filter=[pd.Period("2020-01", freq="M")],
        )
