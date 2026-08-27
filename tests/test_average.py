"""Test average weighting in ``ForecastCombo``."""

import numpy as np
import pytest

import forecast_combo as fc


@pytest.fixture(scope="module")
def average_fit(fer_data):
    """Fit an equal-weight combination on the penultimate vintage."""
    forecast_data = fer_data.copy()

    # Create a combiner with multiple sources and variables.
    combo = fc.ForecastCombo(
        forecast_data=forecast_data,
    )

    penultimate_vintage = sorted(forecast_data.forecasts["vintage_date"].unique())[-2]

    # Fit one training vintage.
    combo.fit(
        sources=["mpr", "baseline ar(p) model"],
        variables=["cpisa"],
        method="average",
        training_start=penultimate_vintage,
        metric="pop",
    )

    return combo, penultimate_vintage


@pytest.mark.parametrize("horizon", [0, 1])
def test_average_weights_gives_average_forecasts(average_fit, horizon):
    """The 'average' source must equal the mean of its inputs, matched by date."""
    combo, penultimate_vintage = average_fit
    variable = "cpisa"
    metric = "pop"

    forecasts = combo.forecast_data.forecasts
    selection = forecasts[
        (forecasts["variable"] == variable)
        & (forecasts["forecast_horizon"] == horizon)
        & (forecasts["metric"] == metric)
        & (forecasts["vintage_date"] == penultimate_vintage)
    ]

    # Align all series by date; row order then has no effect on the comparison.
    columns = ["date", "value"]
    average = selection[selection["source"] == "average"][columns].set_index("date")["value"]
    source1 = selection[selection["source"] == "mpr"][columns].set_index("date")["value"]
    source2 = selection[selection["source"] == "baseline ar(p) model"][columns].set_index("date")["value"]

    assert not average.empty, "No combined forecasts were selected"
    assert not source1.empty, "No 'mpr' forecasts were selected"
    assert not source2.empty, "No 'baseline ar(p) model' forecasts were selected"
    assert average.index.is_unique, "Combined forecasts contain duplicate dates"

    expected_average = ((source1 + source2) / 2).reindex(average.index)

    assert expected_average.notna().all(), "Combined forecast dates are missing from the individual sources"
    assert np.allclose(average.to_numpy(), expected_average.to_numpy(), rtol=1e-6), (
        "Combined forecasts do not match expected average"
    )
