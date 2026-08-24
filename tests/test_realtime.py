"""
Tests for real-time usage of ForecastCombo.

These cover the behaviour required when the combination is run over a real-time
panel, where

* the outturns hold several frequencies (a quarterly target alongside monthly
  indicators), while the forecasts are single-frequency,
* vintages are calendar dates that do not sit on period boundaries,
* the same target period is forecast at several successive vintages,
* stored dates may follow a period-start or a period-end convention.
"""

import forecast_evaluation as fe
import numpy as np
import pandas as pd
import pytest

from forecast_combo import ForecastCombo

OUTTURN_QUARTERS = pd.period_range("2015Q1", "2024Q4", freq="Q")
OUTTURN_MONTHS = pd.period_range("2014-01", "2024-12", freq="M")
FORECAST_QUARTERS = pd.period_range("2018Q1", "2024Q4", freq="Q")

# Two biased sources, so that the combination weights are not degenerate.
BIASES = {"model_a": 0.4, "model_b": -0.2}

# Days after the start of the quarter at which forecasts are produced. The
# second vintage is more accurate, which lets us check that the latest vintage
# is the one used for a given target period.
VINTAGE_OFFSETS = {15: 1.0, 45: 0.5}


def _truth(period: pd.Period) -> float:
    """Deterministic outturn path for the quarterly target."""
    return 1.0 + 0.5 * np.sin(period.ordinal / 3.0)


def _stamp(periods: pd.PeriodIndex, convention: str) -> pd.DatetimeIndex:
    return periods.to_timestamp(how="end").normalize() if convention == "end" else periods.to_timestamp(how="start")


def _build_realtime_data(convention: str = "end") -> fe.ForecastData:
    """Build a small real-time panel with mixed-frequency outturns.

    The monthly indicator is listed first on purpose: reading the frequency
    from the first outturn row would return 'M' even though the combination
    targets a quarterly variable.
    """
    monthly_outturns = pd.DataFrame(
        {
            "date": _stamp(OUTTURN_MONTHS, convention),
            "vintage_date": (OUTTURN_MONTHS + 1).to_timestamp(how="end").normalize(),
            "variable": "monthly_a",
            "frequency": "M",
            "value": np.arange(len(OUTTURN_MONTHS), dtype=float),
            "metric": "levels",
        }
    )
    quarterly_outturns = pd.DataFrame(
        {
            "date": _stamp(OUTTURN_QUARTERS, convention),
            # released 30 days into the following quarter
            "vintage_date": (OUTTURN_QUARTERS + 1).to_timestamp(how="start") + pd.Timedelta(days=30),
            "variable": "quarterly_a",
            "frequency": "Q",
            "value": [_truth(q) for q in OUTTURN_QUARTERS],
            "metric": "levels",
        }
    )
    outturns = pd.concat([monthly_outturns, quarterly_outturns], ignore_index=True)

    rows = []
    for vintage_quarter in FORECAST_QUARTERS:
        for offset, shrink in VINTAGE_OFFSETS.items():
            vintage = vintage_quarter.to_timestamp(how="start") + pd.Timedelta(days=offset)
            for horizon in (0, 1):
                target = vintage_quarter + horizon
                for source, bias in BIASES.items():
                    rows.append(
                        {
                            "date": _stamp(pd.PeriodIndex([target]), convention)[0],
                            "vintage_date": vintage,
                            "variable": "quarterly_a",
                            "frequency": "Q",
                            "forecast_horizon": horizon,
                            "value": _truth(target) + bias * shrink,
                            "metric": "levels",
                            "unique_id": source,
                            "source": source,
                        }
                    )
    forecasts = pd.DataFrame(rows)

    return fe.ForecastData(outturns_data=outturns, forecasts_data=forecasts)


def _fit(convention: str = "end", method: str = "average") -> ForecastCombo:
    combo = ForecastCombo(forecast_data=_build_realtime_data(convention))
    combo.fit(
        sources=list(BIASES),
        variables=["quarterly_a"],
        method=method,
        metric="levels",
        label="combo",
        print_warning=False,
    )
    return combo


def test_frequency_is_taken_from_the_forecasts_not_the_outturns():
    """A quarterly combination must not inherit the frequency of monthly outturns."""
    data = _build_realtime_data()

    # sanity check on the fixture: the first outturn row is monthly
    assert data.outturns["frequency"].iloc[0] == "M"
    assert set(data.outturns["frequency"].unique()) == {"M", "Q"}

    combo = ForecastCombo(forecast_data=data)
    combo.fit(
        sources=list(BIASES),
        variables=["quarterly_a"],
        method="average",
        metric="levels",
        label="combo",
        print_warning=False,
    )

    assert not combo._combined_forecasts.empty
    assert set(combo._combined_forecasts["frequency"].unique()) == {"Q"}
    assert set(combo.weights["frequency"].unique()) == {"Q"}

    # the combined forecasts can be written back into the (quarterly) container,
    # which only accepts forecasts of a single frequency
    assert set(combo.forecast_data.forecasts["frequency"].unique()) == {"Q"}


def test_combined_forecast_dates_match_the_targeted_period():
    """The target period is the vintage period shifted by the horizon."""
    combined = _fit()._combined_forecasts

    vintage_periods = combined["vintage_date"].dt.to_period("Q")
    expected = (vintage_periods + combined["horizon"]).dt.to_timestamp(how="end").dt.normalize()

    assert combined["date"].dt.normalize().equals(expected)

    # forecasts are produced at (almost) every vintage, not only at those whose
    # vintage date happens to fall on a period boundary
    n_vintages = len(FORECAST_QUARTERS) * len(VINTAGE_OFFSETS)
    assert combined["vintage_date"].nunique() >= n_vintages - 3


def test_latest_vintage_is_used_when_a_target_is_forecast_repeatedly():
    """Successive forecasts of the same target must not be averaged together."""
    combined = _fit(method="average")._combined_forecasts

    # 2019-02-15 is the second vintage of 2019Q1; 2019Q1 has already been
    # forecast at the 2019-01-16 vintage (and at both 2018Q4 vintages).
    row = combined[(combined["vintage_date"] == pd.Timestamp("2019-02-15")) & (combined["horizon"] == 0)]
    assert len(row) == 1

    target = pd.Period("2019Q1", freq="Q")
    latest_shrink = VINTAGE_OFFSETS[45]
    expected = _truth(target) + np.mean(list(BIASES.values())) * latest_shrink

    assert row["value"].iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize("method", ["average", "least_squares"])
def test_results_are_invariant_to_the_date_convention(method):
    """Period-start and period-end date conventions describe the same panel."""
    sort_by = ["vintage_date", "horizon", "model"]
    weights_end = _fit("end", method).weights.sort_values(sort_by).reset_index(drop=True)
    weights_start = _fit("start", method).weights.sort_values(sort_by).reset_index(drop=True)

    assert len(weights_end) == len(weights_start)
    assert np.allclose(weights_end["weight"], weights_start["weight"], equal_nan=True)
