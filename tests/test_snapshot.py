"""Syrupy snapshot tests for the end-to-end combination results.

These tests pin the *numeric output* of ``ForecastCombo.fit`` for every
supported method. The expected values live in
``tests/__snapshots__/test_snapshot.ambr`` and are managed by syrupy, so any
refactor that changes the weights or the combined forecasts fails here even if
every other test still passes.

The input panel is generated deterministically in this file (fixed seed, no
network access and no dependence on the current date), so the snapshots are
reproducible on any machine.

To review and accept an intentional change:

    pytest tests/test_snapshot.py --snapshot-update
"""

import forecast_evaluation as fe
import numpy as np
import pandas as pd
import pytest

from forecast_combo import ForecastCombo

# Round values to keep snapshots stable across BLAS and optimiser builds while
# retaining enough precision to catch substantive result changes.
DECIMALS = 6

METHODS = ["average", "least_squares", "constrained_least_squares", "rmse", "mse", "mae", "huber"]
SOURCES = ["model_a", "model_b", "model_c"]

OUTTURN_QUARTERS = pd.period_range("2010Q1", "2024Q4", freq="Q")
VINTAGE_QUARTERS = pd.period_range("2015Q1", "2024Q4", freq="Q")
HORIZONS = [0, 1]

# The full panel provides enough observations for every method. Snapshot only
# the final vintages to keep the .ambr file readable.
SNAPSHOT_VINTAGES = [
    pd.Timestamp("2024-06-30"),
    pd.Timestamp("2024-09-30"),
    pd.Timestamp("2024-12-31"),
]

# Give each source a distinct bias and noise scale so the weights differ.
SOURCE_PARAMS = {
    "model_a": (0.05, 0.20),
    "model_b": (-0.30, 0.45),
    "model_c": (0.10, 0.80),
}

# Pin the seeded forecast checksum so fixture changes fail directly.
EXPECTED_LEVELS_SUM = 481.94333408904293


def _truth(period: pd.Period) -> float:
    """Deterministic outturn path, independent of the current date."""
    return 2.0 + 0.6 * np.sin(period.ordinal / 4.0) + 0.15 * np.cos(period.ordinal / 1.5)


def build_snapshot_data() -> fe.ForecastData:
    """Build the fixed synthetic panel that the snapshots are computed from."""
    rng = np.random.default_rng(20240101)

    outturns = pd.DataFrame(
        {
            "date": OUTTURN_QUARTERS.to_timestamp(how="end").normalize(),
            "vintage_date": (OUTTURN_QUARTERS + 1).to_timestamp(how="end").normalize(),
            "variable": "gdp",
            "frequency": "Q",
            "value": [_truth(quarter) for quarter in OUTTURN_QUARTERS],
            "metric": "levels",
        }
    )

    rows = []
    for vintage_quarter in VINTAGE_QUARTERS:
        vintage = vintage_quarter.to_timestamp(how="end").normalize()
        for horizon in HORIZONS:
            target = vintage_quarter + horizon
            for source, (bias, scale) in SOURCE_PARAMS.items():
                # Keep the loop order fixed so the seeded panel is reproducible.
                error = bias + scale * rng.standard_normal()
                rows.append(
                    {
                        "date": target.to_timestamp(how="end").normalize(),
                        "vintage_date": vintage,
                        "variable": "gdp",
                        "frequency": "Q",
                        "forecast_horizon": horizon,
                        "value": _truth(target) + error,
                        "metric": "levels",
                        "source": source,
                        "unique_id": source,
                    }
                )
    forecasts = pd.DataFrame(rows)

    return fe.ForecastData(outturns_data=outturns, forecasts_data=forecasts)


def fit_snapshot_combo(discount_param: float = 0.95) -> ForecastCombo:
    """Fit every supported method on the fixed panel."""
    combo = ForecastCombo(forecast_data=build_snapshot_data())
    combo.fit(
        sources=SOURCES,
        variables=["gdp"],
        method=METHODS,
        metric="levels",
        training_start="2015-01-01",
        training_end="2024-12-31",
        window_size=12,
        discount_param=discount_param,
        print_warning=False,
    )
    return combo


def _round(value) -> float:
    number = float(value)
    # Reject NaN values before comparing snapshot summaries.
    assert np.isfinite(number), f"Non-finite result value: {value}"
    return round(number, DECIMALS)


def weights_summary(combo: ForecastCombo) -> dict:
    """Every estimated weight, keyed so the snapshot diff stays readable.

    Shape: ``{method: {"<vintage>|h<horizon>": [w_model_a, w_model_b, w_model_c]}}``.
    Sorting by model makes each list position meaningful and independent of the
    row order produced by ``fit``.
    """
    weights = combo.weights
    weights = weights[weights["vintage_date"].isin(SNAPSHOT_VINTAGES)]
    assert not weights.empty, "No weights for the snapshotted vintages"

    summary = {}
    for method, method_rows in weights.groupby("method"):
        entries = {}
        for (vintage, horizon), group in method_rows.groupby(["vintage_date", "horizon"]):
            key = f"{pd.Timestamp(vintage).date()}|h{int(horizon)}"
            ordered = group.sort_values("model")
            assert ordered["model"].tolist() == SOURCES, "Unexpected model set in weights"
            entries[key] = [_round(weight) for weight in ordered["weight"]]
        summary[str(method)] = entries
    return summary


def combined_forecasts_summary(combo: ForecastCombo) -> dict:
    """Every combined forecast value, keyed by method, vintage, target and horizon."""
    forecasts = combo.forecast_data.forecasts
    combined = forecasts[
        forecasts["source"].isin(METHODS)
        & (forecasts["metric"] == "levels")
        & forecasts["vintage_date"].isin(SNAPSHOT_VINTAGES)
    ]
    assert not combined.empty, "No combined forecasts for the snapshotted vintages"

    summary = {}
    for method, method_rows in combined.groupby("source"):
        entries = {}
        for row in method_rows.sort_values(["vintage_date", "date"]).itertuples():
            vintage = pd.Timestamp(row.vintage_date).date()
            target = pd.Timestamp(row.date).date()
            entries[f"{vintage}|{target}|h{int(row.forecast_horizon)}"] = _round(row.value)
        summary[str(method)] = entries
    return summary


@pytest.fixture(scope="module")
def snapshot_combo():
    return fit_snapshot_combo()


def test_input_panel_is_reproducible():
    """The synthetic panel stays fixed because snapshots depend on it."""
    first = build_snapshot_data().forecasts
    second = build_snapshot_data().forecasts

    pd.testing.assert_frame_equal(first, second)

    # ForecastData derives additional metrics, so count the input rows only.
    levels = first[first["metric"] == "levels"]
    assert len(levels) == len(VINTAGE_QUARTERS) * len(HORIZONS) * len(SOURCES)
    assert float(levels["value"].sum()) == pytest.approx(EXPECTED_LEVELS_SUM, abs=1e-9)


def test_weights_match_snapshot(snapshot_combo, snapshot):
    """Estimated weights for every method must match the stored snapshot."""
    summary = weights_summary(snapshot_combo)

    assert set(summary) == set(METHODS)
    assert summary == snapshot


def test_combined_forecasts_match_snapshot(snapshot_combo, snapshot):
    """Combined forecast values for every method must match the stored snapshot."""
    summary = combined_forecasts_summary(snapshot_combo)

    assert set(summary) == set(METHODS)
    assert summary == snapshot


def test_fit_is_deterministic(snapshot_combo):
    """Refitting the same configuration must reproduce the same numbers."""
    refit = fit_snapshot_combo()

    assert weights_summary(refit) == weights_summary(snapshot_combo)
    assert combined_forecasts_summary(refit) == combined_forecasts_summary(snapshot_combo)


def test_snapshot_is_sensitive_to_the_fit_configuration(snapshot_combo):
    """Guard the guard: a changed configuration must move the pinned values.

    Without this, a summary that rounded or aggregated too coarsely could keep
    matching the snapshot while the underlying numbers moved.
    """
    perturbed = weights_summary(fit_snapshot_combo(discount_param=0.5))
    baseline = weights_summary(snapshot_combo)

    assert perturbed.keys() == baseline.keys()
    for method in ("rmse", "mse", "mae"):
        assert perturbed[method] != baseline[method], f"Discounting did not change '{method}' weights"
    # These methods ignore discounting, so their weights stay unchanged.
    for method in ("average", "least_squares", "constrained_least_squares"):
        assert perturbed[method] == baseline[method], f"Discounting should not change '{method}' weights"


def test_average_snapshot_weights_are_exactly_equal(snapshot_combo):
    """A sanity anchor: the snapshot must encode equal weights for 'average'."""
    average_entries = weights_summary(snapshot_combo)["average"]

    assert average_entries
    for weights in average_entries.values():
        assert weights == pytest.approx([1 / 3] * len(SOURCES), abs=1e-6)
