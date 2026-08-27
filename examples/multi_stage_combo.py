"""Usage examples for the ``forecast_combo`` package."""

# %% [markdown]
# #### Load the packages

# %%
import forecast_evaluation as fe

import forecast_combo as fc

# Load the FER dataset.
forecast_data = fe.ForecastData(load_fer=True)

# Stage 1a: average the MPR and COMPASS forecasts.
stage_mpr = fc.ComboSpec(
    name="mpr_combo",
    sources=["mpr", "compass unconditional"],
    method="average",
)

# Stage 1b: average the baseline AR and random walk models.
stage_ar = fc.ComboSpec(
    name="ar_combo",
    sources=["baseline ar(p) model", "baseline random walk model"],
    method="average",
)

# Stage 2: combine the stage-1 outputs with RMSE weighting.
# The nested specifications determine the fitting order.
top_combo = fc.ComboSpec(
    name="final_combo",
    sources=[stage_mpr, stage_ar],
    method="rmse",
    window_size=20,
    training_start="2020-01-01",
)

# Create the combiner and fit the hierarchy.
combo = fc.ForecastCombo(forecast_data=forecast_data)
combo.fit(sources=top_combo, variables=["gdpkp", "cpisa"])

# Start a dashboard; this call blocks until the server stops.
combo.run_forecast_dashboard()
# combo.run_combo_dashboard()
