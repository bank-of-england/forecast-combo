"""Usage examples for the ``forecast_combo`` package."""

# %% [markdown]
# #### Load forecast-combo and forecast_evaluation packages

# %%
import forecast_evaluation as fe

import forecast_combo as fc

# Load FER dataset
forecast_data = fe.ForecastData(load_fer=True)

# Stage 1a: average the MPR and COMPASS forecasts
stage_mpr = fc.ComboSpec(
    name="mpr_combo",
    sources=["mpr", "compass unconditional"],
    method="average",
)

# Stage 1b: average the baseline AR and random walk models
stage_ar = fc.ComboSpec(
    name="ar_combo",
    sources=["baseline ar(p) model", "baseline random walk model"],
    method="average",
)

# Stage 2: combine the two stage-1 outputs using RMSE weighting.
# Nest the stage-1 specs directly so their outputs are resolved automatically.
top_combo = fc.ComboSpec(
    name="final_combo",
    sources=[stage_mpr, stage_ar],
    method="rmse",
    window_size=20,
    training_start="2020-01-01",
)

# Initialise ForecastCombo and fit the multi-stage hierarchy
combo = fc.ForecastCombo(forecast_data=forecast_data)
combo.fit(sources=top_combo, variables=["gdpkp", "cpisa"])

# Launch dashboards (blocks until the server is stopped)
combo.run_forecast_dashboard()
# combo.run_combo_dashboard()
