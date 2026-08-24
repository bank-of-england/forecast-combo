"""Usage examples for the ``forecast_combo`` package."""

# %% [markdown]
# #### Load forecast-combo and forecast_evaluation packages

# %%
import forecast_evaluation as fe

import forecast_combo as fc

# %% [markdown]
# #### Basic forecast combination
#
# Combine a set of model forecasts with equal weights and with unconstrained
# least squares, then launch the forecast evaluation dashboard.

# %%
# Load FER dataset
forecast_data = fe.ForecastData(load_fer=True)

# Initialise ForecastCombo
combo = fc.ForecastCombo(forecast_data=forecast_data)

# Fit the combination models
combo.fit(
    sources=["mpr", "baseline ar(p) model"],
    variables=["gdpkp", "cpisa"],
    method=["average", "least_squares"],
    training_start="2020-01-01",
)

# Launch dashboards (blocks until the server is stopped)
# combo.run_forecast_dashboard()
combo.run_combo_dashboard()
