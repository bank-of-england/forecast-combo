"""Usage examples for the ``forecast_combo`` package."""

# %% [markdown]
# #### Load the packages

# %%
import forecast_evaluation as fe

import forecast_combo as fc

# %% [markdown]
# #### Fit a basic combination
#
# Combine model forecasts with equal weights and unconstrained least squares.

# %%
# Load the FER dataset.
forecast_data = fe.ForecastData(load_fer=True)

# Create the combiner.
combo = fc.ForecastCombo(forecast_data=forecast_data)

# Fit two combination methods.
combo.fit(
    sources=["mpr", "baseline ar(p) model"],
    variables=["gdpkp", "cpisa"],
    method=["average", "least_squares"],
    training_start="2020-01-01",
)

# Start a dashboard; this call blocks until the server stops.
# combo.run_forecast_dashboard()
combo.run_combo_dashboard()
