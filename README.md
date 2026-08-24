# Forecast Combination Package

A Python package for combining forecasts.


## Installation

Python 3.10 or newer is required.

#### Installing from PyPI
```sh
pip install forecast-combo[dashboard]
# Use pip install forecast-combo if you do not need dashboard or plotting support.
```

#### Installing the development version
```sh
git clone https://github.com/bank-of-england/forecast-combo.git
cd forecast-combo
pip install -e .
```

## Quick demo
```python
import forecast_evaluation as fe
import forecast_combo as fc

# Load forecast data
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

# Launch the forecast evaluation dashboard
combo.run_forecast_dashboard()
```