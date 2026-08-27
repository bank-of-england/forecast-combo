# Forecast Combination Package

A Python package for combining forecasts.


## Installation

Python 3.10 or newer is required.

### Install from PyPI
```sh
pip install forecast-combo[dashboard]
# Omit the extra when you need only the core package.
```

### Install the development version
```sh
git clone https://github.com/bank-of-england/forecast-combo.git
cd forecast-combo
pip install -e .
```

## Quick Start
```python
import forecast_evaluation as fe
import forecast_combo as fc

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

# Start the forecast evaluation dashboard.
combo.run_forecast_dashboard()
```

## Data Classification
Bank of England Data Classification: OFFICIAL BLUE
