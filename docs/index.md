# Forecast Combination Toolkit

A Python package for combining forecasts from multiple sources. The project is
developed at the Bank of England and builds on its forecast evaluation package.

---

## Overview

The `forecast-combo` package combines forecasts from different models into one
forecast. It supports methods from simple averaging to constrained optimisation,
and it includes interactive dashboards for comparing combination weights.

The package works with the Bank of England's `forecast_evaluation` and
`forecast_realtime` ecosystems. The workflow runs from model outputs to combined
forecasts and evaluation.

### Key features

- **Combination methods** — average, RMSE, MSE, MAE, Huber, OLS, and constrained least squares
- **Rolling windows and discounting** — estimate weights with a trailing window and exponential discount factors
- **Hierarchical combinations** — pass intermediate combinations into later stages
- **Period filtering** — exclude selected periods, such as the COVID pandemic, from training
- **Interactive dashboards** — inspect combination weights with Shiny and forecast evaluation dashboards
- **Visualisations** — compare weights with line plots, heatmaps, and bar charts

---

## Installation

### 1. Install from PyPI

```bash
pip install forecast-combo[dashboard]
# Use pip install forecast-combo if you do not need dashboard or plotting support.
```

### 2. Fork and clone the repository

```bash
git clone https://github.com/bank-of-england/forecast-combo.git
cd forecast-combo
```

### 3. Set up the development environment

```bash
conda create --name forecast-combo
conda activate forecast-combo
conda install pip
pip install -e ".[dev]"
```

### 4. Install pre-commit hooks

```bash
pre-commit install
```

### 5. Verify the installation

```bash
pytest
```

---

## Dependencies

| Category | Packages |
|----------|----------|
| **Core** | `forecast_evaluation`, `pandas`, `numpy`, `scipy`, `tqdm` |
| **Plots** | `matplotlib` (install with `pip install "forecast-combo[plots]"`) |
| **Dashboard** | `shiny`, `matplotlib` (install with `pip install "forecast-combo[dashboard]"`) |
| **Dev** | `matplotlib`, `shiny`, `pre_commit`, `pytest`, `pytest-cov`, `ruff`, `build`, `pyarrow`, `syrupy` |
| **Docs** | `zensical` (install with `pip install -e ".[docs]"`) |

The plotting and dashboard dependencies are optional for this package. The
current version of `forecast_evaluation` also installs Matplotlib and Shiny
as transitive dependencies, so the core installation may still include them
until that package makes those dependencies optional.

If a plotting function reports a missing dependency, install the plots extra:

```bash
pip install "forecast-combo[plots]"
```

For the interactive dashboard, install the dashboard extra instead:

```bash
pip install "forecast-combo[dashboard]"
```

---

## Quick start

```python
import forecast_evaluation as fe
import forecast_combo as fc

# Load the FER dataset.
forecast_data = fe.ForecastData(load_fer=True)

# Create the combiner.
combo = fc.ForecastCombo(forecast_data=forecast_data)

# Fit two combination methods.
combo.fit(
    sources=["mpr", "bvar unconditional"],
    variables=["gdpkp", "cpisa"],
    method=["average", "constrained_least_squares"],
    training_start="2016-01-01",
)

# Start the forecast evaluation dashboard; this call blocks until the server stops.
combo.run_forecast_dashboard()
```

The repository example script runs the fitting and plotting workflow without
starting a dashboard:

```bash
conda activate forecast-combo
python examples/simple_combo.py
```

---

## Package structure

```
src/forecast_combo/
├── __init__.py              # Public API
├── forecast_combo.py        # Main class + estimation loop
├── utils.py                 # Utility functions
├── combinations/            # Weighting algorithms
│   ├── __init__.py
│   └── static_combinations.py
├── dashboard/               # Shiny interactive dashboard
│   ├── create_app.py
│   ├── ui.py
│   └── tabs/
│       ├── by_horizon.py
│       └── by_vintage.py
└── visualisations/          # Matplotlib plot functions
    ├── _plot_utils.py
    ├── lineplot.py
    ├── heatmap.py
    └── barplot.py
```
