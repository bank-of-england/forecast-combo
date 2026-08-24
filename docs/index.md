# Forecast Combination Toolkit

A Python package for combining forecasts from multiple sources, developed at the Bank of England. It builds on the forecast evaluation package developed by the Bank to evaluate Bank of England's macroeconomic forecasts.

---

## Overview

The `forecast-combo` package provides a flexible framework for **combining forecasts** produced by different models into a single, more accurate composite forecast. It supports a range of combination methods — from simple averaging to optimisation-based approaches — and includes interactive dashboards for evaluating and comparing combination weights.

The package is designed to work with the Bank of England's `forecast_evaluation` and `forecast_realtime` ecosystems, making it easy to go from raw model outputs to combined forecasts and evaluation in a unified workflow.

### Key features

- **Multiple combination methods** — average, RMSE, MSE, MAE, Huber, OLS, and constrained least squares
- **Rolling windows & discounting** — time-varying weight estimation with configurable window sizes and exponential discount factors
- **Hierarchical combinations** — multi-stage combination pipelines where intermediate combos feed into higher-level combos
- **Period filtering** — exclude specific time periods (e.g. COVID) from the training sample
- **Interactive dashboards** — Shiny-based combo dashboard and forecast evaluation dashboard
- **Visualisations** — line plots, heatmaps, and bar charts of combination weights.

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
| **Core** | `forecast_evaluation`, `jax`, `pandas`, `numpy`, `scipy`, `tqdm` |
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

# Load forecast data
forecast_data = fe.ForecastData(load_fer=True)

# Initialise ForecastCombo
combo = fc.ForecastCombo(forecast_data=forecast_data)

# Fit combination models
combo.fit(
    sources=["mpr", "bvar unconditional"],
    variables=["gdpkp", "cpisa"],
    method=["average", "constrained_least_squares"],
    training_start="2016-01-01",
)

# Launch the forecast evaluation dashboard (this blocks until stopped)
combo.run_forecast_dashboard()
```

The repository example script runs the fitting and plotting workflow without
starting a blocking dashboard:

```bash
conda activate ma-forecast-combo
python examples/examples.py
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
