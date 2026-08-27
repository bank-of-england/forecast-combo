# API Reference

This script writes the API manifest from each package's `__all__` declaration. Zensical renders the API content from the current source.

## Forecast combination

::: forecast_combo.ForecastCombo
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.ComboSpec
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.get_weights
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.validate_spec_graph
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.SUPPORTED_METHODS
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.create_period_filter
    options:
      show_source: false
      show_root_heading: true

## Weighting methods

::: forecast_combo.combinations.average
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.combinations.least_squares
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.combinations.constrained_least_squares
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.combinations.rmse_weights
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.combinations.mse_weights
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.combinations.mae_weights
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.combinations.huber_weights
    options:
      show_source: false
      show_root_heading: true

## Weight visualisations

::: forecast_combo.visualisations.heatmap.heatmap_by_vintage
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.visualisations.heatmap.heatmap_by_horizon
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.visualisations.lineplot.line_plot_by_vintage
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.visualisations.lineplot.line_plot_by_horizon
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.visualisations.barplot.bar_plot_by_vintage
    options:
      show_source: false
      show_root_heading: true

::: forecast_combo.visualisations.barplot.bar_plot_by_horizon
    options:
      show_source: false
      show_root_heading: true
