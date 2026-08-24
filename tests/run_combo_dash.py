"""Run the forecast dashboard on sample FER data."""

import forecast_evaluation as fe

import forecast_combo as fc

forecast_data = fe.ForecastData(load_fer=True)

combo = fc.ForecastCombo(forecast_data=forecast_data)
combo.fit(
    sources=["mpr", "bvar unconditional"],
    variables=["gdpkp", "cpisa"],
    method=["average", "least_squares"],
    training_start="2016-01-01",
)

combo.run_combo_dashboard()
