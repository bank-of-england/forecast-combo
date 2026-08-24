import forecast_evaluation as fe
import pytest


@pytest.fixture(scope="session")
def fer_data():
    """Load the FER dataset once and provide copies to tests that fit it."""
    return fe.ForecastData(load_fer=True)
