import pytest

import forecast_combo._optional as optional


def test_missing_optional_dependency_explains_how_to_install(monkeypatch):
    def raise_missing_dependency(module_name):
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr(optional, "import_module", raise_missing_dependency)

    with pytest.raises(ImportError, match=r"pip install 'forecast-combo\[plots\]'"):
        optional.require_optional_dependency("matplotlib", "plots", "Plotting")
