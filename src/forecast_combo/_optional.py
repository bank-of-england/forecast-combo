"""Helpers for optional feature dependencies."""

from importlib import import_module


def require_optional_dependency(module_name: str, extra: str, feature: str):
    """Import an optional dependency and explain how to install its feature."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ImportError(
            f"{feature} requires the optional dependency '{module_name}'. "
            f"Install it with `pip install 'forecast-combo[{extra}]'`."
        ) from exc
