"""Helpers for building inputs to ``ForecastCombo.fit``."""

import pandas as pd


def create_period_filter(
    start_period: str | pd.Period | pd.Timestamp, end_period: str | pd.Period | pd.Timestamp, freq: str
) -> list[pd.Period]:
    """Return every period from the start through the end point.

    Parameters
    ----------
    start_period : str | pd.Period | pd.Timestamp
        First period in the result.
    end_period : str | pd.Period | pd.Timestamp
        Last period in the result.
    freq : str
        Frequency accepted by :class:`pandas.Period`, such as ``"Q"`` or
        ``"M"``.

    Returns
    -------
    list[pd.Period]
        Periods from ``start_period`` through ``end_period``, inclusive.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If a period is missing, invalid for ``freq``, or out of order.
    """
    if not isinstance(freq, str):
        raise TypeError(f"freq must be a string, got {type(freq).__name__}")
    if not freq.strip():
        raise ValueError("freq must not be empty")
    if not pd.api.types.is_scalar(start_period) or isinstance(start_period, bool):
        raise TypeError("start_period must be a scalar period-like value")
    if not pd.api.types.is_scalar(end_period) or isinstance(end_period, bool):
        raise TypeError("end_period must be a scalar period-like value")
    if pd.isna(start_period):
        raise ValueError("start_period must not be missing")
    if pd.isna(end_period):
        raise ValueError("end_period must not be missing")

    try:
        start_date = pd.Period(start_period, freq=freq)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"start_period must be valid for frequency '{freq}', got {start_period!r}") from exc
    try:
        end_date = pd.Period(end_period, freq=freq)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"end_period must be valid for frequency '{freq}', got {end_period!r}") from exc

    if pd.isna(start_date):
        raise ValueError("start_period must not be missing")
    if pd.isna(end_date):
        raise ValueError("end_period must not be missing")
    if start_date > end_date:
        raise ValueError(f"start_period ({start_date}) must not be after end_period ({end_date})")

    return list(pd.period_range(start=start_date, end=end_date, freq=freq))
