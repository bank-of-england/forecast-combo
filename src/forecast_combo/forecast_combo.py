import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import forecast_combo.combinations as combos
from forecast_combo._validation import (
    validate_discount_param,
    validate_forecast_matrix,
    validate_k,
    validate_window_size,
)

if TYPE_CHECKING:
    pass

SUPPORTED_METHODS = (
    "average",
    "least_squares",
    "constrained_least_squares",
    "rmse",
    "mse",
    "mae",
    "huber",
)


def _combo_config(
    *,
    sources: list[str],
    metric: str,
    k: int,
    period_filter: list | None,
    window_size: int | None,
    discount_param: float,
    training_start,
    training_end,
) -> tuple:
    """Return a hashable tuple describing a fit configuration."""
    periods = tuple(sorted(str(period) for period in period_filter)) if period_filter else ()
    return (
        tuple(sorted(sources)),
        metric,
        k,
        periods,
        window_size,
        discount_param,
        pd.Timestamp(training_start),
        pd.Timestamp(training_end),
    )


def _validate_numeric_params(k: int, window_size: "int | None", discount_param: float) -> None:
    """Validate the scalar numeric fit parameters.

    Parameters
    ----------
    k : int
        Outturn maturity.
    window_size : int | None
        Optional trailing estimation-window length.
    discount_param : float
        Exponential discount factor.
    """
    validate_k(k)
    validate_window_size(window_size)
    validate_discount_param(discount_param)


def _validate_nonempty_unique(values: list[str], name: str) -> None:
    """Check that ``values`` is nonempty and free of duplicates."""
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    duplicates = sorted({v for v in values if values.count(v) > 1})
    if duplicates:
        raise ValueError(f"{name} must not contain duplicates, got repeated entries: {duplicates}")


def _validate_optional_date(value, name: str) -> None:
    """Validate an optional scalar date-like value."""
    if value is None:
        return
    if isinstance(value, (bool, list, tuple, dict, set)):
        raise TypeError(f"{name} must be a scalar date-like value or None")
    try:
        parsed = pd.to_datetime(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date-like value, got {value!r}") from exc
    if not isinstance(parsed, pd.Timestamp) or pd.isna(parsed):
        raise ValueError(f"{name} must be a valid scalar date-like value, got {value!r}")


def _forecast_data_type():
    """Load ForecastData only when a forecast combiner is constructed."""
    from forecast_evaluation import ForecastData

    return ForecastData


def _validate_bool(value, name: str) -> None:
    """Validate a boolean parameter."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")


def _validate_label(label: str | None) -> None:
    """Validate an optional combination label."""
    if label is None:
        return
    if not isinstance(label, str):
        raise TypeError(f"label must be a string or None, got {type(label).__name__}")
    if not label.strip():
        raise ValueError("label must not be empty")


def _validate_dashboard_params(host, port) -> None:
    """Validate common dashboard server arguments."""
    if not isinstance(host, str):
        raise TypeError(f"host must be a string, got {type(host).__name__}")
    if not host.strip():
        raise ValueError("host must not be empty")
    if any(character.isspace() for character in host):
        raise ValueError("host must not contain whitespace")
    if isinstance(port, bool) or not isinstance(port, (int, np.integer)):
        raise TypeError(f"port must be an integer, got {type(port).__name__}")
    if not 1 <= port <= 65535:
        raise ValueError(f"port must be in the range 1 to 65535, got {port}")


def _period_filter_label(period_filter: list | None) -> str | None:
    """Summarise an excluded-period list as a short, deterministic string."""
    if not period_filter:
        return None
    periods = sorted(str(period) for period in period_filter)
    if len(periods) <= 2:
        return ", ".join(periods)
    return f"{periods[0]}, ..., {periods[-1]} ({len(periods)} periods)"


def _default_combo_label(config: tuple) -> str:
    """Build a readable label describing a fit's configuration."""
    sources, metric, k, periods, window_size, discount_param, training_start, training_end = config
    window_label = "all" if window_size is None else str(window_size)
    period_label = ", ".join(periods) if periods else "none"
    return (
        f"sources=[{', '.join(sources)}] metric={metric} k={k} "
        f"training={training_start.date()}..{training_end.date()} "
        f"window={window_label} discount={discount_param:g} excluded={period_label}"
    )


@dataclass
class ComboSpec:
    """Specify one node in a hierarchical forecast combination.

    ``sources`` may contain model-name strings or nested ``ComboSpec``
    objects.

    Attributes
    ----------
    name : str
        Unique name for this combination and its ``source`` label.
    sources : list
        Input sources for this combo.  Can be raw model names *or* other
        ``ComboSpec`` objects for hierarchical combinations.
    method : str
        Single combination method, such as ``"average"`` or ``"rmse"``.
    training_start : str | None
        Start date for the training window (``"YYYY-MM-DD"``). Defaults
        to the earliest available vintage.
    training_end : str | None
        End date for the training window (``"YYYY-MM-DD"``). Defaults to
        the latest available vintage.
    metric : str
        Forecast error metric used for evaluation. Defaults to ``"pop"``.
    k : int
        Outturn maturity. For target date ``t``, selects the outturn vintage
        ``t + (k + 1)`` periods when available, otherwise the most mature
        earlier release available at the estimation vintage. Defaults to 0.
    period_filter : list | None
        Periods to exclude from training (e.g. crisis windows).
    window_size : int | None
        Rolling window size for error-based methods. ``None`` uses all data.
    discount_param : float
        Discount factor for error-based methods. 1.0 means no discounting.
    allow_partial_sources : bool
        Whether to fit with the subset of ``sources`` that is available for a
        given variable/horizon. When ``False``, a missing source raises a
        ``ValueError`` instead. Defaults to ``True``.
    print_warning : bool
        Whether to surface warnings during fitting. Defaults to ``True``.

    Examples
    --------
    >>> bvar_combo = ComboSpec(
    ...     name="bvar_combo",
    ...     sources=["bvar_cv_diff", "bvar_ml_diff", "bvar_ml_levels"],
    ...     method="average",
    ... )
    >>> top_combo = ComboSpec(
    ...     name="top_combo",
    ...     sources=[bvar_combo, "compass"],
    ...     method="rmse",
    ...     window_size=20,
    ... )
    >>> combo.fit(sources=top_combo, variables=["gdpkp", "cpisa"])
    """

    name: str
    sources: list  # list[str | ComboSpec]
    method: str = "average"
    training_start: str | None = None
    training_end: str | None = None
    metric: str = "pop"
    k: int = 0
    period_filter: list | None = None
    window_size: int | None = None
    discount_param: float = 1.0
    allow_partial_sources: bool = True
    print_warning: bool = True

    def __post_init__(self) -> None:
        """Validate the specification at construction time."""
        _validate_combo_spec(self)

    @property
    def source_names(self) -> list[str]:
        """Return source names, resolving nested ``ComboSpec`` objects."""
        return [s.name if isinstance(s, ComboSpec) else s for s in self.sources]

    def flatten_and_validate(self, raw_sources: "set[str] | None" = None) -> list["ComboSpec"]:
        """Return the specification nodes in dependency order.

        Parameters
        ----------
        raw_sources : set[str] | None
            Names of raw forecast sources in the data.  When provided, node
            names are checked for collisions against them.

        Returns
        -------
        list[ComboSpec]
            Nodes in dependency order, with each shared node returned once.
        """
        return validate_spec_graph([self], raw_sources=raw_sources)


def validate_spec_graph(
    roots: "list[ComboSpec]",
    raw_sources: "set[str] | None" = None,
) -> list["ComboSpec"]:
    """Validate a hierarchical ``ComboSpec`` graph and order its nodes.

    Traverses every node reachable from ``roots`` and returns them in
    dependency order (leaves first), with each distinct node appearing
    exactly once even when shared between multiple parents or roots.

    Parameters
    ----------
    roots : list[ComboSpec]
        Root specifications whose graphs should be validated together.
    raw_sources : set[str] | None
        Names of raw forecast sources in the data.  When provided, node
        names are checked for collisions against them.

    Returns
    -------
    list[ComboSpec]
        All nodes in dependency order (leaves first), deduplicated.

    Raises
    ------
    TypeError
        If ``roots`` or ``raw_sources`` has an invalid type.
    ValueError
        If the graph contains a self-cycle or mutual cycle, two distinct
        specs share a name, a node's ``method`` is not a single string, or a
        node name collides with a raw forecast source.
    """
    if not isinstance(roots, list):
        raise TypeError(f"roots must be a list of ComboSpec objects, got {type(roots).__name__}")
    if not roots:
        raise ValueError("roots must not be empty")
    if not all(isinstance(root, ComboSpec) for root in roots):
        raise TypeError("roots must contain only ComboSpec objects")
    if raw_sources is not None:
        if not isinstance(raw_sources, set):
            raise TypeError(f"raw_sources must be a set of strings or None, got {type(raw_sources).__name__}")
        if not all(isinstance(source, str) for source in raw_sources):
            raise TypeError("raw_sources must contain only strings")

    ordered: list[ComboSpec] = []
    # Map name -> spec object to enforce globally unique names.
    names_to_spec: dict[str, ComboSpec] = {}
    # Three-state DFS colouring keyed by object identity: absent = white
    # (unvisited), "grey" = on the current path, "black" = fully processed.
    state: dict[int, str] = {}
    path: list[str] = []

    def visit(node: "ComboSpec") -> None:
        _validate_combo_spec(node)
        node_id = id(node)
        colour = state.get(node_id)
        if colour == "grey":
            cycle = " -> ".join([*path, node.name])
            raise ValueError(f"Cycle detected in ComboSpec graph: {cycle}")
        if colour == "black":
            return

        # Distinct specs must not share a name.
        existing = names_to_spec.get(node.name)
        if existing is not None and existing is not node:
            raise ValueError(f"Duplicate ComboSpec name '{node.name}': distinct specs must have globally unique names.")
        names_to_spec[node.name] = node

        # Node names must not collide with raw forecast source names.
        if raw_sources is not None and node.name in raw_sources:
            raise ValueError(f"ComboSpec name '{node.name}' collides with a raw forecast source of the same name.")

        state[node_id] = "grey"
        path.append(node.name)
        for src in node.sources:
            if isinstance(src, ComboSpec):
                visit(src)
        path.pop()
        state[node_id] = "black"
        ordered.append(node)

    for root in roots:
        visit(root)
    return ordered


def _validate_combo_spec(spec: ComboSpec) -> None:
    """Validate all fields on a ComboSpec, including after mutation."""
    if not isinstance(spec.name, str):
        raise TypeError(f"ComboSpec name must be a string, got {type(spec.name).__name__}")
    if not spec.name.strip():
        raise ValueError("ComboSpec name must not be empty")
    if not isinstance(spec.sources, list):
        raise TypeError(f"ComboSpec '{spec.name}' sources must be a list")
    if not spec.sources:
        raise ValueError(f"ComboSpec '{spec.name}' sources must not be empty")
    if not all(isinstance(source, (str, ComboSpec)) for source in spec.sources):
        raise TypeError(f"ComboSpec '{spec.name}' sources must contain only strings or ComboSpec objects")
    if any(isinstance(source, str) and not source.strip() for source in spec.sources):
        raise ValueError(f"ComboSpec '{spec.name}' source names must not be empty")
    if not isinstance(spec.method, str):
        raise TypeError(f"ComboSpec '{spec.name}' method must be a single string")
    if spec.method not in SUPPORTED_METHODS:
        raise ValueError(
            f"ComboSpec '{spec.name}' has invalid method '{spec.method}'. Supported methods: {SUPPORTED_METHODS}"
        )
    if not isinstance(spec.metric, str):
        raise TypeError(f"ComboSpec '{spec.name}' metric must be a string")
    if not spec.metric.strip():
        raise ValueError(f"ComboSpec '{spec.name}' metric must not be empty")
    _validate_optional_date(spec.training_start, "training_start")
    _validate_optional_date(spec.training_end, "training_end")
    if spec.training_start is not None and spec.training_end is not None:
        if pd.to_datetime(spec.training_start) > pd.to_datetime(spec.training_end):
            raise ValueError("training_start must be before or equal to training_end")
    _validate_numeric_params(spec.k, spec.window_size, spec.discount_param)
    if spec.period_filter is not None and not isinstance(spec.period_filter, (str, list, tuple, np.ndarray, pd.Index)):
        raise TypeError("period_filter must be a string or a list-like collection of periods")
    _validate_bool(spec.allow_partial_sources, "allow_partial_sources")
    _validate_bool(spec.print_warning, "print_warning")


class ForecastCombo:
    """A class for combining and managing forecast data from multiple sources.

    A copy of the input forecast data is stored on the instance, and the
    estimated weights accumulate across all :meth:`fit` calls.

    Attributes
    ----------
    supported_methods : tuple[str, ...]
        The combination methods accepted by :meth:`fit`.

    Parameters
    ----------
    forecast_data : Any
        Forecast data used for fitting. The data is copied before use.

    Raises
    ------
    TypeError
        If ``forecast_data`` is not a :class:`ForecastData` instance.
    """

    supported_methods: tuple[str, ...] = SUPPORTED_METHODS

    def __init__(
        self,
        forecast_data: Any,
    ) -> None:
        if not isinstance(forecast_data, _forecast_data_type()):
            raise TypeError("forecast_data must be an instance of ForecastData")

        self.forecast_data = forecast_data.copy()
        self.weights = pd.DataFrame()  # to store combination weights
        # Maps a fitted ComboSpec name to the unique_id of its output series so
        # that parent specs can reference children by their unique_id.
        self._combo_unique_ids: dict[str, str] = {}
        # Maps every combo_label used so far to its configuration, so that two
        # different fit configurations can never silently share the same label.
        self._combo_labels: dict[str, tuple] = {}

    def fit(
        self,
        sources: list[str] | list[str | ComboSpec] | ComboSpec,
        variables: list[str],
        method: str | list[str] = "average",
        training_start: str | None = None,
        training_end: str | None = None,
        metric: str = "pop",
        k: int = 0,
        period_filter: list[pd.Period] | None = None,
        window_size: int | None = None,
        discount_param: float = 1.0,
        label: str | None = None,
        automatic_labelling: bool = False,
        allow_partial_sources: bool = True,
        print_warning: bool = True,
    ) -> "ForecastCombo":
        """Fit one or more forecast combinations.

        ``sources`` may be a list of source names, a single ``ComboSpec``, or
        a list containing source names and ``ComboSpec`` objects. A
        ``ComboSpec`` supplies the fitting parameters for its own node.

        Parameters
        ----------
        sources : list[str] | list[str | ComboSpec] | ComboSpec
            Forecast sources or hierarchical combination specifications.
        variables : list[str]
            Variables to combine.
        method : str | list[str]
            Combination method or methods. Defaults to ``"average"``.
        training_start : str | None
            First vintage used for training. Defaults to the earliest
            available vintage.
        training_end : str | None
            Last vintage used for training. Defaults to the latest
            available vintage.
        metric : str
            Forecast error metric. Defaults to ``"pop"``.
        k : int
            Outturn maturity. Defaults to ``0``.
        period_filter : list[pd.Period] | None
            Periods excluded from training.
        window_size : int | None
            Number of trailing observations used by applicable methods.
        discount_param : float
            Exponential discount factor for applicable methods. Defaults
            to ``1.0``.
        label : str | None
            Label assigned to the combined forecast source.
        automatic_labelling : bool
            Whether to add fit configuration columns to combined
            forecasts. Defaults to ``False``.
        allow_partial_sources : bool
            Whether to fit using only sources available for a given
            variable and horizon. Defaults to ``True``.
        print_warning : bool
            Whether to display fitting warnings. Defaults to ``True``.

        Returns
        -------
        ForecastCombo
            This object, allowing method chaining.

        Raises
        ------
        TypeError
            If an input has an invalid type.
        ValueError
            If an input is invalid or no combination can be fitted.
        """
        # Variables apply to every fit path, including nested ComboSpecs.
        if isinstance(variables, str):
            variables = [variables]
        if not isinstance(variables, list):
            raise TypeError("variables must be a list of strings")
        if not all(isinstance(variable, str) for variable in variables):
            raise ValueError("All items in variables must be strings")
        _validate_nonempty_unique(variables, "variables")

        if isinstance(sources, ComboSpec):
            return self._fit_from_spec(sources, variables)

        self._validate_fit_options(
            method=method,
            training_start=training_start,
            training_end=training_end,
            metric=metric,
            k=k,
            period_filter=period_filter,
            window_size=window_size,
            discount_param=discount_param,
            label=label,
            automatic_labelling=automatic_labelling,
            allow_partial_sources=allow_partial_sources,
            print_warning=print_warning,
        )

        # --- Handle ComboSpec sources (hierarchical) ----------------------
        if isinstance(sources, list) and any(isinstance(s, ComboSpec) for s in sources):
            if not sources:
                raise ValueError("sources must not be empty")
            if not all(isinstance(source, (str, ComboSpec)) for source in sources):
                raise TypeError("sources must contain only strings or ComboSpec objects")
            return self._fit_from_spec_list(
                sources,
                variables,
                method=method,
                training_start=training_start,
                training_end=training_end,
                metric=metric,
                k=k,
                period_filter=period_filter,
                window_size=window_size,
                discount_param=discount_param,
                label=label,
                automatic_labelling=automatic_labelling,
                allow_partial_sources=allow_partial_sources,
                print_warning=print_warning,
            )

        # --- Plain list[str] path (original behaviour) --------------------
        # Validate sources
        if not isinstance(sources, list):
            raise TypeError("sources must be a list of strings or a ComboSpec")

        if not all(isinstance(source, str) for source in sources):
            raise ValueError("All items in sources must be strings (or use ComboSpec for hierarchical combinations)")

        _validate_nonempty_unique(sources, "sources")

        # Normalise method to list
        if isinstance(method, str):
            method = [method]
        elif not isinstance(method, list):
            raise TypeError("method must be a string or list of strings")
        if not all(isinstance(c, str) for c in method):
            raise ValueError("All items in method must be strings")

        _validate_nonempty_unique(method, "method")

        invalid_methods = set(method) - set(self.supported_methods)
        if invalid_methods:
            raise ValueError(f"Invalid method(s): {invalid_methods}. Supported methods: {self.supported_methods}")

        # Validate that all sources exist in the data
        available_sources = set(self.forecast_data.forecasts["unique_id"].unique())
        invalid_sources = set(sources) - available_sources
        if invalid_sources:
            raise ValueError(f"Invalid sources: {invalid_sources}. Available sources: {available_sources}")

        # Validate that all variables exist in the data
        available_variables = set(self.forecast_data.forecasts["variable"].unique())
        invalid_variables = set(variables) - available_variables
        if invalid_variables:
            raise ValueError(f"Invalid variables: {invalid_variables}. Available variables: {available_variables}")

        # filter forecast_data
        filtered_data = self.forecast_data.copy()
        filtered_data.filter(sources=sources, variables=variables)

        outturns = filtered_data.outturns.copy()
        forecasts = filtered_data.forecasts.copy()

        # filter metric
        if metric in forecasts["metric"].unique():
            forecasts = forecasts[forecasts["metric"] == metric]
            outturns = outturns[outturns["metric"] == metric]
        else:
            raise ValueError(
                f"Metric '{metric}' not found in forecast data. Available metrics: {forecasts['metric'].unique()}"
            )

        outturns = outturns.sort_values("date")

        # Frequency of the target being combined. Outturns may legitimately hold
        # several frequencies, so it is taken from the forecasts, which a
        # ForecastData instance guarantees to be single-frequency. Since
        # 'frequency' is a merge key below, only outturns of this frequency are
        # used for estimation.
        freq = forecasts["frequency"].iloc[0]

        # Merge forecasts and outturns
        # Pre-select only needed columns to reduce memory footprint
        merge_cols = ["date", "variable", "frequency", "metric"]
        forecast_cols = merge_cols + ["vintage_date", "target_minus_vintage", "unique_id", "value"]
        outturn_cols = merge_cols + ["vintage_date", "target_minus_vintage", "value"]

        # "horizon" here is vintage-distance (the old "forecast_horizon" meaning),
        # sourced from the derived "target_minus_vintage" column on both sides.
        forecasts_slim = forecasts[[c for c in forecast_cols if c in forecasts.columns]].rename(
            columns={
                "vintage_date": "vintage_date_forecast",
                "value": "value_forecast",
                "target_minus_vintage": "horizon",
            }
        )
        outturns_slim = outturns[[c for c in outturn_cols if c in outturns.columns]].rename(
            columns={
                "vintage_date": "vintage_date_outturn",
                "target_minus_vintage": "outturn_target_minus_vintage",
                "value": "value_outturn",
            }
        )

        merged_data = pd.merge(
            forecasts_slim,
            outturns_slim,
            on=merge_cols,
            how="left",
        )

        # Validate training period
        if training_start is not None:
            training_start = pd.to_datetime(training_start)
        else:
            training_start = outturns["vintage_date"].min()

        if training_end is not None:
            training_end = pd.to_datetime(training_end)
        else:
            training_end = outturns["vintage_date"].max()

        if training_start > training_end:
            raise ValueError("training_start must be before training_end")

        if training_start < outturns["vintage_date"].min():
            raise ValueError("training_start must be after the earliest date in the data")

        if training_end > outturns["vintage_date"].max():
            raise ValueError("training_end can't be after the latest date in the data")

        # Validate and convert period_filter if provided
        if period_filter is not None:
            # Convert single string to list
            if isinstance(period_filter, str):
                period_filter = [period_filter]

            if not isinstance(period_filter, (list, tuple, np.ndarray, pd.Index)):
                raise TypeError("period_filter must be a list of pandas periods or strings")

            # Convert strings to pd.Period objects
            if len(period_filter) > 0:
                data_freqstr = pd.Period(forecasts["date"].iloc[0], freq=freq).freqstr
                converted_periods = []
                for p in period_filter:
                    if isinstance(p, str):
                        try:
                            converted_periods.append(pd.Period(p, freq=freq))
                        except (TypeError, ValueError) as e:
                            raise ValueError(
                                f"Could not convert '{p}' to Period with frequency '{freq}'. "
                                f"Use format like '2020Q1' for quarterly or '2020M01' for monthly. Error: {e}"
                            )
                    elif isinstance(p, pd.Period):
                        if p.freqstr != data_freqstr:
                            raise ValueError(
                                f"period_filter entry '{p}' has frequency '{p.freqstr}', which does not match "
                                f"the data frequency '{data_freqstr}'."
                            )
                        converted_periods.append(p)
                    else:
                        raise ValueError(f"period_filter items must be strings or pd.Period objects, got {type(p)}")
                period_filter = converted_periods

        # Store results for each combination method and variable
        training_vintages = forecasts["vintage_date"].unique()
        training_vintages = training_vintages[
            (training_vintages >= training_start) & (training_vintages <= training_end)
        ]
        training_vintages = np.sort(training_vintages)

        list_weights, list_combined_forecasts = _estimation_loop(
            merged_data=merged_data,
            training_vintages=training_vintages,
            freq=freq,
            k=k,
            metric=metric,
            period_filter=period_filter,
            sources=sources,
            methods=method,
            window_size=window_size,
            discount_param=discount_param,
            allow_partial_sources=allow_partial_sources,
            print_warning=print_warning,
            label=label,
        )
        # save internal attributes
        self._combined_forecasts = pd.DataFrame(list_combined_forecasts)

        # Discard rows produced by failed fits before checking whether any
        # usable combined forecasts remain.
        if not self._combined_forecasts.empty:
            self._combined_forecasts = self._combined_forecasts.dropna(subset=["value"])

        # give an error if no combination have been fitted successfully
        # (e.g. because there was no overlapping outturns/forecasts for the specified sources/variables)
        if self._combined_forecasts.empty:
            raise ValueError(
                "No combined forecasts were fitted successfully. "
                "This could be because there are no overlapping outturns and "
                "forecasts for the specified sources, variables and vintages."
            )

        config = _combo_config(
            sources=sources,
            metric=metric,
            k=k,
            period_filter=period_filter,
            window_size=window_size,
            discount_param=discount_param,
            training_start=training_start,
            training_end=training_end,
        )
        combo_label = label if label is not None else _default_combo_label(config)

        existing_config = self._combo_labels.get(combo_label)
        if existing_config is not None and existing_config != config:
            raise ValueError(
                f"combo_label '{combo_label}' is already used by a different fit configuration. "
                "Provide a distinct 'label' to disambiguate."
            )
        self._combo_labels[combo_label] = config

        # Configuration metadata, written with identical values to both the
        # weights table and the combined-forecast table.
        combo_metadata = {
            "combo_sources": ", ".join(sources),
            "discount_param": discount_param,
            "estimation_window_size": window_size,
            "period_filter": _period_filter_label(period_filter),
        }

        # save weights dataset
        df_weights = pd.DataFrame(list_weights)
        for column, value in combo_metadata.items():
            df_weights[column] = value
        df_weights["combo_label"] = combo_label

        # add new identification cols
        extra_id = ["type", "method"]
        self._combined_forecasts["type"] = "combo"
        self._combined_forecasts["combo_label"] = combo_label

        if automatic_labelling:
            for column, value in combo_metadata.items():
                # Columns that do not apply to this fit (no window, no period
                # filter) are omitted rather than declared as empty ids.
                if value is None:
                    continue
                extra_id.append(column)
                self._combined_forecasts[column] = value

        # add label to combined forecasts if provided
        if label is not None:
            self._combined_forecasts["combo"] = label
            df_weights["combo"] = label
            df_weights["model"] = df_weights["model"] + " (" + label + ")"

            # Rename the source column to the label so that downstream
            # ComboSpecs can reference this combo by name.
            self._combined_forecasts["source"] = label

        # Write back to ForecastData using the forecaster-supplied information
        # horizon ("forecast_horizon" required by ForecastData), computed
        # per-row in _estimation_loop. The public-facing "_combined_forecasts"
        # table keeps only "horizon" (vintage-distance) to avoid ambiguity.
        forecasts_for_write = self._combined_forecasts.drop(columns=["combo_label", "combo"], errors="ignore").copy()
        forecasts_for_write["forecast_horizon"] = forecasts_for_write["forecast_horizon"].astype(int)
        self.forecast_data.add_forecasts(forecasts_for_write, extra_ids=extra_id, compute_levels=True)
        self._combined_forecasts = self._combined_forecasts.drop(columns=["forecast_horizon"])

        # save weights
        self.weights = pd.concat([self.weights, df_weights], ignore_index=True)
        return self

    def _validate_fit_options(
        self,
        *,
        method: str | list[str],
        training_start,
        training_end,
        metric,
        k,
        period_filter,
        window_size,
        discount_param,
        label,
        automatic_labelling,
        allow_partial_sources,
        print_warning,
    ) -> None:
        """Validate fit options before any forecast data is mutated."""
        methods = [method] if isinstance(method, str) else method
        if not isinstance(methods, list):
            raise TypeError("method must be a string or list of strings")
        if not all(isinstance(value, str) for value in methods):
            raise ValueError("All items in method must be strings")
        _validate_nonempty_unique(methods, "method")
        invalid_methods = set(methods) - set(self.supported_methods)
        if invalid_methods:
            raise ValueError(f"Invalid method(s): {invalid_methods}. Supported methods: {self.supported_methods}")
        if not isinstance(metric, str):
            raise TypeError(f"metric must be a string, got {type(metric).__name__}")
        if not metric.strip():
            raise ValueError("metric must not be empty")
        _validate_optional_date(training_start, "training_start")
        _validate_optional_date(training_end, "training_end")
        if training_start is not None and training_end is not None:
            if pd.to_datetime(training_start) > pd.to_datetime(training_end):
                raise ValueError("training_start must be before or equal to training_end")
        _validate_numeric_params(k=k, window_size=window_size, discount_param=discount_param)
        if period_filter is not None:
            if not isinstance(period_filter, (str, list, tuple, np.ndarray, pd.Index)):
                raise TypeError("period_filter must be a string or a list-like collection of periods")
            values = [period_filter] if isinstance(period_filter, str) else list(period_filter)
            if not all(isinstance(value, (str, pd.Period)) for value in values):
                raise TypeError("period_filter items must be strings or pandas Period objects")
        _validate_label(label)
        _validate_bool(automatic_labelling, "automatic_labelling")
        _validate_bool(allow_partial_sources, "allow_partial_sources")
        _validate_bool(print_warning, "print_warning")

    # ------------------------------------------------------------------
    # Internal helpers for ComboSpec-based fitting
    # ------------------------------------------------------------------

    def _fit_single_spec(self, spec: "ComboSpec", variables: list[str]) -> None:
        """Fit one ComboSpec node (assumes dependencies are already fitted)."""
        self.fit(
            sources=self._resolve_spec_sources(spec.sources),
            variables=variables,
            method=spec.method,
            training_start=spec.training_start,
            training_end=spec.training_end,
            metric=spec.metric,
            k=spec.k,
            period_filter=spec.period_filter,
            window_size=spec.window_size,
            discount_param=spec.discount_param,
            label=spec.name,
            allow_partial_sources=spec.allow_partial_sources,
            print_warning=spec.print_warning,
        )
        self._register_combo_unique_id(spec.name)

    def _resolve_spec_sources(self, sources: list) -> list[str]:
        """Resolve specification sources to forecast ``unique_id`` values."""
        resolved: list[str] = []
        for src in sources:
            if isinstance(src, ComboSpec):
                resolved.append(self._combo_unique_ids.get(src.name, src.name))
            else:
                resolved.append(self._combo_unique_ids.get(src, src))
        return resolved

    def _register_combo_unique_id(self, name: str) -> None:
        """Record the forecast ``unique_id`` for a named combination."""
        forecasts = self.forecast_data.forecasts
        unique_ids = forecasts.loc[forecasts["source"] == name, "unique_id"].unique()
        if len(unique_ids) > 0:
            self._combo_unique_ids[name] = unique_ids[-1]

    @contextmanager
    def _atomic_hierarchical_fit(self):
        """Restore all mutable fit state if a hierarchy fails part-way through."""
        forecast_data = self.forecast_data.copy()
        weights = self.weights.copy()
        combo_unique_ids = self._combo_unique_ids.copy()
        combo_labels = self._combo_labels.copy()
        had_combined_forecasts = hasattr(self, "_combined_forecasts")
        combined_forecasts = self._combined_forecasts.copy() if had_combined_forecasts else None
        try:
            yield
        except Exception:
            self.forecast_data = forecast_data
            self.weights = weights
            self._combo_unique_ids = combo_unique_ids
            self._combo_labels = combo_labels
            if had_combined_forecasts:
                self._combined_forecasts = combined_forecasts
            elif hasattr(self, "_combined_forecasts"):
                del self._combined_forecasts
            raise

    def _fit_from_spec(self, spec: "ComboSpec", variables: list[str]) -> "ForecastCombo":
        """Fit a root specification and its dependency nodes."""
        raw_sources = set(self.forecast_data.forecasts["unique_id"].unique())
        nodes = spec.flatten_and_validate(raw_sources=raw_sources)
        with self._atomic_hierarchical_fit():
            for node in nodes:
                self._fit_single_spec(node, variables)
        return self

    def _fit_from_spec_list(
        self,
        sources: list,
        variables: list[str],
        *,
        method: str | list[str] = "average",
        training_start: str | None = None,
        training_end: str | None = None,
        metric: str = "pop",
        k: int = 0,
        period_filter=None,
        window_size: int | None = None,
        discount_param: float = 1.0,
        label: str | None = None,
        automatic_labelling: bool = False,
        allow_partial_sources: bool = True,
        print_warning: bool = True,
    ) -> "ForecastCombo":
        """Fit nested specifications and a mixed top-level source list."""
        # Validate the full graph of nested specs together and fit each node
        # exactly once (shared children are not re-fitted across roots).
        raw_sources = set(self.forecast_data.forecasts["unique_id"].unique())
        spec_roots = [src for src in sources if isinstance(src, ComboSpec)]
        nodes = validate_spec_graph(spec_roots, raw_sources=raw_sources)
        with self._atomic_hierarchical_fit():
            for node in nodes:
                self._fit_single_spec(node, variables)

            # Resolve the top-level source names, using fitted combo unique_ids
            # for nested specs and plain strings otherwise.
            resolved_names = self._resolve_spec_sources(sources)

            # Now fit the top-level combination with resolved string names
            return self.fit(
                sources=resolved_names,
                variables=variables,
                method=method,
                training_start=training_start,
                training_end=training_end,
                metric=metric,
                k=k,
                period_filter=period_filter,
                window_size=window_size,
                discount_param=discount_param,
                label=label,
                automatic_labelling=automatic_labelling,
                allow_partial_sources=allow_partial_sources,
                print_warning=print_warning,
            )

    def fit_hierarchical(
        self,
        specs: "list[ComboSpec]",
        variables: list[str],
    ) -> "ForecastCombo":
        """Fit an ordered list of hierarchical combination specifications.

        .. deprecated::
            Use :meth:`fit` with a :class:`ComboSpec` as ``sources`` instead.
            ``fit_hierarchical`` will be removed in a future release.

        Parameters
        ----------
        specs : list[ComboSpec]
            Ordered list of combination nodes.
        variables : list[str]
            Variables to combine.

        Returns
        -------
        ForecastCombo
            Returns ``self`` to allow method chaining.

        Raises
        ------
        TypeError
            If ``specs`` or ``variables`` has an invalid type.
        ValueError
            If ``specs`` or ``variables`` is empty, or a specification is invalid.
        """
        warnings.warn(
            "fit_hierarchical() is deprecated. Pass a ComboSpec to fit(sources=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(specs, list) or not all(isinstance(s, ComboSpec) for s in specs):
            raise TypeError("specs must be a list of ComboSpec objects")
        if not specs:
            raise ValueError("specs must not be empty")
        if isinstance(variables, str):
            variables = [variables]
        if not isinstance(variables, list) or not all(isinstance(variable, str) for variable in variables):
            raise TypeError("variables must be a list of strings")
        _validate_nonempty_unique(variables, "variables")

        with self._atomic_hierarchical_fit():
            for spec in specs:
                _validate_combo_spec(spec)
                self._fit_single_spec(spec, variables)

        return self

    def run_combo_dashboard(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the Shiny combo dashboard on the fitted combination weights.

        Parameters
        ----------
        host : str
            Host address for the dashboard server. Default is "127.0.0.1".
        port : int
            Port number for the dashboard server. Default is 8000.
        """
        _validate_dashboard_params(host, port)

        from forecast_combo.dashboard.create_app import dashboard_app

        app = dashboard_app(self)
        app.run(host=host, port=port)

    def run_forecast_dashboard(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the forecast_evaluation dashboard on the original and combined forecasts.

        Parameters
        ----------
        host : str
            Host address for the dashboard server. Default is "127.0.0.1".
        port : int
            Port number for the dashboard server. Default is 8000.
        """

        _validate_dashboard_params(host, port)
        from forecast_combo._optional import require_optional_dependency

        require_optional_dependency("matplotlib", "dashboard", "The forecast dashboard")
        require_optional_dependency("shiny", "dashboard", "The forecast dashboard")
        self.forecast_data.run_dashboard(host=host, port=port)


def get_weights(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    window_size: int | None = None,
    discount_param: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate forecast combination weights.

    Parameters
    ----------
    X : np.ndarray
        Forecast matrix with rows as observations and columns as sources.
    y : np.ndarray
        Target values.
    method : str
        Combination method. Must be one of :data:`SUPPORTED_METHODS`.
    window_size : int | None
        Number of trailing observations used by applicable methods.
        Defaults to ``None``.
    discount_param : float
        Exponential discount factor for applicable methods. Defaults to
        ``1.0``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Estimated weights and standard errors, with one entry per source.
        Methods without estimated uncertainty return ``NaN`` standard
        errors.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If an input is invalid or the requested method cannot be fitted.
    """
    validate_forecast_matrix(X, y)
    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got {type(method).__name__}")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown combination method '{method}'. Supported methods: {', '.join(SUPPORTED_METHODS)}.")
    validate_window_size(window_size)
    validate_discount_param(discount_param)

    n_sources = X.shape[1]

    if method == "average":
        weights = combos.average(X)
        std_err = np.full(n_sources, np.nan)
    elif method == "least_squares":
        weights, std_err = combos.least_squares(X, y, window_size)
    elif method == "constrained_least_squares":
        weights = combos.constrained_least_squares(X, y, window_size)
        std_err = np.full(n_sources, np.nan)
    elif method == "rmse":
        weights, std_err = combos.rmse_weights(X, y, window_size, discount_param)
    elif method == "mse":
        weights, std_err = combos.mse_weights(X, y, window_size, discount_param)
    elif method == "mae":
        weights, std_err = combos.mae_weights(X, y, window_size, discount_param)
    elif method == "huber":
        weights = combos.huber_weights(X, y, window_size)
        std_err = np.full(n_sources, np.nan)
    return weights, std_err


def _estimation_loop(
    merged_data: pd.DataFrame,
    training_vintages: np.ndarray,
    freq: str,
    k: int,
    metric: str,
    period_filter: list[pd.Period] | None,
    sources: list[str],
    methods: list[str],
    window_size: int | None,
    discount_param: float,
    print_warning: bool,
    label: str | None = None,
    allow_partial_sources: bool = True,
) -> tuple[list, list]:
    """Fit combination models for the supplied training vintages.

    Parameters
    ----------
    merged_data : pd.DataFrame
        Merged forecasts and outturns.
    training_vintages : np.ndarray
        Vintage dates used for training.
    freq : str
        Data frequency.
    k : int
        Requested outturn maturity.
    metric : str
        Forecast error metric.
    period_filter : list[pd.Period] | None
        Periods excluded from training.
    sources : list[str]
        Forecast sources to combine.
    methods : list[str]
        Combination methods to fit.
    window_size : int | None
        Number of trailing observations used by applicable methods.
    discount_param : float
        Exponential discount factor for applicable methods.
    print_warning : bool
        Whether to display fitting warnings.
    label : str | None
        Optional progress-bar label.
    allow_partial_sources : bool
        Whether to fit using only available sources.

    Returns
    -------
    tuple[list, list]
        Lists of weight records and combined-forecast records.

    Raises
    ------
    ValueError
        If a fitting operation cannot produce a valid combination.
    """
    list_weights = []
    list_combined_forecasts = []
    warned_missing: set[tuple] = set()
    warned_no_forecasts: set[tuple] = set()
    warned_single_source: set[tuple] = set()

    desc = f"Fitting '{label}'" if label else "Fitting combination models"
    for vintage in tqdm(training_vintages, desc=desc):
        vintage_data = merged_data[merged_data["vintage_date_forecast"] <= vintage].copy()

        grouped_vintage = vintage_data.groupby(["variable", "horizon"])

        for (variable, horizon), group in grouped_vintage:
            target_period = pd.Period(pd.Timestamp(vintage), freq=freq)
            target_period = target_period + horizon
            forecast_date = target_period.to_timestamp(how="end").normalize()

            # Get forecasts, keeping the latest vintage available at this
            # estimation vintage for each (date, source).
            latest_forecasts = group.sort_values("vintage_date_forecast", ascending=False).drop_duplicates(
                subset=["date", "unique_id"], keep="first"
            )
            X = latest_forecasts.pivot(index="date", columns="unique_id", values="value_forecast")

            if group["vintage_date_outturn"].notna().any():
                # Use exact maturity k when it has been published by this
                # estimation vintage. For recent targets that have not yet
                # reached k, use the most mature published release below k.
                outturns_available = group[
                    group["vintage_date_outturn"].notna() & (group["vintage_date_outturn"] <= vintage)
                ][["date", "vintage_date_outturn", "outturn_target_minus_vintage", "value_outturn"]].drop_duplicates()
                outturns_available["outturn_maturity"] = -outturns_available["outturn_target_minus_vintage"] - 1
                outturns_available = outturns_available[outturns_available["outturn_maturity"] <= k]
                y = (
                    outturns_available.sort_values(["date", "outturn_maturity", "vintage_date_outturn"])
                    .drop_duplicates("date", keep="last")
                    .set_index("date")["value_outturn"]
                    .dropna()
                )
            else:
                # No vintage dimension: there is one final outturn per target,
                # so k has no effect (matching ForecastData.filter_k semantics).
                y = group[["date", "value_outturn"]].drop_duplicates().set_index("date")["value_outturn"].dropna()

            # Filter to the requested sources available for the target row.
            # A source can have historical forecasts while being unavailable
            # for the target, so checking only the pivot columns is not enough.
            target_mask = X.index.to_period(freq) == target_period
            if target_mask.any():
                target_values = X.loc[target_mask].iloc[0]
                available_sources = [
                    source for source in sources if source in X.columns and pd.notna(target_values[source])
                ]
                missing_sources = [source for source in sources if source not in available_sources]
            else:
                # Preserve the no-target-forecast path below, which emits a
                # NaN forecast after training on the available history.
                available_sources = [source for source in sources if source in X.columns]
                missing_sources = [source for source in sources if source not in X.columns]

            if missing_sources and not allow_partial_sources:
                raise ValueError(
                    f"Sources {missing_sources} not available for variable '{variable}', "
                    f"horizon {horizon}, vintage {vintage}. "
                    "Set allow_partial_sources=True to fit with the available sources."
                )

            if len(available_sources) == 0:
                continue

            missing_key = (variable, horizon, tuple(missing_sources))
            if missing_sources and print_warning and missing_key not in warned_missing:
                warned_missing.add(missing_key)
                warnings.warn(
                    f"Sources {missing_sources} not available for "
                    f"variable '{variable}', horizon {horizon}. "
                    f"Fitting with remaining sources."
                )

            # Order the rows; important for storing the weights correctly
            X = X[available_sources]
            # drop rows with missing values and ensure correct order of columns
            # missing values will occur if the sources don't have the same number
            # of forecasts
            X = X.dropna()
            y = y.dropna()

            # If there are no outturns for this vintage/horizon there is
            # nothing to train on and no combined forecast to produce.
            if y.empty:
                continue

            # training indices
            X_index = X.index
            y_index = y.index

            training_indices = X_index.intersection(y_index)
            # Compare in period space: stored dates need not sit exactly on the
            # normalised period end, so timestamp comparison is not reliable.
            training_indices = training_indices[training_indices.to_period(freq) < target_period]

            # filter training indices based on period_filter if provided
            if period_filter is not None:
                training_indices = training_indices[~training_indices.to_period(freq).isin(period_filter)]

            # Information horizon written back to ForecastData: the distance
            # between the target period and the most recent training
            # observation actually used. Falls back to the vintage-distance
            # "horizon" when no training observations were available.
            if len(training_indices) > 0:
                last_training_period = training_indices.to_period(freq).max()
                forecast_horizon = (target_period - last_training_period).n - 1
            else:
                forecast_horizon = horizon

            # get training data (data before the vintage date or last available obs)
            X_train = X[X_index.isin(training_indices)]
            y_train = y[y_index.isin(training_indices)]

            # convert to numpy
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy()

            # get the data for producing the forecast (as opposed to training)
            forecast_row = X_index.to_period(freq) == target_period
            if forecast_row.any():
                X_forecast = X[forecast_row].to_numpy()
            else:
                X_forecast = np.zeros((1, X.shape[1])) + np.nan

                no_fc_key = (variable, horizon)
                if print_warning and no_fc_key not in warned_no_forecasts:
                    warned_no_forecasts.add(no_fc_key)
                    warnings.warn(
                        f"Don't have any forecasts for "
                        f"variable '{variable}', horizon {horizon}. "
                        "Filling with NaN values."
                    )

            if X_train.shape[1] == 1:
                single_key = (variable, horizon)
                if print_warning and single_key not in warned_single_source:
                    warned_single_source.add(single_key)
                    warnings.warn(f"Only 1 source available for variable '{variable}', horizon {horizon}.")

            # Fit each combination method
            for method in methods:
                # Effective sample size after the rolling window has been applied,
                # since the window is what actually determines the estimation sample.
                n_train = X_train.shape[0]
                if window_size is not None:
                    n_train = min(n_train, window_size)
                n_sources_train = X_train.shape[1]

                # Skip if we don't have enough data
                insufficient_least_squares = (
                    method in ["least_squares", "constrained_least_squares"] and n_train < n_sources_train
                )
                insufficient_huber = method == "huber" and n_train < 2
                if insufficient_least_squares or insufficient_huber or (method != "average" and n_train == 0):
                    if print_warning:
                        warnings.warn(
                            f"Not enough data to fit '{method}' for vintage {vintage}, "
                            f"variable {variable}, horizon {horizon}. "
                        )
                        if method in ["least_squares", "constrained_least_squares"]:
                            warnings.warn(f"Need at least {n_sources_train} obs for OLS, have {n_train}.")
                        elif method == "huber":
                            warnings.warn(f"Need at least 2 observations for Huber weights, have {n_train}.")
                    skip_training = True
                    weights = np.full(X_train.shape[1], np.nan)
                    std_error = np.full(X_train.shape[1], np.nan)
                    X_forecast_method = np.zeros((1, X.shape[1])) + np.nan
                else:
                    skip_training = False
                    X_forecast_method = X_forecast

                if not skip_training:
                    # Estimate weights using the training data
                    weights, std_error = get_weights(
                        X=X_train,
                        y=y_train,
                        method=method,
                        window_size=window_size,
                        discount_param=discount_param,
                    )
                # compute combination based on estimated weights
                combined_forecast = X_forecast_method.dot(weights)[0]

                # Create one row per model source for the weights dataframe.
                # Weight uncertainty is reported here only: it is a per-source
                # quantity, so it has no scalar counterpart on a combined
                # forecast row.
                for model_name, weight, se in zip(available_sources, weights, std_error):
                    list_weights.append(
                        {
                            "date": forecast_date,
                            "vintage_date": vintage,
                            "variable": variable,
                            "frequency": freq,
                            "method": method,
                            "metric": metric,
                            "horizon": horizon,
                            "model": model_name,
                            "weight": weight,
                            "std_error": se,
                        }
                    )

                list_combined_forecasts.append(
                    {
                        "date": forecast_date,
                        "vintage_date": vintage,  # use the latest vintage date in the training data
                        "variable": variable,
                        "frequency": freq,
                        "source": method,
                        "method": method,
                        "metric": metric,
                        "horizon": horizon,
                        "forecast_horizon": forecast_horizon,
                        "value": combined_forecast,
                    }
                )

    return list_weights, list_combined_forecasts


def example_usage():
    from forecast_evaluation import ForecastData

    # Load forecast data
    forecast_data = ForecastData(load_fer=True)

    # Initialise ForecastCombo
    combo = ForecastCombo(
        forecast_data=forecast_data,
    )

    # Fit the combination model with training period
    combo.fit(
        sources=["mpr", "bvar unconditional"],
        variables=["gdpkp", "cpisa"],
        method=["average", "least_squares", "constrained_least_squares"],
        training_start="2016-01-01",
    )

    # launch dashboard
    combo.run_forecast_dashboard()
    # combo.forecast_data.run_dashboard()


if __name__ == "__main__":
    example_usage()
