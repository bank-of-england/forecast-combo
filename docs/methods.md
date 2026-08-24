# Combination Methods

This section describes the methods available for combining forecasts. All methods estimate a vector of **weights** `w = (w_1, ..., w_m)` that are applied to `m` model forecasts to produce a single combined forecast. The combined forecast is computed as:

```python
import numpy as np

X_forecast = np.array([[1.0, 1.2], [1.1, 1.0]])
weights = np.array([0.4, 0.6])
combined_forecast = X_forecast.dot(weights)
```

1. Each model's forecast is multiplied by its weight and summed.

---

## Supported methods

| Method | Key | Constraints | Optimisation |
|--------|-----|-------------|--------------|
| [Simple average](#simple-average) | `average` | `w_i = 1/m` | None |
| [Inverse RMSE](#inverse-rmse) | `rmse` | `w_i ≥ 0`, `sum = 1` | Closed-form |
| [Inverse MSE](#inverse-mse) | `mse` | `w_i ≥ 0`, `sum = 1` | Closed-form |
| [Inverse MAE](#inverse-mae) | `mae` | `w_i ≥ 0`, `sum = 1` | Closed-form |
| [Inverse Huber](#inverse-huber-loss) | `huber` | `w_i ≥ 0`, `sum = 1` | Closed-form |
| [Least squares](#least-squares-ols) | `least_squares` | None | OLS |
| [Constrained LS](#constrained-least-squares) | `constrained_least_squares` | `w_i ≥ 0`, `sum = 1` | SLSQP with JAX gradients |

---

## Simple average

```python
import numpy as np

from forecast_combo.combinations import average

X = np.array([[1.0, 1.2], [1.1, 1.0]])
weights = average(X)
```

The simplest combination uses equal weights for all models:

```python
import numpy as np

from forecast_combo.combinations import average

X = np.array([[1.0, 1.2], [1.1, 1.0]])
weights = average(X)
```

If there are 3 models, each gets weight `1/3`.

---

## Inverse RMSE

```python
method = "rmse"
```

Weights are inversely proportional to each model's root mean squared error over the training sample. Models with smaller errors get higher weights.

```python
import numpy as np

from forecast_combo.combinations import rmse_weights

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.0, 1.0])
weights, standard_errors = rmse_weights(X, y, window_size=None, discount_param=0.9)
```

The implementation applies the discount factor `lambda^(T-1-t)`, computes each
model's discounted RMSE, then normalises the inverse losses so the weights sum
to one. It returns both weights and delta-method standard errors.

**Standard errors** of the weights are computed via the delta method, which propagates the uncertainty in the RMSE estimates through to the normalised weights:

```python
import numpy as np

from forecast_combo.combinations.static_combinations import delta_method

rmse = np.array([0.5, 1.0])
variance = np.array([0.01, 0.04])
standard_errors = delta_method(rmse, variance)
```

---

## Inverse MSE

```python
method = "mse"
```

Like inverse RMSE but **without the square root**, so larger errors are penalised more heavily:

```python
import numpy as np

from forecast_combo.combinations import mse_weights

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.0, 1.0])
weights, standard_errors = mse_weights(X, y, window_size=None, discount_param=1.0)
```

Standard errors are computed via the same delta method.

---

## Inverse MAE

```python
method = "mae"
```

The key difference is the use of absolute errors instead of squared errors.
This makes the method less sensitive to a single large outlier.

```python
import numpy as np

from forecast_combo.combinations import mae_weights

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.0, 1.0])
weights, standard_errors = mae_weights(X, y, window_size=None, discount_param=1.0)
```


---

## Inverse Huber loss

```python
method = "huber"
```

Uses a Huber loss function that is **quadratic for small errors** and **linear for large errors**, providing a balance between MSE (efficient) and MAE (robust):

```python
import numpy as np

from forecast_combo.combinations import huber_weights

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.0, 1.0])
weights = huber_weights(X, y, window_size=None)
```

The implementation uses the threshold `delta = 1.345 sigma`, then chooses the
quadratic branch for small errors and the linear branch for large errors.

!!! note "No discounting"

    Unlike RMSE/MSE/MAE, the Huber method does **not** support the `discount_param`. All observations in the training window are weighted equally.

---

## Least squares (OLS)

```python
method = "least_squares"
```

Unconstrained ordinary least squares regression of outturns on forecasts — the weights are just OLS coefficients:

```python
import numpy as np

from forecast_combo.combinations import least_squares

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.0, 1.0])
weights, standard_errors = least_squares(X, y)
```

The implementation uses `np.linalg.lstsq`, returning the minimum-norm solution
when the design is rank deficient. Standard errors are returned when the
design has full rank and more observations than sources; otherwise they are
`NaN` and a warning is emitted.

!!! warning "No constraints"

    OLS weights are unconstrained — they can be **negative** (shorting a model) and do **not** sum to one. This provides the most flexible fit but can lead to extreme or unstable weights, especially when forecasts are highly correlated or the training sample is short.

!!! note "Data requirements"

    OLS requires at least as many training observations as models (`T ≥ m`). If this condition is not met for a given vintage/horizon, the combination is skipped with a warning.

!!! note "Standard errors are not always identified"

    The snippet above is simplified. In the implementation the weights come
    from a minimum-norm least-squares solve, which is well defined even for a
    rank-deficient design. Standard errors require `T > m` **and** a full-rank
    design; otherwise `NaN` standard errors are returned with a warning.

---

## Constrained least squares

```python
method = "constrained_least_squares"
```

Minimises the sum of squared errors subject to **non-negativity** and **sum-to-one** constraints — the weights form a proper convex combination:

```python
import numpy as np

from forecast_combo.combinations import constrained_least_squares

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.15, 0.95])
weights = constrained_least_squares(X, y)
```

The implementation minimises the least-squares loss with JAX gradients and
SciPy's SLSQP solver. It constrains every weight to be non-negative and makes
the weights sum to one.

!!! tip "When to use constrained LS"

    Constrained LS is the recommended choice when you want interpretable, non-negative weights that sum to one while still allowing the data to determine the optimal allocation across models. It is particularly useful when the number of models is moderate and the training sample is sufficiently large.

!!! warning "No standard errors"

    Constrained least squares returns weights only — standard errors are not
    identified for the constrained solution and are reported as `NaN`. The
    same applies to `average` and `huber`. Scaling by `np.std(y)` also means a
    `ValueError` is raised if the outturn has zero variance over the
    estimation sample.

---

## Common parameters

All methods (except `average`) support the following parameters that control the training window:

### Rolling window (`window_size`)

```python
import forecast_evaluation as fe

from forecast_combo import ForecastCombo

forecast_data = fe.ForecastData(load_fer=True)
combo = ForecastCombo(forecast_data)
combo.fit(
    sources=["mpr", "baseline ar(p) model"],
    variables=["cpisa"],
    method="average",
    training_start="2016-01-01",
    window_size=8,
    print_warning=False,
)
```

When set, only the most recent `window_size` observations are used for weight estimation. This makes the weights **adaptive** — they respond to recent changes in relative model performance. The implementation is the same across all methods:

```python
import numpy as np

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.0, 1.0])
window_size = 2
if window_size is not None and len(y) > window_size:
    X = X[-window_size:]
    y = y[-window_size:]
```

When `window_size = None` (default), all available training data is used (expanding window).

### Discount parameter (`discount_param`)

```python
import forecast_evaluation as fe

from forecast_combo import ForecastCombo

forecast_data = fe.ForecastData(load_fer=True)
combo = ForecastCombo(forecast_data)
combo.fit(
    sources=["mpr", "baseline ar(p) model"],
    variables=["cpisa"],
    method="rmse",
    training_start="2016-01-01",
    discount_param=0.9,
    print_warning=False,
)
```

Applies exponential discounting to the training observations. Recent observations get weight close to 1, older observations are down-weighted:

```python
import numpy as np

discount_param = 0.9
T = 5
# With discount_param=0.9 and T=5:
# discount = [0.9^4, 0.9^3, 0.9^2, 0.9^1, 0.9^0]
#           = [0.656, 0.729, 0.810, 0.900, 1.000]
discount = discount_param ** np.arange(T - 1, -1, -1)
```

When `discount_param = 1` (default), all observations are weighted equally (every discount factor is `1^k = 1`).

!!! info "Applicability"

    The discount parameter is currently implemented for the error-based methods (`rmse`, `mse`, `mae`). It is **not** applied to `least_squares`, `constrained_least_squares` or `huber`.

### Period filter (`period_filter`)

```python
import forecast_evaluation as fe

from forecast_combo import ForecastCombo, create_period_filter

covid_filter = create_period_filter("2020Q1", "2021Q4", freq="Q")
forecast_data = fe.ForecastData(load_fer=True)
combo = ForecastCombo(forecast_data)
combo.fit(
    sources=["mpr", "baseline ar(p) model"],
    variables=["cpisa"],
    method="average",
    training_start="2016-01-01",
    period_filter=covid_filter,
    print_warning=False,
)
```

Excludes specific time periods from the training sample. This is useful for removing crisis periods (e.g. the COVID pandemic) that might distort weight estimates.

### Outturn maturity (`k`)

```python
import forecast_evaluation as fe

from forecast_combo import ForecastCombo

forecast_data = fe.ForecastData(load_fer=True)
combo = ForecastCombo(forecast_data)
combo.fit(
    sources=["mpr", "baseline ar(p) model"],
    variables=["cpisa"],
    method="average",
    training_start="2016-01-01",
    k=0,
    print_warning=False,
)
```

Controls outturn maturity. For a forecast target date $t$, requested maturity
`k` corresponds to outturn vintage $t + (k + 1)$ periods, so `k = 0` is the
first post-target release. At each estimation vintage, exact maturity `k` is
used when it has been published. For recent targets that have not reached `k`,
the most mature earlier release available at that estimation vintage is used.
Outturn releases from later vintages are never used.

---

## Hierarchical combinations

The toolkit supports multi-stage combination pipelines using nested `ComboSpec`
objects passed to `fit()`:

```python
import forecast_evaluation as fe

from forecast_combo import ComboSpec

from forecast_combo import ForecastCombo

forecast_data = fe.ForecastData(load_fer=True)

# Stage 1: combine the institutional forecast and AR benchmark
benchmark_combo = ComboSpec(
    name="benchmark_combo",
    sources=["mpr", "baseline ar(p) model"],
    method="average",
)

# Stage 1: combine the two COMPASS variants
compass_combo = ComboSpec(
    name="compass_combo",
    sources=["compass conditional", "compass unconditional"],
    method="average",
)

# Stage 2: combine the stage-1 outputs
top_combo = ComboSpec(
    name="top_combo",
    sources=[benchmark_combo, compass_combo],
    method="rmse",
    window_size=20,
)

combo = ForecastCombo(forecast_data)
combo.fit(
    sources=top_combo,
    variables=["gdpkp", "cpisa"],
)
```

The root spec's `flatten_and_validate()` method resolves the full dependency tree, and each
node is fitted in order (leaves first). The combined forecasts from earlier
specs are written back into the `ForecastData` object with `source = spec.name`,
so downstream specs can reference them as input sources.

!!! note "Dependencies resolve automatically"

    Because nested `ComboSpec` objects are passed directly in `sources`, the
    toolkit works out the correct fitting order via `flatten_and_validate()` — you do not
    need to order the specs manually.

!!! warning "`fit_hierarchical()` is deprecated"

    Earlier versions used `combo.fit_hierarchical()` with an ordered list of
    specifications. This still works but is deprecated;
    pass a nested `ComboSpec` to `fit(sources=...)` instead.
