# Combination Methods

Each method estimates a vector of **weights** `w = (w_1, ..., w_m)` for `m`
model forecasts. The weighted forecasts produce one combined forecast:

```python
import numpy as np

X_forecast = np.array([[1.0, 1.2], [1.1, 1.0]])
weights = np.array([0.4, 0.6])
combined_forecast = X_forecast.dot(weights)
```

Each model forecast is multiplied by its weight, and the products are summed.

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
| [Constrained LS](#constrained-least-squares) | `constrained_least_squares` | `w_i ≥ 0`, `sum = 1` | SLSQP with analytical gradients |

---

## Simple average

```python
import numpy as np

from forecast_combo.combinations import average

X = np.array([[1.0, 1.2], [1.1, 1.0]])
weights = average(X)
```

The method assigns every model the same weight. With three models, each weight
equals `1/3`.

---

## Inverse RMSE

```python
method = "rmse"
```

The method assigns weights in inverse proportion to each model's root mean
squared error over the training sample. Models with smaller errors receive
higher weights.

```python
import numpy as np

from forecast_combo.combinations import rmse_weights

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.0, 1.0])
weights, standard_errors = rmse_weights(X, y, window_size=None, discount_param=0.9)
```

The implementation applies the discount factor `lambda^(T-1-t)`, computes a
discounted RMSE for each model, and normalises the inverse losses. The weights
sum to one, and the function returns delta-method standard errors.

The delta method propagates uncertainty in the RMSE estimates to the normalised
weights:

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

Inverse MSE follows the same scheme as inverse RMSE but omits the square root.
It therefore penalises larger errors more heavily:

```python
import numpy as np

from forecast_combo.combinations import mse_weights

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.0, 1.0])
weights, standard_errors = mse_weights(X, y, window_size=None, discount_param=1.0)
```

The function uses the same delta method for standard errors.

---

## Inverse MAE

```python
method = "mae"
```

The method uses absolute errors instead of squared errors, so a single large
outlier has less influence.

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

The Huber loss is **quadratic for small errors** and **linear for large errors**.
It balances the behavior of MSE and MAE:

```python
import numpy as np

from forecast_combo.combinations import huber_weights

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.0, 1.0])
weights = huber_weights(X, y, window_size=None)
```

The implementation sets `delta = 1.345 sigma` and applies the quadratic or
linear branch according to the error size.

!!! note "No discounting"

    The Huber method ignores `discount_param` and gives every observation in
    the training window equal weight.

---

## Least squares (OLS)

```python
method = "least_squares"
```

Ordinary least squares regresses outturns on forecasts without constraints.
The weights are the OLS coefficients:

```python
import numpy as np

from forecast_combo.combinations import least_squares

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.0, 1.0])
weights, standard_errors = least_squares(X, y)
```

The implementation uses `np.linalg.lstsq`. It returns the minimum-norm solution
when the design is rank deficient. It returns standard errors only when the
design has full rank and more observations than sources; otherwise it returns
`NaN` and emits a warning.

!!! warning "No constraints"

    OLS weights are unconstrained. They can be **negative** and need not sum to
    one. This flexibility can produce extreme or unstable weights when forecasts
    are highly correlated or the training sample is short.

!!! note "Data requirements"

    OLS requires at least as many training observations as models (`T ≥ m`). If
    a vintage and horizon do not meet this condition, the method skips that
    combination and emits a warning.

!!! note "Standard errors are not always identified"

    The weights come from a minimum-norm least-squares solve, which remains
    defined for a rank-deficient design. Standard errors require `T > m` **and**
    a full-rank design; otherwise the function returns `NaN` standard errors
    with a warning.

---

## Constrained least squares

```python
method = "constrained_least_squares"
```

The method minimises the sum of squared errors subject to **non-negativity** and
**sum-to-one** constraints. The weights therefore form a convex combination:

```python
import numpy as np

from forecast_combo.combinations import constrained_least_squares

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.15, 0.95])
weights = constrained_least_squares(X, y)
```

The implementation minimises the least-squares loss with analytical gradients and
SciPy's SLSQP solver. It constrains every weight to be non-negative and makes
the weights sum to one.

!!! tip "When to use constrained LS"

    Constrained LS suits applications that require interpretable, non-negative
    weights that sum to one while the data determines the allocation across
    models. It works best with a moderate number of models and a sufficiently
    large training sample.

!!! warning "No standard errors"

    Constrained least squares returns weights only. The implementation reports
    `NaN` for standard errors because the constraints leave their sampling
    uncertainty unidentified. The same rule applies to `average` and `huber`.
    Scaling by `np.std(y)` also raises `ValueError` when the outturn has zero
    variance in the estimation sample.

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

When set, the method uses only the most recent `window_size` observations. The
weights then respond to recent changes in relative model performance. Every
method applies the same window:

```python
import numpy as np

X = np.array([[1.0, 1.2], [1.1, 1.0], [0.9, 1.1]])
y = np.array([1.0, 1.0, 1.0])
window_size = 2
if window_size is not None and len(y) > window_size:
    X = X[-window_size:]
    y = y[-window_size:]
```

When `window_size = None` (the default), the method uses all available training
data in an expanding window.

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

The method applies exponential discounting to the training observations. Recent
observations receive weights near 1, while older observations receive lower
weights:

```python
import numpy as np

discount_param = 0.9
T = 5
# With discount_param=0.9 and T=5, the factors are:
# discount = [0.9^4, 0.9^3, 0.9^2, 0.9^1, 0.9^0]
#           = [0.656, 0.729, 0.810, 0.900, 1.000]
discount = discount_param ** np.arange(T - 1, -1, -1)
```

When `discount_param = 1` (the default), every observation receives weight 1.

!!! info "Applicability"

    The discount parameter applies to the error-based methods (`rmse`, `mse`,
    `mae`) and not to `least_squares`, `constrained_least_squares`, or `huber`.

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

The filter excludes selected periods from the training sample. Use it to remove
crisis periods, such as the COVID pandemic, that could distort weight estimates.

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

The parameter controls outturn maturity. For a forecast target date $t$, maturity
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

The root specification's `flatten_and_validate()` method resolves the dependency
tree. The combiner fits each node in order, starting with the leaves, and writes
each result to `ForecastData` with `source = spec.name`. Later specifications can
then use those results as input sources.

!!! note "Dependencies resolve automatically"

    Nested `ComboSpec` objects in `sources` determine the fitting order through
    `flatten_and_validate()`. You do not need to order the specifications by hand.

!!! warning "`fit_hierarchical()` is deprecated"

    Earlier versions used `combo.fit_hierarchical()` with an ordered list of
    specifications. This still works but is deprecated;
    pass a nested `ComboSpec` to `fit(sources=...)` instead.
