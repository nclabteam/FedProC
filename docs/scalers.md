# Scalers

This project includes feature-wise scalers under `scalers/`.
All scalers operate column-wise (`axis=0`) on NumPy arrays.

In this repository, the primary pattern is to construct scalers from precomputed
client statistics (`self.stats`) rather than calling `fit()` at runtime.

## Available Scalers

| Class | Transform (per feature) | Inverse transform |
| --- | --- | --- |
| `BaseScaler` | `x` | `x` |
| `Standard` | `(x - mean) / std` | `(x * std) + mean` |
| `MinMax` | `(x - min) / (max - min)` | `x * (max - min) + min` |
| `Robust` | `(x - q1) / (q3 - q1)` | `x * (q3 - q1) + q1` |
| `MaxAbs` | `x / max(abs(x))` | `x * max(abs(x))` |

## Common Interface

Each scaler follows the same method pattern:

- `fit(data)`: estimate statistics from training data
- `transform(data)`: scale input data
- `inverse_transform(data)`: map scaled data back to the original space

## Repo-Default Workflow (`Scaler(self.stats)`)

The client pipeline initializes scalers from precomputed train stats:

```python
self.stats = self.private_data["stats"]["train"]
self.scaler = getattr(__import__("scalers"), self.scaler)(self.stats)
```

Then uses:

```python
x_scaled = self.scaler.transform(x)
y_scaled = self.scaler.transform(y)
pred = self.scaler.inverse_transform(pred_scaled)
```

## Optional Workflow (`fit()`)

You can still fit stats directly from arrays when precomputed stats are not available:

```python
scaler.fit(train_x)
train_x_scaled = scaler.transform(train_x)
test_x_scaled = scaler.transform(test_x)
pred_y = scaler.inverse_transform(pred_y_scaled)
```

## BaseScaler

`BaseScaler` is a base class with no-op defaults for `fit`, `transform`, and `inverse_transform`.

Helper utility:

- `divide_no_nan(a, b)`: computes `a / b` and replaces `NaN`/`Inf` with `0.0`

This helper is used by scalers that need safe division.

## Standard

File: `scalers/Standard.py`

### Behavior

- `fit`: computes per-feature `mean` and `std`
- `transform`: `divide_no_nan((x - mean), std)`
- `inverse_transform`: `(x * std) + mean`

### Notes

- If a feature has zero variance (`std == 0`), transformed values become `0` for that feature due to `divide_no_nan`.

### Optional `stat` constructor format

```python
stat = {
	"feature_0": {"mean": 10.2, "std": 3.1},
	"feature_1": {"mean": 5.0, "std": 1.7},
}
```

## MinMax

File: `scalers/MinMax.py`

### Behavior

- `fit`: computes per-feature `min` and `max`
- `transform`: `(x - min) / (max - min)`
- `inverse_transform`: `x * (max - min) + min`

### Notes

- Range is typically `[0, 1]` on data similar to the fit distribution.
- This scaler currently uses direct division; if `max == min` for a feature, division-by-zero can occur.

### Optional `stat` constructor format

```python
stat = {
	"feature_0": {"min": -4.0, "max": 9.0},
	"feature_1": {"min": 0.0, "max": 3.0},
}
```

## Robust

File: `scalers/Robust.py`

### Behavior

- `fit`: computes per-feature quartiles `q1` (25th) and `q3` (75th)
- `transform`: `(x - q1) / (q3 - q1)`
- `inverse_transform`: `x * (q3 - q1) + q1`

### Notes

- More resistant to outliers than mean/std scaling.
- This scaler currently uses direct division; if `q3 == q1` for a feature, division-by-zero can occur.

### Optional `stat` constructor format

```python
stat = {
	"feature_0": {"q1": 2.0, "q3": 8.0},
	"feature_1": {"q1": -1.0, "q3": 1.0},
}
```

## MaxAbs

File: `scalers/MaxAbs.py`

### Behavior

- `fit`: computes per-feature `max_abs = max(abs(x))`
- `transform`: `divide_no_nan(x, max_abs)`
- `inverse_transform`: `x * max_abs`

### Notes

- Preserves sign.
- Useful for data centered around zero.
- If a feature is all zeros, `divide_no_nan` keeps output stable (zeros).

### Optional `stat` constructor format

```python
stat = {
	"feature_0": {"max_abs": 5.0},
	"feature_1": {"max_abs": 12.0},
}
```

## Dynamic Discovery

`scalers/__init__.py` imports scaler classes dynamically and exposes them through `SCALERS` and `__all__`.

Class naming requirement:

- filename and class name must match (for example `MaxAbs.py` -> `class MaxAbs`)