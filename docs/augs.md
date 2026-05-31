# Time-Series Augmentations

FedProC provides a dedicated, high-performance, GPU-native time-series augmentation library under the `augs` package. Built entirely in pure PyTorch, these augmentations eliminate CPU-GPU round-trip latency bottlenecks, enabling ultra-fast, learnable differentiable data augmentation directly on GPU memory.

These augmentations are fully compatible with batch inputs of shape `[B, T, D]` (Batch size, Time length, Dimensions/Features) and adapt automatically to the active device of the input tensor.

---

## Augmentations API Reference

All augmentations are implemented as callable classes.

### 1. Cutout
Stochastically zeroes out a single contiguous temporal segment within the time-series.
* **Class**: `cutout(perc=0.1)`
* **Parameters**:
  * `perc` (float): The percentage of the sequence length to zero out.

```python
from augs import cutout

aug = cutout(perc=0.15)
augmented_x = aug(x)  # x is [B, T, D]
```

---

### 2. Jitter
Adds random zero-mean Gaussian noise directly to the signal.
* **Class**: `jitter(sigma=0.3)`
* **Parameters**:
  * `sigma` (float): Standard deviation of the Gaussian noise.

```python
from augs import jitter

aug = jitter(sigma=0.2)
augmented_x = aug(x)
```

---

### 3. Scaling
Scales each feature's magnitude along the entire sequence length stochastically using random Gaussian multiplier.
* **Class**: `scaling(sigma=0.5)`
* **Parameters**:
  * `sigma` (float): Standard deviation of the scaling factor distribution centered at `1.0`.

```python
from augs import scaling

aug = scaling(sigma=0.3)
augmented_x = aug(x)
```

---

### 4. Time Warp
Implements highly optimized, GPU-native time-warping using PyTorch linear interpolation over a random monotonic speed-change anchor grid.
* **Class**: `time_warp(n_speed_change=4, max_speed_ratio=2.0)`
* **Parameters**:
  * `n_speed_change` (int): Number of speed change segments.
  * `max_speed_ratio` (float): Maximum speed scaling ratio.

```python
from augs import time_warp

aug = time_warp(n_speed_change=3, max_speed_ratio=1.5)
augmented_x = aug(x)
```

---

### 5. Magnitude Warp
Applies the monotonic warping function along the channel/feature magnitude dimension instead of the temporal dimension.
* **Class**: `magnitude_warp(n_speed_change=4, max_speed_ratio=2.0)`
* **Parameters**: Same as `time_warp`.

```python
from augs import magnitude_warp

aug = magnitude_warp(n_speed_change=3, max_speed_ratio=1.5)
augmented_x = aug(x)
```

---

### 6. Window Slice
Stochastically crops a random window of the time series and linear interpolates it back to its original length `T`.
* **Class**: `window_slice(reduce_ratio=0.5)`
* **Parameters**:
  * `reduce_ratio` (float): The fraction of the original sequence length to keep.

```python
from augs import window_slice

aug = window_slice(reduce_ratio=0.7)
augmented_x = aug(x)
```

---

### 7. Window Warp
Extracts a random window of the time series, compresses or expands it stochastically by a factor selected from `scales`, and then linear interpolates the concatenated sequences back to the original sequence length `T`.
* **Class**: `window_warp(window_ratio=0.3, scales=[0.5, 2.0])`
* **Parameters**:
  * `window_ratio` (float): The percentage length of the window to warp.
  * `scales` (list): Candidates for window length scale ratio.

```python
from augs import window_warp

aug = window_warp(window_ratio=0.3, scales=[0.6, 1.8])
augmented_x = aug(x)
```

---

### 8. Subsequence
Stochastically crops a random subsequence, leaving it active, and zero-masks all elements outside the crop boundaries.
* **Class**: `subsequence()`

```python
from augs import subsequence

aug = subsequence()
augmented_x = aug(x)
```
