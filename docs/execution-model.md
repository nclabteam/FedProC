# Execution Model

This page describes how FedProC runs per-client work — the single dispatch
contract shared by every strategy. Understanding it is enough to add a new
strategy that runs correctly in **both** serial and Ray-parallel mode without
writing any parallel boilerplate.

## Two orthogonal knobs

| Knob | Flag | What it controls |
|------|------|------------------|
| **Parallelism** | derived | Whether selected clients run concurrently via Ray. |
| **Efficiency** | `--efficiency` | Device residency policy: when models move GPU↔CPU. |

These are **independent concerns** and can be combined freely.

- **Parallelism** is enabled automatically when GPUs *and* workers are available:
  `configs.parallel = (num_gpus > 0 and num_workers > 0)`. It is read in exactly
  one place — `tFL._dispatch` on the server. Clients never branch on it.
- **Efficiency** (`low` / `med` / `high`) only decides *when* a model is offloaded
  to CPU during training/eval to manage VRAM:
    - `low` — offload after every epoch / eval step,
    - `med` — offload once after each local train/eval call,
    - `high` — keep resident when possible (default).

  Efficiency does **not** affect what `train()` returns. The CPU-ification needed
  to cross the Ray process boundary is handled separately, only on the parallel
  path (see [`_offload_package`](#_offload_package)).

## The dispatch contract

All per-client method calls go through one server method:

```python
self._dispatch(method_name, clients)
```

`_dispatch` chooses the execution mode:

- **Serial** — for each client, call the method and apply its return value:
  ```python
  self._apply_client_result(client, getattr(client, method)())
  ```
- **Parallel** — a bounded Ray worker pool (one in-flight job per worker). Each
  worker runs the method and CPU-ifies the returned package *before* it crosses
  the process boundary; the driver then applies it:
  ```python
  # in the worker
  package = client._offload_package(getattr(client, method)())
  # on the driver, as each future completes
  self._apply_client_result(client, ray.get(future))
  ```

Because both paths route through `_apply_client_result`, **the same code applies
results in serial and parallel.** In serial the package holds live references to
the client's own objects, so the apply is a true no-op (identity checks); in
parallel the package holds CPU copies deserialized from the worker, so the values
are copied back onto the original client.

### `train()` returns a package

Client `train()` (and any other dispatched method) **always returns a package
dict** — it never inspects `self.parallel` or `self.efficiency`:

```python
def train(self) -> Dict[str, Any]:
    ...  # train in place; efficiency governs residency here
    return {
        "model": self.model,            # live reference
        "optimizer_state": self.optimizer,
        "train_time": time.time() - start_time,
        "train_samples": self.train_samples,
    }
```

### `_apply_client_result`

The base implementation (`tFL._apply_client_result`) applies the four standard
keys, with identity checks that make serial a no-op, and a guard that raises
`NotImplementedError` if the package contains any key outside
`tFL._KNOWN_PACKAGE_KEYS = {"model", "optimizer_state", "train_time", "train_samples", "id"}`.
The guard exists to catch a strategy that returns extra state but forgets to
handle it — failing loudly instead of silently dropping the state.

### `_offload_package`

Called **only on the Ray worker path**. The base implementation moves
`model` to CPU and converts `optimizer_state` to a CPU state dict, leaving every
other key untouched. Strategies that return additional GPU tensors override it
(calling `super()` first) to CPU-ify those too.

### `_gather_losses`

Evaluation is the map-*reduce* counterpart of `_dispatch`: instead of applying
state back, it gathers one scalar loss per client. `tFL._gather_losses(mode,
dataset_type)` uses the same bounded worker pool and is shared by
`evaluate_generalization_loss` and `evaluate_personalization_loss`.

## Adding a strategy

### 1. Standard gradient strategy — nothing to do

Inherit and you are done. The inherited `train()` / `_dispatch` /
`_apply_client_result` / `_offload_package` already handle everything:

```python
class FedNew(tFL):
    def aggregate_models(self): ...

class FedNew_Client(tFL_Client):
    pass  # or override train_one_epoch for a custom local objective
```

### 2. Strategy that returns custom state (e.g. closed-form statistics)

Return the extra keys from `train()`, and give the **server** a full override of
`_apply_client_result` that writes them back. Because the override does not call
`super()`, the base key guard never runs — so you do **not** set
`_KNOWN_PACKAGE_KEYS`. (Reference: `FedRidge`.)

```python
class FedStat(tFL):
    def _apply_client_result(self, client, package):
        if package is None:
            return
        client._sigma_xx = package["sigma_xx"]
        client._sigma_xy = package["sigma_xy"]
        client.train_samples = package["train_samples"]

class FedStat_Client(tFL_Client):
    def train(self) -> Dict[str, Any]:
        ...  # compute on CPU; set self._sigma_xx / self._sigma_xy in place
        return {
            "sigma_xx": self._sigma_xx,
            "sigma_xy": self._sigma_xy,
            "train_samples": self.train_samples,
        }
```

If a custom tensor could live on the GPU, also override `_offload_package` on the
client to move it to CPU for the Ray return:

```python
class FedStat_Client(tFL_Client):
    def _offload_package(self, package):
        package = super()._offload_package(package)
        if package is not None and "extra" in package:
            package["extra"] = package["extra"].cpu()
        return package
```

### 3. Gradient strategy with *extra* keys on top of the standard package

Extend the key set and call `super()` so the base still applies model/optimizer.
(Reference: `FedALA`, `MOTAR`.)

```python
class FedExtra(pFL):
    _KNOWN_PACKAGE_KEYS = tFL._KNOWN_PACKAGE_KEYS | frozenset({"aux"})

    def _apply_client_result(self, client, package):
        if package is None:
            return
        super()._apply_client_result(client, package)
        client.aux = package["aux"]
```

### Multi-phase strategies

A strategy that dispatches *different* methods in different phases (e.g. a
one-shot warm-start followed by gradient rounds) returns different package
schemas. Route on the schema in `_apply_client_result`:

```python
def _apply_client_result(self, client, package):
    if package is None:
        return
    if "model" in package:                       # gradient phase
        tFL._apply_client_result(self, client, package)
    else:                                          # statistics phase
        super()._apply_client_result(client, package)
```

(References: `FedTELOSInit`, `FedTELOSLocalFinev7`.)

!!! warning "Every dispatched method must return its state"
    A dispatched method that returns `None` is treated as a no-op by
    `_apply_client_result`. That is correct for a method whose only effect is
    already captured elsewhere, but a method that trains a model and returns
    `None` will **silently lose its work in parallel mode** (the Ray worker
    trains a throwaway copy). Always return a package from any method that
    mutates client state you need back on the driver.
