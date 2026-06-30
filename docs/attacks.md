# Attacks

Pluggable attacks for adversarial evaluation of sFL-based strategies. Each attack is a subclass of `Attack` (`attacks/base.py`) that overrides `craft()`. Pass the class name to `--attack`; set `--malicious_frac` to control the injected fraction of clients.

## Available Attacks

| Name | Description |
|------|-------------|
| `NoAttack` | No-op; benign mode. Packages are returned unmodified (default) |
| `SignFlip` | Sends `-W` instead of `W` for every malicious client. Destroys gradient signal |
| `GaussianNoise` | Replaces malicious updates with zero-mean Gaussian noise scaled to the parameter's own norm |
| `ScaleBoost` | Scales honest updates by a constant factor (default 10×) to overpower the aggregate |
| `BackdoorHF` | Adds out-of-band high-frequency noise mimicking a spectral trigger; amplitude proportional to parameter norm |
| `StealthHF` | Low-amplitude (0.01×) HF noise designed to evade magnitude-based detectors; damage is negligible due to 1/N averaging |

## Adding a Custom Attack

Create a file `attacks/MyAttack.py` with a class of the same name:

```python
from .base import Attack

class MyAttack(Attack):
    scale: float = 2.0  # optional class-level hyperparameter

    def craft(self, packages, malicious_ids, ctx):
        for cid in malicious_ids:
            pkg = packages[cid]
            pkg["regular_model_params"] = {
                k: v * self.scale
                for k, v in pkg["regular_model_params"].items()
            }
        return packages
```

`ctx` is the `sFL` server instance — use it to access `ctx.current_iter`, `ctx.public_model_params`, `ctx.configs`, etc.

The class is auto-discovered and added to `ATTACKS` on import; no registration step needed.
