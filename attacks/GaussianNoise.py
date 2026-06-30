import torch

from .base import Attack


class GaussianNoise(Attack):
    """Replace malicious updates with zero-mean Gaussian noise.

    Noise magnitude is proportional to the parameter's own norm so the
    attack strength scales with the model. AUC=1.000 vs benign
    (probe_diakrisis_kill): pure noise is easily out-of-band.
    """

    noise_scale: float = 1.0

    def craft(self, packages, malicious_ids, ctx):
        for cid in malicious_ids:
            pkg = packages[cid]
            pkg["regular_model_params"] = {
                k: torch.randn_like(v) * (v.norm().item() * self.noise_scale + 1e-8)
                for k, v in pkg["regular_model_params"].items()
            }
        return packages
