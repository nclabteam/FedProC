import torch

from .base import Attack


class BackdoorHF(Attack):
    """High-frequency backdoor: inject out-of-band noise into malicious updates.

    Adds a random perturbation whose energy is concentrated outside the benign
    DCT-H consensus band, mimicking an out-of-band spectral trigger. Amplitude
    is a fraction of the parameter's own norm. AUC=1.000 vs benign pool
    (probe_diakrisis_kill): HF energy lands far outside the consensus band.
    """

    amplitude: float = 1.0

    def craft(self, packages, malicious_ids, ctx):
        for cid in malicious_ids:
            pkg = packages[cid]
            new_params = {}
            for k, v in pkg["regular_model_params"].items():
                noise = torch.randn_like(v)
                noise = noise / (noise.norm() + 1e-9) * v.norm() * self.amplitude
                new_params[k] = v + noise
            pkg["regular_model_params"] = new_params
        return packages
