import torch

from .base import Attack


class StealthHF(Attack):
    """Stealth high-frequency injection: small amplitude to evade magnitude detection.

    Low-amplitude HF noise designed to slip under the magnitude-score threshold
    of temporal-plausibility detectors. Probe result: AUC=0.575 (evades), but
    damage <=0.7% — evasion is moot because the attack is too small to survive
    1/N averaging and cause meaningful harm (probe_diakrisis_kill/damage).
    """

    amplitude: float = 0.01

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
