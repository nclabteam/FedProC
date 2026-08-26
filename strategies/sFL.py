"""sFL - Security-aware FL base."""

from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from attacks import ATTACKS

from .base import SharedMethods
from .tFL import tFL, tFL_Client


class sFLShared:
    """Paper-defined robust aggregators shared by the sFL family."""

    @staticmethod
    def krum_scores(models: Any, num_malicious: int) -> torch.Tensor:
        """Return Krum's sum of the n-f-2 nearest squared distances."""
        n = len(models)
        if num_malicious < 0 or 2 * num_malicious + 2 >= n:
            raise ValueError(
                "Krum requires a non-negative f and 2 * f + 2 < "
                f"n; got f={num_malicious}, n={n}."
            )
        flat = torch.stack(
            [
                torch.cat([parameter.flatten() for parameter in model.values()])
                for model in models
            ]
        ).float()
        distances = torch.cdist(flat, flat).square()
        distances.fill_diagonal_(float("inf"))
        return torch.topk(
            distances, n - num_malicious - 2, dim=1, largest=False
        ).values.sum(dim=1)

    @staticmethod
    def coordinate_median(models: Any) -> OrderedDict:
        """Return the usual coordinate-wise median, including even-n midpoint."""
        if not models:
            raise ValueError("Coordinate median requires at least one model.")
        n = len(models)
        result = OrderedDict()
        for name in models[0]:
            values = torch.sort(
                torch.stack([model[name] for model in models]), dim=0
            ).values
            result[name] = (
                values[n // 2].clone()
                if n % 2
                else ((values[n // 2 - 1] + values[n // 2]) / 2).clone()
            )
        return result

    @staticmethod
    def coordinate_trimmed_mean(models: Any, beta: float) -> OrderedDict:
        """Return Yin et al.'s coordinate-wise beta-trimmed mean."""
        if not 0 <= beta < 0.5:
            raise ValueError(f"beta must be in [0, 0.5); got {beta}.")
        if not models:
            raise ValueError("Coordinate trimmed mean requires at least one model.")
        cut = int(beta * len(models))
        upper = len(models) - cut
        return OrderedDict(
            (
                name,
                torch.sort(torch.stack([model[name] for model in models]), dim=0)
                .values[cut:upper]
                .mean(dim=0)
                .clone(),
            )
            for name in models[0]
        )


class sFL(sFLShared, tFL):
    """Byzantine-adversarial FL server base."""

    optional = {
        "attack": "NoAttack",
        "malicious_frac": 0.0,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument(
            "--attack",
            type=str,
            default=None,
            choices=ATTACKS,
            help="Byzantine attack to inject (sFL-based strategies only)",
        )
        parser.add_argument(
            "--malicious_frac",
            type=float,
            default=None,
            help="Fraction of clients designated as Byzantine (0 = benign mode)",
        )

    def __init__(self, configs: Any, times: Any) -> None:
        super().__init__(configs=configs, times=times)

        if not 0 <= self.malicious_frac <= 1:
            raise ValueError("malicious_frac must be in [0, 1].")

        attack_cls = SharedMethods._get_objective_function(
            func_type="attacks", func_name=self.attack
        )
        self._attack = attack_cls()

        n_mal = int(self.num_clients * self.malicious_frac)
        if n_mal > 0:
            rng = np.random.default_rng(self.seed)
            self.malicious_ids: set[int] = set(
                int(i) for i in rng.choice(self.num_clients, n_mal, replace=False)
            )
            self.logger.info(
                f"Byzantine clients ({n_mal}/{self.num_clients}, "
                f"attack={self.attack}): {sorted(self.malicious_ids)}"
            )
        else:
            self.malicious_ids: set[int] = set()

    def _inject_attacks(self, packages: Any) -> Any:
        """Inject attack into malicious clients' packages. No-op in benign mode."""
        if not self.malicious_ids:
            return packages
        malicious_in_round = [cid for cid in packages if cid in self.malicious_ids]
        if not malicious_in_round:
            return packages
        return self._attack.craft(packages, malicious_in_round, ctx=self)

    def train_one_round(self) -> Any:
        packages = self.trainer.train(self.selected_clients)
        packages = self._inject_attacks(packages=packages)
        self.aggregate_client_updates(packages=packages)
        return packages


class sFL_Client(sFLShared, tFL_Client):
    """tFL client with the unused sample-count payload removed from the wire."""

    def package(self) -> Any:
        package = super().package()
        package["__wire__"] = ("regular_model_params",)
        package.pop("score", None)
        return package
