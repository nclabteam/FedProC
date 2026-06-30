"""sFL - Security-aware FL base.

Extends tFL with a pluggable threat injection seam between client training
and server aggregation:
  - Benign mode (default, --attack NoAttack / --malicious_frac 0): no-op.
  - Adversarial mode (--attack <name> --malicious_frac >0): attack.craft()
    modifies malicious clients' packages before aggregation each round.

Canonical class hierarchy:
  SharedMethods  <- base.py
    └─ tFL_Client
    └─ tFL
         └─ sFL          <- this file: adds _inject_attacks to train_one_round
              └─ (defense strategies)

Defense strategies inherit from sFL so they get the injection seam for free.
Flip --attack at run time to switch between benign and adversarial eval
without changing the strategy code.
"""

import numpy as np
from typing import Set

from attacks import ATTACKS
from .base import SharedMethods
from .tFL import tFL, tFL_Client


class sFL(tFL):
    """Byzantine-adversarial FL server base.

    In benign mode (malicious_frac=0 or attack=NoAttack), identical to tFL.
    In adversarial mode, craft() is called on malicious clients' packages
    after training but before aggregation each round.

    Subclass and override aggregate_client_updates() to implement a
    Byzantine-robust aggregation rule on top of this injection seam.
    """

    optional = {
        "attack": "NoAttack",
        "malicious_frac": 0.0,
    }

    @classmethod
    def args_update(cls, parser):
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

    def __init__(self, configs, times) -> None:
        super().__init__(configs, times)

        attack_name = getattr(self, "attack", "NoAttack")
        attack_cls = SharedMethods._get_objective_function("attacks", attack_name)
        self._attack = attack_cls()

        n_mal = int(self.num_clients * getattr(self, "malicious_frac", 0.0))
        if n_mal > 0:
            rng = np.random.default_rng(self.seed)
            self.malicious_ids: Set[int] = set(
                int(i) for i in rng.choice(self.num_clients, n_mal, replace=False)
            )
            self.logger.info(
                f"Byzantine clients ({n_mal}/{self.num_clients}, "
                f"attack={attack_name}): {sorted(self.malicious_ids)}"
            )
        else:
            self.malicious_ids: Set[int] = set()

    def _inject_attacks(self, packages):
        """Inject attack into malicious clients' packages. No-op in benign mode."""
        if not self.malicious_ids:
            return packages
        malicious_in_round = [cid for cid in packages if cid in self.malicious_ids]
        if not malicious_in_round:
            return packages
        return self._attack.craft(packages, malicious_in_round, ctx=self)

    def train_one_round(self):
        packages = self.trainer.train(self.selected_clients)
        packages = self._inject_attacks(packages)
        self.aggregate_client_updates(packages)
        return packages


class sFL_Client(tFL_Client):
    """Passthrough — same as tFL_Client. Named for class-discovery consistency."""
