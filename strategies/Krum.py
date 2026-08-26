from typing import Any

import torch

from .sFL import sFL, sFL_Client


class Krum(sFL):
    """Krum and Multi-Krum Byzantine-robust aggregation (Blanchard et al., NeurIPS 2017)."""

    optional = {
        "num_malicious_clients": 0,
        "num_clients_to_keep": 0,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        super().args_update(parser=parser)
        parser.add_argument(
            "--num_malicious_clients",
            type=int,
            default=None,
            help="f in Krum: assumed Byzantine count. 0 = derive from --malicious_frac.",
        )
        parser.add_argument(
            "--num_clients_to_keep",
            type=int,
            default=None,
            help="Number of clients to keep before averaging (MultiKrum). Defaults to 0, in that case classical Krum is applied.",
        )

    def aggregate_client_updates(self, packages: Any) -> None:
        ordered_packages = sorted(packages.items())
        client_weights = [p["regular_model_params"] for _, p in ordered_packages]
        f = self.num_malicious_clients or sum(
            client_id in self.malicious_ids for client_id, _ in ordered_packages
        )
        scores = self.krum_scores(models=client_weights, num_malicious=f)

        if not 0 <= self.num_clients_to_keep <= len(client_weights):
            raise ValueError(
                "num_clients_to_keep must be between 0 and the number of "
                "participating clients."
            )
        if self.num_clients_to_keep:
            best = torch.argsort(scores, stable=True)[: self.num_clients_to_keep]
            self._commit_global(
                new_params=self.mean_models(
                    models=[client_weights[int(i)] for i in best]
                )
            )
            return
        self._commit_global(new_params=client_weights[int(torch.argmin(scores))])


class Krum_Client(sFL_Client):
    """Use the security-aware stateless worker."""
