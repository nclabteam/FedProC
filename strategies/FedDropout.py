"""Federated Dropout with fresh random physical submodels per client.

Each selected client receives exactly ``floor(beta_n K_i)`` randomly selected
hidden nodes from every layer.  Input and task-output dimensions stay fixed,
and the server maps trained dense submodels back through retained manifests.

References: Caldas et al., "Expanding the Reach of Federated Learning by
Reducing Client Resource Requirements," arXiv:1812.07210, Section 3.2; and the
``ResnetServerRandom``/``TransformerServerRandomSO`` baselines in the official
FedRolex repository: https://github.com/AIoT-MLSys-Lab/FedRolex.
"""

import zlib

import torch

from .ptFL import ptFL, ptFL_Client


class FedDropout(ptFL):
    """Fresh independently random, fixed-width submodels each round."""

    optional = {"capacity": "0.75"}

    def _pt_select_indices(
        self,
        group_name: str,
        full_width: int,
        retained: int,
        client_id: int,
    ) -> torch.Tensor:
        if self.seed is None:
            return torch.randperm(full_width)[:retained]

        group_id = zlib.crc32(group_name.encode("utf-8"))
        derived_seed = self._derive_seed(
            int(self.seed) + int(self.times),
            self.current_iter,
            client_id,
            group_id,
        )
        generator = torch.Generator().manual_seed(derived_seed)
        return torch.randperm(full_width, generator=generator)[:retained]


class FedDropout_Client(ptFL_Client):
    """Federated Dropout client using the shared physical PT protocol."""
