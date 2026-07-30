"""FedRolex with manuscript-faithful rolling submodel extraction.

The rolling window for layer ``i`` in round ``j`` starts at ``j mod K_i`` and
contains ``floor(beta_n K_i)`` consecutive cyclic node indices.  The physical
submodel and selective per-coordinate aggregation are provided by ``ptFL``.

Reference: Alam et al., "FedRolex: Model-Heterogeneous Federated Learning with
Rolling Sub-Model Extraction," NeurIPS 2022, arXiv:2212.01548, Eq. (3) and
Algorithm 1.  Tensor-axis mechanics follow the authors' official repository.
"""

import torch

from .ptFL import ptFL, ptFL_Client


class FedRolex(ptFL):
    """Deterministic unit-stride rolling physical partial training."""

    optional = {"capacity": "1,0.5,0.25,0.125,0.0625"}

    def _pt_select_indices(
        self,
        group_name: str,
        full_width: int,
        retained: int,
        client_id: int,
    ) -> torch.Tensor:
        del group_name, client_id
        start = self.current_iter % full_width
        return (torch.arange(retained, dtype=torch.long) + start) % full_width


class FedRolex_Client(ptFL_Client):
    """FedRolex client; all mechanics are inherited from ``ptFL_Client``."""
