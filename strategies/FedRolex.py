"""FedRolex with manuscript-faithful rolling submodel extraction."""

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
