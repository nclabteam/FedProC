"""Federated Partial Model Training."""

from argparse import ArgumentParser, Namespace

from .ptFL import TrainableSpec, ptFLUpdate, ptFLUpdate_Client


class FedPMT(ptFLUpdate):
    """Train a client-specific suffix of layers with a full forward pass."""

    optional = {"fedpmt_depths": "all"}

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--fedpmt_depths", type=str, default=None)

    @staticmethod
    def _fedpmt_parse_depths(raw: str, layer_count: int) -> tuple[int, ...]:
        if layer_count < 1:
            raise ValueError("FedPMT requires at least one parameterized layer")
        if raw.strip().lower() == "all":
            return tuple(range(1, layer_count + 1))

        try:
            depths = tuple(int(value.strip()) for value in raw.split(","))
        except ValueError as error:
            raise ValueError("fedpmt_depths must be 'all' or comma-separated integers") from error
        if (
            not depths
            or depths != tuple(sorted(set(depths)))
            or depths[0] < 1
            or depths[-1] != layer_count
        ):
            raise ValueError(
                "fedpmt_depths must be strictly increasing valid suffix depths "
                "ending with the full model depth"
            )
        return depths

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self._fedpmt_groups = tuple(self._pt_parameter_groups(model=self.model))
        self._fedpmt_depths = self._fedpmt_parse_depths(
            raw=self.fedpmt_depths,
            layer_count=len(self._fedpmt_groups),
        )

    def _pt_update_spec(self, client_id: int) -> TrainableSpec:
        depth = self._fedpmt_depths[client_id % len(self._fedpmt_depths)]

        # Paper pseudocode and Eq. (6): back-propagate only through deep layers.
        return TrainableSpec(
            (name, None) for name in self._fedpmt_groups[-depth:]
        )


class FedPMT_Client(ptFLUpdate_Client):
    """FedPMT client using the shared masked-update transport."""
