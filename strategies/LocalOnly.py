from .pFL import pFL, pFL_Client


class LocalOnly(pFL):
    """Local-only training: each client trains independently without any aggregation."""

    def __init__(self, configs, times):
        super().__init__(configs, times)
        self._send_mb_per_round = 0.0

    def aggregate_client_updates(self, packages) -> None:
        for cid, pkg in packages.items():
            self.clients_personal_model_params[cid].update(pkg["regular_model_params"])

    def evaluate_generalization(self, *args, **kwargs):
        pass


class LocalOnly_Client(pFL_Client):
    pass
