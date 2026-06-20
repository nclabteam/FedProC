from .tFL import tFL, tFL_Client


class nFL(tFL):
    """No-federation base: clients train independently, global model never updated."""

    def aggregate_client_updates(self, packages) -> None:
        for cid, pkg in packages.items():
            self.clients_personal_model_params[cid].update(pkg["regular_model_params"])


class nFL_Client(tFL_Client):
    """Passthrough — same as tFL_Client."""
