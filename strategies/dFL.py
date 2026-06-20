from collections import OrderedDict

from topologies import TOPOLOGIES

from .pFL import pFL, pFL_Client


class dFL(pFL):
    """Decentralized FL base: server orchestrates gossip; no global model aggregation.

    Each round, every node trains locally, then exchanges models with its topology
    neighbors via a server-managed gossip step. Per-client models live in
    clients_personal_model_params; gossip is computed entirely server-side.
    """

    optional = {"topology": "FullyConnected"}
    compulsory = {"exclude_server_model_processes": True}

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--topology", type=str, default=None, choices=TOPOLOGIES)

    def __init__(self, configs, times):
        # set_configs first so self.topology (string) is available for get_topology()
        # get_topology() writes configs.neighbors so Trainer can pass it to clients
        # super().__init__() re-runs set_configs, resetting self.topology to the string —
        # so we save the neighbors dict and restore it after.
        self.set_configs(configs=configs, times=times)
        self.get_topology()
        _neighbors = self.topology
        super().__init__(configs=configs, times=times)
        self.topology = _neighbors
        self.name = "  ORCHES  "

    def get_topology(self):
        self.topology = getattr(__import__("topologies"), self.topology)(
            num_nodes=self.num_clients
        ).neighbors
        self.configs.__dict__["neighbors"] = self.topology

    def select_clients(self) -> None:
        self.selected_clients = [i for i in range(self.num_clients) if not self.is_new[i]]

    def package(self, client_id: int) -> dict:
        result = super().package(client_id)
        personal = self.clients_personal_model_params[client_id]
        if personal:
            result["regular_model_params"] = dict(personal)
        return result

    def _gossip_once(self) -> None:
        snap = {
            cid: dict(self.clients_personal_model_params[cid])
            for cid in range(self.num_clients)
        }
        for cid in range(self.num_clients):
            if not snap[cid]:
                continue
            peers = [cid] + [j for j in self.topology[cid] if snap[j]]
            w = 1.0 / len(peers)
            agg = OrderedDict(
                (k, sum(snap[j][k] * w for j in peers))
                for k in snap[cid]
            )
            self.clients_personal_model_params[cid].update(agg)

    def aggregate_client_updates(self, packages) -> None:
        for cid, pkg in packages.items():
            self.clients_personal_model_params[cid].update(pkg["regular_model_params"])
        self._gossip_once()


class dFL_Client(pFL_Client):
    """Stateless DFL client — gossip aggregation is done server-side."""
    pass
