from argparse import Namespace
from typing import Any, Mapping

from topologies import TOPOLOGIES

from .pFL import pFL, pFL_Client


class dFLShared:
    """Topology operations shared by decentralized strategies."""

    @staticmethod
    def undirected_topology(
        topology: Mapping[int, list[int]], num_nodes: int
    ) -> dict[int, list[int]]:
        neighbors = {client_id: set() for client_id in range(num_nodes)}
        for client_id, peers in topology.items():
            if client_id not in neighbors:
                raise ValueError(f"unknown topology node: {client_id}")
            for peer in peers:
                if peer not in neighbors:
                    raise ValueError(f"unknown topology node: {peer}")
                if peer != client_id:
                    neighbors[client_id].add(peer)
                    neighbors[peer].add(client_id)
        return {client_id: sorted(peers) for client_id, peers in neighbors.items()}

    @staticmethod
    def metropolis_weights(
        client_id: int,
        peers: list[int],
        topology: Mapping[int, list[int]],
        active: set[int],
    ) -> list[float]:
        degree = {
            node: sum(neighbor in active for neighbor in topology[node])
            for node in active
        }
        neighbor_weights = [
            1.0 / (1 + max(degree[client_id], degree[peer])) for peer in peers[1:]
        ]
        return [1.0 - sum(neighbor_weights), *neighbor_weights]


class dFL(dFLShared, pFL):
    """Stateless simulation of peer-to-peer local training and gossip."""

    optional = {"topology": "FullyConnected"}
    compulsory = {"exclude_server_model_processes": True}

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--topology", type=str, default=None, choices=TOPOLOGIES)

    def __init__(self, configs: Namespace, times: int) -> None:
        topology_name = getattr(configs, "topology", None) or self.optional["topology"]
        generated = getattr(__import__("topologies"), topology_name)(
            num_nodes=configs.num_clients
        ).neighbors
        neighbors = self.undirected_topology(
            topology=generated, num_nodes=configs.num_clients
        )
        configs.__dict__["neighbors"] = neighbors
        super().__init__(configs=configs, times=times)
        self.topology = neighbors
        self.name = "  ORCHES  "

    def select_clients(self) -> None:
        self._select_all_clients()

    def package(self, client_id: int) -> dict[str, Any]:
        result = super().package(client_id=client_id)
        personal = self.clients_personal_model_params[client_id]
        if personal:
            result["regular_model_params"] = dict(personal)
        result["personal_model_params"] = {}
        result["__wire__"] = ()
        return result

    def _gossip_once(self) -> None:
        snapshot = {
            cid: dict(self.clients_personal_model_params[cid])
            for cid in range(self.num_clients)
        }
        active = {cid for cid, model in snapshot.items() if model}
        for cid in range(self.num_clients):
            if not snapshot[cid]:
                continue
            peers = [cid, *(peer for peer in self.topology[cid] if snapshot[peer])]
            # Paper mixing step: x_i^(t+1) = sum_j w_ij z_j^t.
            self.clients_personal_model_params[cid].update(
                self.mean_models(
                    models=[snapshot[peer] for peer in peers],
                    weights=self.metropolis_weights(
                        client_id=cid,
                        peers=peers,
                        topology=self.topology,
                        active=active,
                    ),
                )
            )

    def _num_gossip_steps(self) -> int:
        return 1

    def _compute_send_mb(
        self, packages: Mapping[int, dict[str, Any]]
    ) -> tuple[dict[int, float], float]:
        steps = self._num_gossip_steps()
        uplink = {
            cid: self.get_size(obj=package["regular_model_params"])
            * sum(peer in packages for peer in self.topology[cid])
            * steps
            for cid, package in packages.items()
        }
        return uplink, sum(uplink.values())

    def aggregate_client_updates(self, packages: Mapping[int, dict[str, Any]]) -> None:
        for cid, pkg in packages.items():
            self.clients_personal_model_params[cid].update(pkg["regular_model_params"])
        for _ in range(self._num_gossip_steps()):
            self._gossip_once()


class dFL_Client(pFL_Client):
    """Reusable worker; logical node state remains server-owned."""

    def package(self) -> dict[str, Any]:
        package = super().package()
        package["__wire__"] = ()
        return package
