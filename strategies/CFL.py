import copy
from argparse import Namespace
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .pFL import pFL, pFL_Client


def _vectorize(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().cpu().flatten() for t in tensors])


def _max_norm(diffs: List[List[torch.Tensor]]) -> float:
    return max(_vectorize(d).norm().item() for d in diffs)


def _mean_norm(diffs: List[List[torch.Tensor]]) -> float:
    return float(torch.stack([_vectorize(d) for d in diffs]).mean(dim=0).norm().item())


class CFL(pFL):
    """
    Clustered Federated Learning (CFL).

    Clients are dynamically grouped by the cosine similarity of their gradient
    updates.  Once gradients within a cluster converge (mean_norm < eps_1) but
    the cluster still contains divergent clients (max_norm > eps_2), it is
    split into two sub-clusters via agglomerative clustering.  FedAvg is then
    performed within each cluster independently so that each client ends up
    with a cluster-specific personalised model.

    Reference: Sattler et al., "Clustered Federated Learning: Model-Agnostic
    Distributed Multi-Task Optimization under Privacy Constraints", ArXiv 2019.
    arXiv 1910.01991.
    """

    optional = {
        "eps_1": 0.4,
        "eps_2": 1.6,
        "min_cluster_size": 2,
        "start_clustering_round": 20,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--eps_1", type=float, default=None)
        parser.add_argument("--eps_2", type=float, default=None)
        parser.add_argument("--min_cluster_size", type=int, default=None)
        parser.add_argument("--start_clustering_round", type=int, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False
        all_cids = list(range(self.num_clients))
        # Per-client cluster-assigned model params (starts as copy of global)
        self._client_model: Dict[int, OrderedDict] = {
            cid: copy.deepcopy(self.public_model_params) for cid in all_cids
        }
        # Per-client latest gradient diff (None until first participation)
        self._client_diff: Dict[int, Optional[List[torch.Tensor]]] = {
            cid: None for cid in all_cids
        }
        self._clusters: List[List[int]] = [all_cids]
        self._cfl_round: int = 0

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id)
        # Send the cluster-assigned model instead of the global model
        pkg["regular_model_params"] = copy.deepcopy(self._client_model[client_id])
        # No personal overlay — cluster model is the full starting point
        pkg["personal_model_params"] = {}
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        self._cfl_round += 1
        all_cids = list(range(self.num_clients))
        n = len(all_cids)
        id_to_idx = {cid: i for i, cid in enumerate(all_cids)}

        # Update diffs from participating clients
        for cid, pkg in packages.items():
            self._client_diff[cid] = pkg["model_diff"]

        # Pairwise cosine similarity of gradient diffs
        sim = np.eye(n)
        for i, cid_a in enumerate(all_cids):
            da = self._client_diff[cid_a]
            if da is None:
                continue
            va = _vectorize(da)
            for j, cid_b in enumerate(all_cids):
                if j <= i:
                    continue
                db = self._client_diff[cid_b]
                if db is None:
                    continue
                vb = _vectorize(db)
                score = float(
                    F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
                )
                sim[i, j] = score
                sim[j, i] = score

        # Try to split each cluster
        new_clusters: List[List[int]] = []
        for cluster_ids in self._clusters:
            available_diffs = [
                self._client_diff[cid]
                for cid in cluster_ids
                if self._client_diff[cid] is not None
            ]
            if (
                len(available_diffs) >= 2
                and len(cluster_ids) > self.min_cluster_size
                and self._cfl_round >= self.start_clustering_round
                and _mean_norm(available_diffs) < self.eps_1
                and _max_norm(available_diffs) > self.eps_2
            ):
                idxs = [id_to_idx[cid] for cid in cluster_ids]
                sub_sim = sim[np.ix_(idxs, idxs)]
                c1_local, c2_local = self._split(sub_sim)
                new_clusters.append([cluster_ids[i] for i in c1_local])
                new_clusters.append([cluster_ids[i] for i in c2_local])
            else:
                new_clusters.append(cluster_ids)
        self._clusters = new_clusters

        # FedAvg within each cluster: model_i += mean(Δ_j for j in cluster)
        param_names = list(next(iter(self._client_model.values())).keys())
        for cluster_ids in self._clusters:
            cluster_diffs = [
                self._client_diff[cid]
                for cid in cluster_ids
                if self._client_diff[cid] is not None
            ]
            if not cluster_diffs:
                continue
            mean_diff = [
                torch.stack([d[k] for d in cluster_diffs]).mean(dim=0)
                for k in range(len(cluster_diffs[0]))
            ]
            for cid in cluster_ids:
                for name, diff in zip(param_names, mean_diff):
                    orig = self._client_model[cid][name]
                    self._client_model[cid][name] = (orig + diff).to(orig.dtype)

        # Expose cluster models as personal params for pFL evaluation
        for cid in all_cids:
            self.clients_personal_model_params[cid] = dict(self._client_model[cid])

        # Dummy global: mean of all cluster models for server-side bookkeeping
        new_global = OrderedDict()
        for name in param_names:
            new_global[name] = torch.stack(
                [self._client_model[cid][name].float() for cid in all_cids]
            ).mean(dim=0)
        self._commit_global(new_global)

    @staticmethod
    def _split(sim_matrix: np.ndarray):
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            metric="precomputed", linkage="complete"
        ).fit(-sim_matrix)
        c1 = np.argwhere(clustering.labels_ == 0).flatten().tolist()
        c2 = np.argwhere(clustering.labels_ == 1).flatten().tolist()
        return c1, c2


class CFL_Client(pFL_Client):
    """
    Client for CFL. Receives a cluster-personalised model each round, trains
    locally, and sends back the gradient diff Δ = w_local - w_received.
    """

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        # Snapshot of the received cluster model before training (CPU, by param name)
        self._init_params: Dict[str, torch.Tensor] = {
            name: v.clone().cpu()
            for name, v in package["regular_model_params"].items()
        }

    def package(self) -> dict:
        out = super().package()
        current_state = self.model.state_dict()
        out["model_diff"] = [
            current_state[name].cpu() - self._init_params[name]
            for name in self.regular_params_name
        ]
        return out
