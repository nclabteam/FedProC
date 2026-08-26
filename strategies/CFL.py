import copy
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .pFL import pFL, pFL_Client


class CFLShared:
    """Paper operations used by the clustered server."""

    @staticmethod
    def vectorize(tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([tensor.detach().cpu().flatten() for tensor in tensors])

    @staticmethod
    def weighted_mean(
        diffs: List[List[torch.Tensor]], scores: List[float]
    ) -> List[torch.Tensor]:
        if not diffs or len(diffs) != len(scores):
            raise ValueError("diffs and scores must have the same non-zero length")
        total = float(sum(scores))
        if total <= 0:
            raise ValueError("CFL aggregation scores must sum to a positive value")
        result = []
        for parameter_diffs in zip(*diffs):
            stacked = torch.stack(parameter_diffs)
            weights = torch.as_tensor(
                scores,
                dtype=stacked.dtype,
                device=stacked.device,
            )
            weights = weights / total
            result.append(
                torch.sum(
                    stacked * weights.view(-1, *([1] * (stacked.ndim - 1))),
                    dim=0,
                )
            )
        return result

    @classmethod
    def max_norm(cls, diffs: List[List[torch.Tensor]]) -> float:
        return max(cls.vectorize(tensors=diff).norm().item() for diff in diffs)

    @classmethod
    def mean_norm(
        cls,
        diffs: List[List[torch.Tensor]],
        scores: List[float],
    ) -> float:
        return float(
            cls.vectorize(tensors=cls.weighted_mean(diffs=diffs, scores=scores))
            .norm()
            .item()
        )

    @staticmethod
    def split(similarity: np.ndarray) -> Any:
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="complete",
        ).fit(-similarity)
        first = np.argwhere(clustering.labels_ == 0).flatten().tolist()
        second = np.argwhere(clustering.labels_ == 1).flatten().tolist()
        return first, second


class CFL(CFLShared, pFL):
    """Clustered Federated Learning (CFL)."""

    compulsory = {"return_diff": True}
    optional = {
        "eps_1": 0.4,
        "eps_2": 1.6,
        "min_cluster_size": 2,
        "start_clustering_round": 20,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--eps_1", type=float, default=None)
        parser.add_argument("--eps_2", type=float, default=None)
        parser.add_argument("--min_cluster_size", type=int, default=None)
        parser.add_argument("--start_clustering_round", type=int, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False
        all_cids = [cid for cid in range(self.num_clients) if not self.is_new[cid]]
        # Per-client cluster-assigned model params (starts as copy of global)
        self._client_model: Dict[int, OrderedDict] = {
            cid: copy.deepcopy(self.public_model_params) for cid in all_cids
        }
        self._clusters: List[List[int]] = [all_cids]
        self._cfl_round: int = 0

    def select_clients(self) -> None:
        self._select_all_clients()

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id=client_id)
        # Send the cluster-assigned model instead of the global model
        pkg["regular_model_params"] = copy.deepcopy(self._client_model[client_id])
        # No personal overlay — cluster model is the full starting point
        pkg["personal_model_params"] = {}
        pkg["__wire__"] = ("regular_model_params",)
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        self._cfl_round += 1
        all_cids = list(self._client_model)
        missing = set(all_cids).difference(packages)
        if missing:
            raise ValueError(
                "CFL requires every incumbent client in each round; "
                f"missing {sorted(missing)}"
            )
        n = len(all_cids)
        id_to_idx = {cid: i for i, cid in enumerate(all_cids)}
        client_diffs = {
            cid: [-value for value in packages[cid]["model_params_diff"].values()]
            for cid in all_cids
        }
        client_scores = {cid: float(packages[cid]["score"]) for cid in all_cids}

        # Pairwise cosine similarity of gradient diffs
        sim = np.eye(n)
        for i, cid_a in enumerate(all_cids):
            va = self.vectorize(tensors=client_diffs[cid_a])
            for j, cid_b in enumerate(all_cids):
                if j <= i:
                    continue
                vb = self.vectorize(tensors=client_diffs[cid_b])
                score = float(
                    F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
                )
                sim[i, j] = score
                sim[j, i] = score

        # Try to split each cluster
        new_clusters: List[List[int]] = []
        for cluster_ids in self._clusters:
            cluster_diffs = [client_diffs[cid] for cid in cluster_ids]
            cluster_scores = [client_scores[cid] for cid in cluster_ids]
            if (
                len(cluster_ids) > self.min_cluster_size
                and self._cfl_round > self.start_clustering_round
                and self.mean_norm(diffs=cluster_diffs, scores=cluster_scores)
                < self.eps_1
                and self.max_norm(diffs=cluster_diffs) > self.eps_2
            ):
                idxs = [id_to_idx[cid] for cid in cluster_ids]
                sub_sim = sim[np.ix_(idxs, idxs)]
                c1_local, c2_local = self.split(similarity=sub_sim)
                new_clusters.append([cluster_ids[i] for i in c1_local])
                new_clusters.append([cluster_ids[i] for i in c2_local])
            else:
                new_clusters.append(cluster_ids)
        self._clusters = new_clusters

        # FedAvg within each cluster: model_i += mean(Δ_j for j in cluster)
        param_names = list(next(iter(self._client_model.values())).keys())
        for cluster_ids in self._clusters:
            cluster_diffs = [client_diffs[cid] for cid in cluster_ids]
            cluster_scores = [client_scores[cid] for cid in cluster_ids]
            mean_diff = self.weighted_mean(diffs=cluster_diffs, scores=cluster_scores)
            for cid in cluster_ids:
                for name, diff in zip(param_names, mean_diff):
                    orig = self._client_model[cid][name]
                    self._client_model[cid][name] = (orig + diff).to(orig.dtype)

        # Expose cluster models as personal params for pFL evaluation
        for cid in all_cids:
            self.clients_personal_model_params[cid] = dict(self._client_model[cid])

        # Dummy global: mean of all cluster models for server-side bookkeeping
        new_global = OrderedDict()
        total_score = sum(client_scores.values())
        for name in param_names:
            values = torch.stack(
                [self._client_model[cid][name].float() for cid in all_cids]
            )
            weights = torch.tensor(
                [client_scores[cid] / total_score for cid in all_cids],
                dtype=values.dtype,
            )
            new_global[name] = torch.sum(
                values * weights.view(-1, *([1] * (values.ndim - 1))),
                dim=0,
            )
        self._commit_global(new_params=new_global)


class CFL_Client(pFL_Client):
    """Client for CFL."""

    return_diff = True
