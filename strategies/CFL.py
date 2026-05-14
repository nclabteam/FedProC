import copy
import time
from argparse import Namespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .pFL import pFL, pFL_Client


def _vectorize(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().cpu().flatten() for t in tensors])


def _max_norm(diffs: List[List[torch.Tensor]]) -> float:
    return max(_vectorize(d).norm().item() for d in diffs)


def _mean_norm(diffs: List[List[torch.Tensor]]) -> float:
    return float(
        torch.stack([_vectorize(d) for d in diffs]).mean(dim=0).norm().item()
    )


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
        self._client_models: Dict[int, Any] = {}
        self._client_diffs: Dict[int, Optional[List[torch.Tensor]]] = {}
        self._clusters: List[List[int]] = []
        self._cfl_round = 0

    def _ensure_init(self) -> None:
        if self._client_models:
            return
        for client in self.clients:
            self._client_models[client.id] = copy.deepcopy(self.model)
            self._client_diffs[client.id] = None
        self._clusters = [[client.id for client in self.clients]]

    def variables_to_be_sent(self) -> Dict[str, Any]:
        self._ensure_init()
        # Save the snapshot each client will subtract from after training
        models = [
            copy.deepcopy(self._client_models[c.id])
            for c in self.selected_clients
        ]
        return {"model": models}

    def receive_from_clients(self) -> None:
        self.client_data = []
        for client in self.selected_clients:
            data = client.send_to_server()
            self.client_data.append(data)
            self._client_diffs[client.id] = data["model_diff"]

    def aggregate_models(self) -> None:
        self._cfl_round += 1
        all_ids = [c.id for c in self.clients]
        id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
        n = len(all_ids)

        # Pairwise cosine similarity of gradient diffs
        sim = np.eye(n)
        for i, cid_a in enumerate(all_ids):
            da = self._client_diffs.get(cid_a)
            if da is None:
                continue
            va = _vectorize(da)
            for j, cid_b in enumerate(all_ids):
                if j <= i:
                    continue
                db = self._client_diffs.get(cid_b)
                if db is None:
                    continue
                vb = _vectorize(db)
                score = float(F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item())
                sim[i, j] = score
                sim[j, i] = score

        # Try to split each cluster
        new_clusters: List[List[int]] = []
        for cluster_ids in self._clusters:
            available_diffs = [
                self._client_diffs[cid]
                for cid in cluster_ids
                if self._client_diffs.get(cid) is not None
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
        for cluster_ids in self._clusters:
            cluster_diffs = [
                self._client_diffs[cid]
                for cid in cluster_ids
                if self._client_diffs.get(cid) is not None
            ]
            if not cluster_diffs:
                continue

            mean_diff = [
                torch.stack([d[k] for d in cluster_diffs]).mean(dim=0)
                for k in range(len(cluster_diffs[0]))
            ]
            for cid in cluster_ids:
                for param, diff in zip(
                    self._client_models[cid].parameters(), mean_diff
                ):
                    param.data.add_(diff.to(param.device))

        # Push updated cluster models back to client.model for evaluation
        client_by_id = {c.id: c for c in self.clients}
        for cid, model in self._client_models.items():
            if cid in client_by_id:
                client_by_id[cid].model.load_state_dict(model.state_dict())

        # Keep self.model as avg of all cluster models for server-side bookkeeping
        self.model = self.reset_model(self.model)
        for param in self.model.parameters():
            param.data.zero_()
        n_clients = len(all_ids)
        for cid in all_ids:
            for gp, cp in zip(
                self.model.parameters(),
                self._client_models[cid].parameters(),
            ):
                gp.data.add_(cp.data.to(gp.device), alpha=1.0 / n_clients)

    def save_models(self, save_type: str) -> None:
        if save_type not in ["last", "best"]:
            raise ValueError("save_type must be 'last' or 'best'")

        should_save = True
        if save_type == "best":
            metric_key = "personal_avg_test_loss"
            vals = self.metrics.get(metric_key, [])
            if not vals or vals[-1] != min(vals):
                should_save = False

        if not should_save:
            return

        if not self.exclude_server_model_processes:
            self.save_model(
                model=self.model,
                path=self.model_path,
                name=self.name,
                postfix=save_type,
                configs=self.configs,
                metadata={"save_type": save_type, "owner": "server"},
                verbose=self.logger,
            )

        for client in self.clients:
            # Save the cluster-aggregated model stored on client.model
            client.save_model(
                model=client.model,
                path=client.model_path,
                name=client.name,
                postfix=save_type,
                configs=client.configs,
                metadata={"save_type": save_type, "owner": client.name},
                verbose=client.logger,
            )

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

    def __init__(self, configs: Namespace, id: int, times: int) -> None:
        super().__init__(configs=configs, id=id, times=times)
        self._snapshot: List[torch.Tensor] = []

    def receive_from_server(self, data: dict) -> None:
        self._snapshot = [p.data.clone().cpu() for p in data["model"].parameters()]
        self.update_model_params(old=self.model, new=data["model"])

    def train(self):
        train_loader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device)
        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                self.optimizer.zero_grad()
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)
                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        if self.efficiency != "high":
            self.model.to("cpu")
        self.metrics["train_time"].append(time.time() - start_time)

    def variables_to_be_sent(self) -> Dict[str, Any]:
        model_diff = [
            lp.data.cpu() - snap
            for lp, snap in zip(self.model.parameters(), self._snapshot)
        ]
        return {"model_diff": model_diff, "score": self.train_samples}
