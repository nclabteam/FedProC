from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from .pFL import pFL, pFL_Client


class FINCH:
    """
    FINCH - First Integer Neighbor Clustering Hierarchy.

    Parameter-free hierarchical clustering using first neighbor relations.
    Reference: Sarfraz et al., CVPR 2019.
    """

    def __init__(self, metric="euclidean", n_jobs=1):
        self.metric = metric
        self.n_jobs = n_jobs
        self.partitions = {}

    def _finch(self, X, prev_clusters, prev_cluster_core_indices):
        if not prev_clusters:
            data = X
        else:
            data = prev_clusters

        nbrs = NearestNeighbors(
            n_neighbors=2, metric=self.metric, n_jobs=self.n_jobs
        ).fit(data)
        connectivity = nbrs.kneighbors_graph(data)
        connectivity @= connectivity.T
        connectivity.setdiag(0)
        connectivity.eliminate_zeros()

        n_components, labels = connected_components(csgraph=connectivity)

        if len(labels) < self.n_samples:
            new_labels = np.full(self.n_samples, 0)
            for i in range(n_components):
                idx = np.where(labels == i)[0]
                idx = sum([prev_cluster_core_indices[j] for j in idx], [])
                new_labels[idx] = i
            labels = new_labels

        cluster_centers = []
        cluster_core_indices = []
        for i in range(n_components):
            idx = np.where(labels == i)[0]
            cluster_core_indices.append(idx.tolist())
            cluster_centers.append(X[idx].mean(axis=0))

        return n_components, labels, cluster_centers, cluster_core_indices

    def fit(self, X):
        self.n_samples = X.shape[0]
        results = {}
        cluster_centers = None
        cluster_core_indices = None
        n_components = len(X)
        i = 0
        while n_components > 1:
            n_components, labels, cluster_centers, cluster_core_indices = self._finch(
                X, cluster_centers, cluster_core_indices
            )
            if n_components == 1:
                break
            results[f"partition_{i}"] = {
                "n_clusters": n_components,
                "labels": labels,
                "cluster_centers": cluster_centers,
                "cluster_core_indices": cluster_core_indices,
            }
            i += 1
        self.partitions = results
        return self


class FDCR(pFL):
    """
    FDCR: Parameter Disparities Dissection for Backdoor Defense
    in Heterogeneous Federated Learning (NeurIPS 2024).

    Server-side aggregation strategy that detects and excludes
    backdoor attackers using Fisher-weighted parameter disparities
    and FINCH clustering.
    """

    optional = {
        "bad_client_rate": 0.3,
        "fisher_epochs": 1,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--bad_client_rate", type=float, default=None)
        parser.add_argument("--fisher_epochs", type=int, default=None)

    def __init__(self, configs, times):
        self.prev_global_params = None
        super().__init__(configs, times)

    def aggregate_client_updates(self, packages) -> None:
        cids = list(packages.keys())
        scores = [packages[cid]["score"] for cid in cids]
        total = float(sum(scores))
        freq = torch.tensor([s / total for s in scores])
        lr = self.learning_rate

        if self.prev_global_params is None:
            self.prev_global_params = OrderedDict(
                (k, v.clone()) for k, v in self.public_model_params.items()
            )

        prev_vec = torch.cat(
            [p.view(-1) for p in self.prev_global_params.values()]
        ).detach()

        grad_list = []
        weight_grad_list = []
        for cid in cids:
            client_params = packages[cid]["regular_model_params"]
            client_vec = torch.cat(
                [p.view(-1) for p in client_params.values()]
            ).detach()
            grad = (prev_vec - client_vec) / lr
            grad_list.append(grad)

            fish = packages[cid].get("fisher_info", torch.ones_like(grad))
            norm_fish = (fish - fish.min()) / (fish.max() - fish.min() + 1e-10)
            weight_grad_list.append(grad * norm_fish)

        weight_global_grad = torch.zeros_like(weight_grad_list[0])
        for wg, w in zip(weight_grad_list, freq):
            weight_global_grad += wg * w

        div_scores = []
        for wg in weight_grad_list:
            score = F.pairwise_distance(
                wg.view(1, -1), weight_global_grad.view(1, -1), p=2
            )
            div_scores.append(score.item())
        div_scores = torch.tensor(div_scores).view(-1, 1)

        fin = FINCH()
        fin.fit(div_scores.numpy())

        if len(fin.partitions) == 0:
            reconstructed_freq = freq
        else:
            partition = fin.partitions["partition_0"]
            evils_center = max(partition["cluster_centers"])
            evils_center_idx = np.where(partition["cluster_centers"] == evils_center)[0]
            evils_idx = partition["cluster_core_indices"][int(evils_center_idx)]
            benign_idx = [i for i in range(len(cids)) if i not in evils_idx]

            self.logger.info(f"FDCR: benign={benign_idx}, evil={evils_idx}")

            freq[evils_idx] = 0
            reconstructed_freq = freq / freq.sum()

            for i in benign_idx:
                cid = cids[i]
                client_params = packages[cid]["regular_model_params"]
                fish = packages[cid].get("fisher_info", torch.ones_like(grad_list[0]))
                norm_fish = (fish - fish.min()) / (fish.max() - fish.min() + 1e-10)
                offset = 0
                for name, prev_param in self.prev_global_params.items():
                    numel = prev_param.numel()
                    size = prev_param.size()
                    weight_para = torch.sigmoid(
                        norm_fish[offset : offset + numel].reshape(size)
                    ) * 2
                    delta = prev_param - client_params[name].detach()
                    client_params[name] = prev_param - delta * weight_para
                    offset += numel

        new_params = OrderedDict()
        for name in self.public_model_params:
            stacked = torch.stack(
                [packages[cids[i]]["regular_model_params"][name].float()
                 for i in range(len(cids))],
                dim=-1,
            )
            new_params[name] = torch.sum(
                stacked * reconstructed_freq.to(stacked.dtype), dim=-1
            ).to(self.public_model_params[name].dtype)
        self._commit_global(new_params)

        self.prev_global_params = OrderedDict(
            (k, v.clone()) for k, v in self.public_model_params.items()
        )


class FDCR_Client(pFL_Client):
    """Client that computes Fisher Information during training."""

    def fit(self) -> None:
        super().fit()
        self._compute_fisher()

    def _compute_fisher(self):
        """Compute diagonal Fisher Information Matrix approximation."""
        self.model.to(self.device)
        self.model.eval()
        fisher = {
            name: torch.zeros_like(p) for name, p in self.model.named_parameters()
        }
        loader = self.load_train_data()
        n_batches = 0
        for batch_x, batch_y, _x_mark, _y_mark in loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            self.model.zero_grad()
            output = self.model(batch_x)
            loss = self.loss(output, batch_y)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.clone() ** 2
            n_batches += 1
        for name in fisher:
            fisher[name] /= max(n_batches, 1)
        self._fisher_info = torch.cat(
            [f.view(-1) for f in fisher.values()]
        ).detach().cpu()
        if self.efficiency != "high":
            self.model.to("cpu")

    def package(self, train_time: float) -> dict:
        result = super().package(train_time)
        if hasattr(self, "_fisher_info"):
            result["fisher_info"] = self._fisher_info
        return result
