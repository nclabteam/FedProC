from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pFL import pFL, pFL_Client


class _pFedLANet(nn.Module):
    """
    pFedLA hypernetwork: embedding -> MLP -> per-layer fc heads.

    alpha[b] is a softmax-like weight vector over clients for layer b.
    Uses U(0,1) init on fc layers to avoid all-negative pre-relu outputs.
    """

    def __init__(
        self,
        n_clients: int,
        emb_dim: int,
        hidden_dim: int,
        layer_num: int,
        K: int = 0,
    ) -> None:
        super().__init__()
        self.K = K
        self.n_clients = n_clients
        self.layer_num = layer_num
        self.embeddings = nn.Embedding(n_clients, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_layers = nn.ParameterList(
            nn.Linear(hidden_dim, n_clients) for _ in range(layer_num)
        )
        for fc in self.fc_layers:
            nn.init.uniform_(fc.weight, 0.0, 1.0)
            nn.init.zeros_(fc.bias)

    def forward(self, client_id: int) -> List[torch.Tensor]:
        emd = self.embeddings(torch.tensor(client_id, dtype=torch.long))
        feature = self.mlp(emd)
        weights = [F.relu(fc(feature)) for fc in self.fc_layers]

        if self.K > 0:
            default_weight = torch.zeros(self.n_clients, dtype=torch.float)
            default_weight[client_id] = 1.0
            self_weights = torch.tensor([w[client_id].item() for w in weights])
            topk_idx = torch.topk(self_weights, self.K, sorted=False)[1]
            for i in topk_idx:
                weights[i] = (weights[i] * default_weight).detach().requires_grad_(True)

        return weights


class pFedLA(pFL):
    """pFedLA: Layer-Wised Model Aggregation for Personalized Federated Learning (Ma et al., CVPR 2022).

    Each client has a dedicated hypernetwork (HN) on the server. The HN maps a
    client embedding → per-layer aggregation weights α over all clients' stored params.
    Per round: clients train locally and return Δθ; server backpropagates -Δθ through
    the HN (via torch.autograd.grad) to update the HN. Updated HN generates new
    per-layer personalized model for next round.

    HeurpFedLA (pfedla_K > 0): top-K layers by self-weight keep one-hot weights
    (no mixing for those layers).

    Reference: arXiv:2205.03993. CVPR 2022.
    """

    optional = {
        "pfedla_emb_dim": 8,
        "pfedla_hyper_hid": 64,
        "pfedla_hn_lr": 1e-2,
        "pfedla_K": 0,
        "norm_clip": 50.0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--pfedla_emb_dim", type=int, default=None)
        parser.add_argument("--pfedla_hyper_hid", type=int, default=None)
        parser.add_argument("--pfedla_hn_lr", type=float, default=None)
        parser.add_argument("--pfedla_K", type=int, default=None)
        parser.add_argument("--norm_clip", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        init_params = {k: v.cpu().clone() for k, v in self.public_model_params.items()}
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid].update(
                {k: v.clone() for k, v in init_params.items()}
            )

        layer_num = len(list(self.model.named_parameters()))
        self._hnet = _pFedLANet(
            n_clients=self.num_clients,
            emb_dim=self.pfedla_emb_dim,
            hidden_dim=self.pfedla_hyper_hid,
            layer_num=layer_num,
            K=self.pfedla_K,
        )
        self._hnet_opt = torch.optim.SGD(
            self._hnet.parameters(), lr=self.pfedla_hn_lr
        )
        # Per-client snapshot of the hypernet (so each client gets its own HN)
        self._client_hnet_params: Dict[int, dict] = {
            i: deepcopy(self._hnet.state_dict())
            for i in range(self.num_clients)
        }
        # Stores differentiable aggregated params for current client (for autograd)
        self._agg_params: List[torch.Tensor] = []

        n_p = sum(p.numel() for p in self._hnet.parameters())
        self.logger.info(
            f"[pFedLA] HN: C={self.num_clients} layers={layer_num} "
            f"emb={self.pfedla_emb_dim} hidden={self.pfedla_hyper_hid} "
            f"K={self.pfedla_K} params={n_p:,}"
        )

    def package(self, client_id: int) -> dict:
        result = super().package(client_id)

        # Load this client's personal HN snapshot
        self._hnet.load_state_dict(self._client_hnet_params[client_id])
        self._hnet.train()
        self._hnet.to(self.device)

        param_names = [n for n, _ in self.model.named_parameters()]
        alpha = self._hnet(client_id)  # list[Tensor[n_clients]], grad retained

        # Aggregate all clients' stored params using HN weights (keeps grad graph)
        agg = OrderedDict()
        for i, name in enumerate(param_names):
            stacked = torch.stack(
                [
                    self.clients_personal_model_params[j][name]
                    .to(self.device)
                    .float()
                    for j in range(self.num_clients)
                ]
            )  # [C, ...]
            w = alpha[i]
            w_sum = w.sum()
            if w_sum == 0:
                w = torch.zeros_like(w)
                w[client_id] = 1.0
            else:
                w = w / w_sum
            shape = stacked[0].shape
            agg[name] = (w.view(-1, *([1] * len(shape))) * stacked).sum(0)

        self._agg_params = list(agg.values())

        result["regular_model_params"] = OrderedDict(
            (k, v.detach().clone().cpu()) for k, v in agg.items()
        )
        return result

    def train_one_round(self) -> dict:
        all_packages = {}
        for client_id in self.selected_clients:
            # package() loads client HN, runs forward, stores graph in self._agg_params
            packages = self.trainer.train([client_id])
            pkg = packages[client_id]

            self._hnet_opt.zero_grad()
            hn_grads = torch.autograd.grad(
                outputs=self._agg_params,
                inputs=list(self._hnet.parameters()),
                grad_outputs=[
                    -diff.to(self.device)
                    for diff in pkg["model_params_diff"].values()
                ],
                allow_unused=True,
            )
            for param, grad in zip(self._hnet.parameters(), hn_grads):
                if grad is not None:
                    param.grad = grad
            torch.nn.utils.clip_grad_norm_(self._hnet.parameters(), self.norm_clip)
            self._hnet_opt.step()
            self._hnet.to("cpu")

            # Save updated HN snapshot for this client
            self._client_hnet_params[client_id] = deepcopy(self._hnet.state_dict())

            # Store trained params in personal_model_params (nFL-style).
            # return_diff=True so pkg["regular_model_params"] is empty;
            # reconstruct: trained = initial - diff.
            diff = pkg["model_params_diff"]
            trained = OrderedDict(
                (k, self.clients_personal_model_params[client_id][k] - diff[k])
                for k in diff
            )
            self.clients_personal_model_params[client_id].update(trained)

            self.trainer._write_back(client_id, pkg)
            all_packages[client_id] = pkg
        return all_packages

    def aggregate_client_updates(self, packages) -> None:
        pass  # All done in train_one_round()


class pFedLA_Client(pFL_Client):
    """Receives aggregated model, runs local training, returns Δθ."""

    return_diff: bool = True
