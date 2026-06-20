from argparse import Namespace
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn

from .pFL import pFL, pFL_Client


class HyperNetwork(nn.Module):
    """
    Generic hypernetwork: client embedding → MLP → one linear head per target
    parameter tensor → generates a full personalized model for each client.
    """

    def __init__(
        self,
        n_nodes: int,
        embedding_dim: int,
        target_model: nn.Module,
        hidden_dim: int = 100,
        n_hidden: int = 3,
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(n_nodes, embedding_dim)

        layers: List[nn.Module] = [
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        self.mlp = nn.Sequential(*layers)

        self.param_names: List[str] = []
        self.param_shapes: List[torch.Size] = []
        self.heads = nn.ModuleList()
        for name, param in target_model.named_parameters():
            self.param_names.append(name)
            self.param_shapes.append(param.shape)
            self.heads.append(nn.Linear(hidden_dim, param.numel()))

        # Retains the (differentiable) output tensors from the last forward pass
        # so train_one_round() can call autograd.grad on them after client training.
        self.outputs: List[torch.Tensor] = []

    def forward(self, client_id: int) -> OrderedDict:
        idx = torch.tensor([client_id], dtype=torch.long)
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        params = []
        weights = OrderedDict()
        for name, shape, head in zip(self.param_names, self.param_shapes, self.heads):
            p = head(features).view(shape)
            params.append(p)
            weights[name] = p.detach().clone().cpu()
        self.outputs = params
        return weights


class pFedHN(pFL):
    """
    Personalized Federated HyperNetworks (pFedHN).

    A server-side hypernetwork h(v_i; φ) generates a full personalized model θ_i
    for each client i. Clients run K inner SGD steps and return the delta
    Δθ_i = θ_i_init − θ_i_trained. The server back-propagates through the
    hypernetwork using Δθ_i as grad_outputs.

    Reference: Shamsian et al., ICML 2021. arXiv 2103.04628.
    """

    optional = {
        "embed_dim": -1,
        "hyper_hid": 100,
        "n_hidden": 3,
        "hn_lr": 1e-2,
        "hn_wd": 1e-3,
        "embed_lr": -1.0,
        "norm_clip": 50,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--embed_dim", type=int, default=None)
        parser.add_argument("--hyper_hid", type=int, default=None)
        parser.add_argument("--n_hidden", type=int, default=None)
        parser.add_argument("--hn_lr", type=float, default=None)
        parser.add_argument("--hn_wd", type=float, default=None)
        parser.add_argument("--embed_lr", type=float, default=None)
        parser.add_argument("--norm_clip", type=int, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)

        embed_dim = (
            self.embed_dim if self.embed_dim > 0 else int(1 + self.num_clients / 4)
        )
        embed_lr = self.embed_lr if self.embed_lr > 0 else self.hn_lr

        self.hnet = HyperNetwork(
            n_nodes=self.num_clients,
            embedding_dim=embed_dim,
            target_model=self.model,
            hidden_dim=self.hyper_hid,
            n_hidden=self.n_hidden,
        ).to(self.device)

        non_embed = [p for n, p in self.hnet.named_parameters() if "embed" not in n]
        embed_p = [p for n, p in self.hnet.named_parameters() if "embed" in n]
        self.hn_optimizer = torch.optim.SGD(
            [{"params": non_embed}, {"params": embed_p, "lr": embed_lr}],
            lr=self.hn_lr,
            momentum=0.9,
            weight_decay=self.hn_wd,
        )

        n_hn = sum(p.numel() for p in self.hnet.parameters())
        self.logger.info(
            f"HyperNetwork: embed_dim={embed_dim}, hidden={self.hyper_hid}, "
            f"n_hidden={self.n_hidden}, total_params={n_hn:,}"
        )

    def package(self, client_id: int) -> dict:
        result = super().package(client_id)
        # Run HN forward (with grad) and inject generated weights as regular_model_params.
        # self.hnet.outputs retains the computation graph for this client's forward pass.
        self.hnet.train()
        self.hnet.to(self.device)
        result["regular_model_params"] = self.hnet(client_id)
        return result

    def train_one_round(self) -> None:
        # Process clients sequentially so the HN computation graph for client i
        # is still alive when we do backprop for client i.
        for client_id in self.selected_clients:
            # package() runs HN forward and stores graph in self.hnet.outputs
            packages = self.trainer.train([client_id])
            pkg = packages[client_id]

            hn_grads = torch.autograd.grad(
                outputs=self.hnet.outputs,
                inputs=list(self.hnet.parameters()),
                grad_outputs=[
                    g.to(self.device)
                    for g in pkg["model_params_diff"].values()
                ],
                allow_unused=True,
            )
            self.hn_optimizer.zero_grad()
            for param, grad in zip(self.hnet.parameters(), hn_grads):
                if grad is not None:
                    param.grad = grad
            torch.nn.utils.clip_grad_norm_(self.hnet.parameters(), self.norm_clip)
            self.hn_optimizer.step()
            self.hnet.to("cpu")

            # Write back personal model params (optimizer/scheduler state)
            self.trainer._write_back(client_id, pkg)

    def aggregate_client_updates(self, packages) -> None:
        # All aggregation done per-client in train_one_round(); this is a no-op.
        pass

    def _pre_eval_hook(self, dataset_type: str) -> None:
        # Sync clients_personal_model_params with current HN weights before eval.
        self.hnet.eval()
        self.hnet.to(self.device)
        with torch.no_grad():
            for cid in range(self.num_clients):
                weights = self.hnet(cid)
                self.clients_personal_model_params[cid] = weights
        self.hnet.to("cpu")
        super()._pre_eval_hook(dataset_type)

    def save_models(self, save_type: str) -> None:
        if save_type not in ["last", "best"]:
            raise ValueError("save_type must be 'last' or 'best'")
        if save_type == "best":
            metric_key = "personal_avg_test_loss"
            if metric_key not in self.metrics or not self.metrics[metric_key]:
                return
            vals = self.metrics[metric_key]
            if vals[-1] != min(vals):
                return
        self.save_model(
            model=self.hnet,
            path=self.model_path,
            name=self.name + "_hnet",
            postfix=save_type,
            configs=self.configs,
            metadata={"save_type": save_type, "owner": "server_hnet"},
            verbose=self.logger,
        )


class pFedHN_Client(pFL_Client):
    """
    Client for pFedHN. Receives HN-generated weights, runs local training,
    and returns Δθ = θ_init − θ_trained via the return_diff mechanism.
    """

    return_diff: bool = True
