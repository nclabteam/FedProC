from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pFL import pFL, pFL_Client


class pFedLAHyperNetwork(nn.Module):
    """Dedicated pFedLA hypernetwork with one head per model block."""

    def __init__(
        self,
        n_clients: int,
        embedding_dim: int,
        hidden_dim: int,
        layer_names: Iterable[str],
        retained_layers: int = 0,
    ) -> None:
        super().__init__()
        self.n_clients = n_clients
        self.layer_names = tuple(layer_names)
        if not self.layer_names:
            raise ValueError("pFedLA requires at least one trainable model layer")
        if not 0 <= retained_layers <= len(self.layer_names):
            raise ValueError(
                "retained_layers must be between 0 and the number of layers"
            )
        self.retained_layers = retained_layers
        self.embeddings = nn.Embedding(n_clients, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_layers = nn.ModuleList(
            nn.Linear(hidden_dim, n_clients) for _ in self.layer_names
        )
        for layer in self.fc_layers:
            nn.init.uniform_(layer.weight, 0.0, 1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, client_id: int) -> tuple[OrderedDict, set[str]]:
        device = self.embeddings.weight.device
        embedding = self.embeddings(
            torch.tensor(client_id, dtype=torch.long, device=device)
        )
        feature = self.mlp(embedding)
        alpha = OrderedDict(
            (name, F.relu(layer(feature)))
            for name, layer in zip(self.layer_names, self.fc_layers)
        )
        retained = set()
        if self.retained_layers:
            self_weights = torch.stack(
                [weights[client_id].detach() for weights in alpha.values()]
            )
            indices = torch.topk(
                self_weights,
                self.retained_layers,
                sorted=False,
            ).indices.tolist()
            local_only = torch.zeros(
                self.n_clients,
                dtype=feature.dtype,
                device=device,
            )
            local_only[client_id] = 1.0
            for index in indices:
                name = self.layer_names[index]
                alpha[name] = local_only
                retained.add(name)
        return alpha, retained


class pFedLA(pFL):
    """Layer-wised personalized aggregation (Ma et al., CVPR 2022)."""

    compulsory = {"return_diff": True}
    optional = {
        "pfedla_emb_dim": 8,
        "pfedla_hyper_hid": 64,
        "pfedla_hn_lr": 1e-2,
        "pfedla_K": 0,
        "norm_clip": 50.0,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--pfedla_emb_dim", type=int, default=None)
        parser.add_argument("--pfedla_hyper_hid", type=int, default=None)
        parser.add_argument("--pfedla_hn_lr", type=float, default=None)
        parser.add_argument("--pfedla_K", type=int, default=None)
        parser.add_argument("--norm_clip", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        initial = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in self.public_model_params.items()
        )
        for client_id in range(self.num_clients):
            self.clients_personal_model_params[client_id].update(deepcopy(initial))

        layer_names = tuple(dict.fromkeys(name.split(".")[0] for name in initial))
        self._hnet = pFedLAHyperNetwork(
            n_clients=self.num_clients,
            embedding_dim=self.pfedla_emb_dim,
            hidden_dim=self.pfedla_hyper_hid,
            layer_names=layer_names,
            retained_layers=self.pfedla_K,
        ).to(self.device)
        self._hnet_opt = torch.optim.SGD(self._hnet.parameters(), lr=self.pfedla_hn_lr)
        private_state = self._private_hypernetwork_state(hypernetwork=self._hnet)
        self._client_hnet_params: Dict[int, OrderedDict] = {
            client_id: deepcopy(private_state) for client_id in range(self.num_clients)
        }
        self._aggregated_params: OrderedDict = OrderedDict()
        self._initial_model_params: OrderedDict = OrderedDict()
        self._round_model_params = None

        parameter_count = sum(
            parameter.numel() for parameter in self._hnet.parameters()
        )
        self.logger.info(
            f"[pFedLA] HN: C={self.num_clients} layers={len(layer_names)} "
            f"emb={self.pfedla_emb_dim} hidden={self.pfedla_hyper_hid} "
            f"K={self.pfedla_K} params={parameter_count:,}"
        )

    def select_clients(self) -> None:
        self._select_all_clients()

    @staticmethod
    def _private_hypernetwork_state(
        hypernetwork: pFedLAHyperNetwork,
    ) -> OrderedDict:
        return OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in hypernetwork.state_dict().items()
            if name != "embeddings.weight"
        )

    def _aggregate_model(
        self,
        client_id: int,
        alpha: OrderedDict,
    ) -> OrderedDict:
        source_models = (
            self._round_model_params
            if self._round_model_params is not None
            else self.clients_personal_model_params
        )
        source_ids = [
            source_id
            for source_id in range(self.num_clients)
            if not self.is_new[source_id]
        ]
        aggregated = OrderedDict()
        for name in self.public_model_params:
            layer_name = name.split(".")[0]
            weights = alpha[layer_name][source_ids]
            total = weights.sum()
            if not torch.isfinite(total) or total.item() <= 0:
                raise RuntimeError(
                    f"client {client_id} has zero pFedLA weights for "
                    f"layer {layer_name}"
                )
            values = torch.stack(
                [
                    source_models[source_id][name].to(self.device).float()
                    for source_id in source_ids
                ]
            )
            normalized = weights / total
            aggregated[name] = torch.sum(
                values * normalized.view(-1, *([1] * (values.ndim - 1))),
                dim=0,
            )
        return aggregated

    def package(self, client_id: int) -> dict:
        package = super().package(client_id=client_id)
        self._hnet.load_state_dict(self._client_hnet_params[client_id], strict=False)
        self._hnet.train()
        alpha, retained = self._hnet(client_id)
        aggregated = self._aggregate_model(client_id=client_id, alpha=alpha)
        self._initial_model_params = OrderedDict(
            (name, value.detach().cpu().clone()) for name, value in aggregated.items()
        )
        self._aggregated_params = OrderedDict(
            (name, value)
            for name, value in aggregated.items()
            if name.split(".")[0] not in retained
        )
        package["regular_model_params"] = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in aggregated.items()
            if name.split(".")[0] not in retained
        )
        package["personal_model_params"] = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in aggregated.items()
            if name.split(".")[0] in retained
        )
        package["__wire__"] = ("regular_model_params",)
        return package

    def train_one_round(self) -> dict:
        self._round_model_params = {
            client_id: dict(parameters)
            for client_id, parameters in self.clients_personal_model_params.items()
        }
        all_packages = {}
        for client_id in self.selected_clients:
            package = self.trainer.train([client_id])[client_id]
            difference = package["model_params_diff"]

            if self._aggregated_params:
                self._hnet_opt.zero_grad()
                gradients = torch.autograd.grad(
                    outputs=list(self._aggregated_params.values()),
                    inputs=list(self._hnet.parameters()),
                    grad_outputs=[
                        -difference[name].to(self.device)
                        for name in self._aggregated_params
                    ],
                    allow_unused=True,
                )
                for parameter, gradient in zip(self._hnet.parameters(), gradients):
                    if gradient is not None:
                        parameter.grad = gradient
                torch.nn.utils.clip_grad_norm_(self._hnet.parameters(), self.norm_clip)
                self._hnet_opt.step()

            self._client_hnet_params[client_id] = self._private_hypernetwork_state(
                hypernetwork=self._hnet
            )
            trained = OrderedDict(
                (
                    name,
                    self._initial_model_params[name] - difference[name],
                )
                for name in difference
            )
            self.clients_personal_model_params[client_id].update(trained)
            all_packages[client_id] = package
        self._round_model_params = None
        return all_packages

    def aggregate_client_updates(self, packages: Any) -> None:
        """Aggregation is completed inside ``train_one_round``."""


class pFedLA_Client(pFL_Client):
    """Train the server-generated personalized model and return its update."""

    return_diff = True
    return_diff_score = False
