import copy
import time
from argparse import Namespace
from collections import OrderedDict
from typing import List

import numpy as np
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

    def forward(self, idx: torch.Tensor) -> OrderedDict:
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        weights = OrderedDict()
        for name, shape, head in zip(self.param_names, self.param_shapes, self.heads):
            weights[name] = head(features).view(shape)
        return weights


class pFedHN(pFL):
    """
    Personalized Federated HyperNetworks (pFedHN).

    A server-side hypernetwork h(v_i; φ) generates a full personalized model
    θ_i for each client i.  Per round the server selects one client, forwards
    the hypernetwork to produce θ_i, sends it to the client which performs K
    local gradient steps to get θ̃_i, then the server back-propagates through
    the hypernetwork using Δθ_i = θ_i − θ̃_i as the gradient signal.

    Reference: Shamsian et al., "Personalized Federated Learning using
    Hypernetworks", ICML 2021.  arXiv 2103.04628.
    """

    optional = {
        "embed_dim": -1,
        "hyper_hid": 100,
        "n_hidden": 3,
        "inner_steps": 50,
        "inner_lr": 5e-3,
        "inner_wd": 5e-5,
        "embed_lr": -1.0,
        "hn_lr": 1e-2,
        "hn_wd": 1e-3,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--embed_dim", type=int, default=None)
        parser.add_argument("--hyper_hid", type=int, default=None)
        parser.add_argument("--n_hidden", type=int, default=None)
        parser.add_argument("--inner_steps", type=int, default=None)
        parser.add_argument("--inner_lr", type=float, default=None)
        parser.add_argument("--inner_wd", type=float, default=None)
        parser.add_argument("--embed_lr", type=float, default=None)
        parser.add_argument("--hn_lr", type=float, default=None)
        parser.add_argument("--hn_wd", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)

        embed_dim = self.embed_dim if self.embed_dim > 0 else int(1 + self.num_clients / 4)
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

    @torch.no_grad()
    def _sync_client_models(self) -> None:
        """Generate HN weights for every client and load into client.model."""
        self.hnet.eval()
        self.hnet.to(self.device)
        for client in self.clients:
            idx = torch.tensor([client.id], dtype=torch.long, device=self.device)
            weights = self.hnet(idx)
            client.model.load_state_dict(
                OrderedDict({k: v.detach().cpu() for k, v in weights.items()})
            )
        self.hnet.to("cpu")

    def _pre_eval_hook(self, dataset_type: str) -> None:
        self._sync_client_models()
        self.evaluate_personalization_loss(dataset_type)

    def train(self) -> None:
        for i in range(self.iterations):
            round_start_time = time.time()
            self.current_iter = i
            self.logger.info("")
            self.logger.info(
                f"-------------Round number: {str(i).zfill(4)}-------------"
            )

            # Evaluate using last round's HN weights (before this round's update)
            if i % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    self._pre_eval_hook(dataset_type)

            # --- pFedHN round: sample one client at random ---
            node_id = int(np.random.randint(0, self.num_clients))
            client = self.clients[node_id]
            client.current_iter = i

            self.hnet.train()
            self.hnet.to(self.device)

            # Forward pass: generate personalized weights (keep computation graph)
            idx = torch.tensor([node_id], dtype=torch.long, device=self.device)
            weights = self.hnet(idx)

            # Send weights to client; client stores initial state internally
            client.receive_from_server(
                {"weights": OrderedDict({k: v.detach().cpu() for k, v in weights.items()})}
            )

            # Client runs K local steps and returns Δθ = θ_initial − θ_final
            delta_theta = client.train_inner(
                inner_steps=self.inner_steps,
                inner_lr=self.inner_lr,
                inner_wd=self.inner_wd,
                device=client.device,
            )

            # Back-prop through HN: Δθ as grad_outputs computes ∂L̃/∂φ
            self.hn_optimizer.zero_grad()
            hnet_grads = torch.autograd.grad(
                outputs=list(weights.values()),
                inputs=list(self.hnet.parameters()),
                grad_outputs=[g.to(self.device) for g in delta_theta.values()],
            )
            for p, g in zip(self.hnet.parameters(), hnet_grads):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(self.hnet.parameters(), 50)
            self.hn_optimizer.step()
            self.hnet.to("cpu")

            # Track communication cost (size of target weights sent)
            send_mb = (
                sum(t.element_size() * t.numel() for t in delta_theta.values())
                / (1024 ** 2)
            )
            self.metrics["send_mb"].append(send_mb)
            client.metrics["train_time"].append(time.time() - round_start_time)

            self.save_models(save_type="best")
            round_duration = time.time() - round_start_time
            self.metrics["time_per_iter"].append(round_duration)
            self.logger.info(f"Node {node_id} | Time cost: {round_duration:.4f}s")
            self.fix_results()
            if self.early_stopping():
                break

        # Sync all client models with the final HN before saving
        self._sync_client_models()
        self.post_process()

    def save_models(self, save_type: str) -> None:
        if save_type not in ["last", "best"]:
            raise ValueError("save_type must be 'last' or 'best'")

        should_save = True
        if save_type == "best":
            metric_key = "personal_avg_test_loss"
            if metric_key not in self.metrics or not self.metrics[metric_key]:
                should_save = False
            else:
                vals = self.metrics[metric_key]
                if vals[-1] != min(vals):
                    should_save = False

        if not should_save:
            return

        # Sync clients before saving so models reflect current HN
        if save_type == "last":
            self._sync_client_models()

        # Save hypernetwork as the server artifact
        self.save_model(
            model=self.hnet,
            path=self.model_path,
            name=self.name + "_hnet",
            postfix=save_type,
            configs=self.configs,
            metadata={"save_type": save_type, "owner": "server_hnet"},
            verbose=self.logger,
        )
        for client in self.clients:
            client.save_model(
                model=client.model,
                path=client.model_path,
                name=client.name,
                postfix=save_type,
                configs=client.configs,
                metadata={"save_type": save_type, "owner": client.name},
                verbose=client.logger,
            )


class pFedHN_Client(pFL_Client):
    """
    Client for pFedHN.  Receives generated weights from the server hypernetwork,
    runs K inner gradient steps, and returns Δθ = θ_initial − θ_final.
    """

    def receive_from_server(self, data: dict) -> None:
        if "weights" in data:
            self.model.load_state_dict(data["weights"])
            self._hn_initial_state = copy.deepcopy(data["weights"])
        else:
            super().receive_from_server(data)

    def train_inner(
        self,
        inner_steps: int,
        inner_lr: float,
        inner_wd: float,
        device: str,
    ) -> OrderedDict:
        """
        Run K local SGD steps on private data and return
        Δθ = θ_initial − θ_final, used as grad_outputs for HN backprop.
        """
        self.model.to(device)
        self.model.train()
        inner_optim = torch.optim.SGD(
            self.model.parameters(),
            lr=inner_lr,
            momentum=0.9,
            weight_decay=inner_wd,
        )
        loader = self.load_train_data()
        loader_iter = iter(loader)

        for _ in range(inner_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            batch_x, batch_y, x_mark, y_mark = batch
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.float32)
            x_mark = x_mark.to(device=device, dtype=torch.float32)
            y_mark = y_mark.to(device=device, dtype=torch.float32)

            inner_optim.zero_grad()
            outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
            loss = self.loss(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
            inner_optim.step()

        final_state = self.model.state_dict()
        delta_theta = OrderedDict(
            {
                k: self._hn_initial_state[k].cpu() - final_state[k].detach().cpu()
                for k in self._hn_initial_state
            }
        )
        self.model.to("cpu")
        return delta_theta
