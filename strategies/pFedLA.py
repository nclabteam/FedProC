import copy
import time
from argparse import Namespace
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pFL import pFL, pFL_Client

# ---------------------------------------------------------------------------
# pFedLA hypernetwork
# ---------------------------------------------------------------------------


class _pFedLANet(nn.Module):
    """
    pFedLA hypernetwork: embedding -> MLP -> per-block fc heads.

    For client k: alpha[b] is a [C] weight vector (ReLU + normalize).
    Uses U(0,1) init on fc layers to avoid all-negative outputs (kaiming can
    produce all-negative pre-relu outputs, collapsing the aggregation weights).
    """

    def __init__(
        self,
        n_clients: int,
        emb_dim: int,
        hidden_dim: int,
        block_names: List[str],
    ) -> None:
        super().__init__()
        self.block_names = block_names
        self.embeddings = nn.Embedding(n_clients, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc = nn.ModuleDict(
            {b: nn.Linear(hidden_dim, n_clients) for b in block_names}
        )
        for fc in self.fc.values():
            nn.init.uniform_(fc.weight, 0.0, 1.0)
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        emd = self.embeddings(idx)
        feature = self.mlp(emd)
        alpha = {}
        for b in self.block_names:
            raw = self.fc[b](feature)
            pos = F.relu(raw) + 1e-8
            alpha[b] = pos / pos.sum(dim=-1, keepdim=True)
        return alpha


# ---------------------------------------------------------------------------
# pFedLA server
# ---------------------------------------------------------------------------


class pFedLA(pFL):
    """
    pFedLA: Layer-Wise Personalized Federated Learning (Ma et al. NeurIPS 2022).

    A persistent hypernetwork generates per-client per-block aggregation
    weights over all clients' models. The aggregated model is sent to the
    client for local training; Δθ = θ_init − θ_trained is backpropagated
    through the hypernetwork via torch.autograd.grad.

    HeurpFedLA (pfedla_K > 0): K blocks with the highest self-weight
    (alpha[k, k]) are replaced with one-hot weights (pure local, no mixing).
    """

    optional = {
        "pfedla_emb_dim": 8,
        "pfedla_hyper_hid": 64,
        "pfedla_hn_lr": 1e-2,
        "pfedla_inner_lr": 1e-3,
        "pfedla_inner_wd": 1e-5,
        "pfedla_inner_steps": 10,
        "pfedla_K": 0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--pfedla_emb_dim", type=int, default=None)
        parser.add_argument("--pfedla_hyper_hid", type=int, default=None)
        parser.add_argument("--pfedla_hn_lr", type=float, default=None)
        parser.add_argument("--pfedla_inner_lr", type=float, default=None)
        parser.add_argument("--pfedla_inner_wd", type=float, default=None)
        parser.add_argument("--pfedla_inner_steps", type=int, default=None)
        parser.add_argument("--pfedla_K", type=int, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self._hnet: _pFedLANet = None
        self._hnet_opt = None
        self._hnet_sig = None

    @staticmethod
    def _block_name(param_name: str) -> str:
        return param_name.split(".")[0]

    def _ensure_hnet(self, trainable_names: List[str]) -> None:
        C = self.num_clients
        block_names = sorted(set(self._block_name(n) for n in trainable_names))
        sig = (C, tuple(block_names))
        if self._hnet is not None and self._hnet_sig == sig:
            return
        if self._hnet is not None:
            self.logger.warning("[pFedLA] Reinitializing hypernetwork (sig changed)")
        self._hnet = _pFedLANet(
            n_clients=C,
            emb_dim=self.pfedla_emb_dim,
            hidden_dim=self.pfedla_hyper_hid,
            block_names=block_names,
        )
        self._hnet_opt = torch.optim.SGD(self._hnet.parameters(), lr=self.pfedla_hn_lr)
        self._hnet_sig = sig
        n_p = sum(p.numel() for p in self._hnet.parameters())
        self.logger.info(
            f"[pFedLA] Hypernetwork: C={C} blocks={len(block_names)} "
            f"emb_dim={self.pfedla_emb_dim} hidden={self.pfedla_hyper_hid} "
            f"params={n_p:,}"
        )

    def _generate_agg_model(self, client_id: int, trainable_names: List[str]) -> tuple:
        """
        Aggregate all clients' models for client_id using hypernet weights.
        Maintains computation graph through alpha for autograd.grad.
        Returns (agg_state, retain_blocks).
        """
        C = self.num_clients
        idx = torch.tensor([client_id], dtype=torch.long, device=self.device)
        alpha = self._hnet(idx)  # {block: [1, C]}

        # HeurpFedLA: replace K highest self-weight blocks with one-hot
        retain_blocks = []
        K = int(self.pfedla_K)
        if K > 0:
            with torch.no_grad():
                self_w = {b: alpha[b][0, client_id].item() for b in alpha}
            retain_blocks = sorted(self_w, key=lambda b: self_w[b], reverse=True)[:K]
            one_hot = torch.zeros(1, C, device=self.device)
            one_hot[0, client_id] = 1.0
            for b in retain_blocks:
                alpha[b] = one_hot

        agg_state = {}
        for lname in trainable_names:
            block = self._block_name(lname)
            w = alpha[block][0]  # [C], grad flows through hnet
            stacked = torch.stack(
                [
                    self.clients[j].model.state_dict()[lname].to(self.device).float()
                    for j in range(C)
                ]
            )  # [C, ...]
            shape = stacked[0].shape
            agg_state[lname] = (w.view(-1, *([1] * len(shape))) * stacked).sum(0)

        return agg_state, retain_blocks

    def _update_hnet(
        self,
        agg_state: Dict[str, torch.Tensor],
        delta_theta: OrderedDict,
        retain_blocks: List[str],
        trainable_names: List[str],
    ) -> None:
        """Backprop Δθ through agg_state into hypernet parameters."""
        outputs, grad_outputs = [], []
        for lname in trainable_names:
            if self._block_name(lname) in retain_blocks:
                continue
            outputs.append(agg_state[lname])
            grad_outputs.append(delta_theta[lname].to(self.device))

        if not outputs:
            return

        self._hnet_opt.zero_grad()
        hn_grads = torch.autograd.grad(
            outputs=outputs,
            inputs=list(self._hnet.parameters()),
            grad_outputs=grad_outputs,
            allow_unused=True,
        )
        for p, g in zip(self._hnet.parameters(), hn_grads):
            if g is not None:
                p.grad = g
        torch.nn.utils.clip_grad_norm_(self._hnet.parameters(), 50.0)
        self._hnet_opt.step()

    def train(self) -> None:
        trainable_names = [
            n for n, p in self.model.named_parameters() if p.requires_grad
        ]
        self._ensure_hnet(trainable_names)

        for i in range(self.iterations):
            round_start = time.time()
            self.current_iter = i
            self.logger.info("")
            self.logger.info(
                f"-------------Round number: {str(i).zfill(4)}-------------"
            )

            if i % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    self._pre_eval_hook(dataset_type)

            self.select_clients()
            round_send_mb = 0.0

            for client in self.selected_clients:
                self._hnet.train()
                self._hnet.to(self.device)

                agg_state, retain_blocks = self._generate_agg_model(
                    client.id, trainable_names
                )
                weights_cpu = OrderedDict(
                    {k: v.detach().cpu() for k, v in agg_state.items()}
                )
                client.receive_from_server({"weights": weights_cpu})
                round_send_mb += sum(
                    t.element_size() * t.numel() for t in weights_cpu.values()
                ) / (1024**2)

                delta_theta = client.train_inner(
                    inner_steps=self.pfedla_inner_steps,
                    inner_lr=self.pfedla_inner_lr,
                    inner_wd=self.pfedla_inner_wd,
                    device=client.device,
                )
                self._update_hnet(
                    agg_state, delta_theta, retain_blocks, trainable_names
                )
                self._hnet.to("cpu")

            self.metrics["send_mb"].append(round_send_mb)
            self.metrics["time_per_iter"].append(time.time() - round_start)
            self.save_models(save_type="best")
            self.fix_results()
            if self.early_stopping():
                break

        self.post_process()


# ---------------------------------------------------------------------------
# pFedLA client
# ---------------------------------------------------------------------------


class pFedLA_Client(pFL_Client):
    """
    Receives aggregated model weights, trains K inner steps,
    returns Δθ = θ_init − θ_trained for hypernetwork backprop.
    """

    def receive_from_server(self, data: dict) -> None:
        if "weights" in data:
            self.model.load_state_dict(data["weights"])
            self._initial_state = copy.deepcopy(data["weights"])
        else:
            super().receive_from_server(data)

    def train_inner(
        self,
        inner_steps: int,
        inner_lr: float,
        inner_wd: float,
        device: str,
    ) -> OrderedDict:
        self.model.to(device)
        self.model.train()
        opt = torch.optim.SGD(
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

            opt.zero_grad()
            outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
            loss = self.loss(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50.0)
            opt.step()

        final_state = self.model.state_dict()
        delta_theta = OrderedDict(
            {
                k: self._initial_state[k].cpu() - final_state[k].detach().cpu()
                for k in self._initial_state
            }
        )
        self.model.to("cpu")
        return delta_theta
