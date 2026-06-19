import copy
import time
from argparse import Namespace
from typing import Any, Dict, List, Optional

import torch

from .pFL import pFL, pFL_Client


class AirMetapFL(pFL):
    """
    Air-meta-pFL: Over-the-Air Meta-Learning Based Personalized Federated Learning.

    Implements Algorithm 1 from Wen et al. (2024), arXiv 2406.11569, adapted for
    digital channels (OTA wireless channel model omitted). Each client runs Q outer
    MAML gradient steps and applies top-k sparsification with error-feedback memory
    before sending model differences. The server performs uniform averaging.

    Two MAML variants via --hf:
    - FO (default): first-order — 2 mini-batches per outer step.
    - HF  (--hf):  Hessian-free correction via finite differences — 3 mini-batches.
      g_hf = g - (α/2δ)·(∇f(θ+δg) - ∇f(θ-δg))

    Reference: "Pre-Training and Personalized Fine-Tuning via Over-the-Air Federated
    Meta-Learning: Convergence-Generalization Trade-Offs", arXiv 2406.11569.
    """

    optional = {
        "alpha": 0.01,
        "sparsity": 1.0,
        "hf": False,
        "delta": 1e-3,
    }
    compulsory = {"return_diff": True}

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "--alpha",
            type=float,
            default=None,
            help="Inner adaptation learning rate α (default: 0.01)",
        )
        parser.add_argument(
            "--sparsity",
            type=float,
            default=None,
            help="Fraction of entries to keep via top-k (1.0 = no sparsification)",
        )
        parser.add_argument(
            "--hf",
            action="store_true",
            default=None,
            help="Hessian-free second-order MAML correction (3 mini-batches/step)",
        )
        parser.add_argument(
            "--delta",
            type=float,
            default=None,
            help="Finite-difference perturbation for HF Hessian approximation",
        )
        return parser

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False

    def calculate_aggregation_weights(self) -> None:
        pass  # uniform 1/n weighting applied directly in aggregate_models

    def aggregate_models(self) -> None:
        n = len(self.client_data)
        if n == 0:
            return
        for cd in self.client_data:
            for global_param, local_param in zip(
                self.model.parameters(), cd["model"].parameters()
            ):
                global_param.data.sub_(
                    local_param.data.to(global_param.device), alpha=1.0 / n
                )


class AirMetapFL_Client(pFL_Client):
    """
    Client for Air-meta-pFL.

    Runs MAML outer gradient steps (FO or HF) and applies top-k sparsification
    with error-feedback memory before transmitting the model difference Δ_i.
    The memory vector m_i persists across rounds to accumulate sparsification error.
    """

    _memory: Optional[List[torch.Tensor]] = None

    def __init__(self, configs: Namespace, id: int, times: int) -> None:
        super().__init__(configs=configs, id=id, times=times)
        if self.hf:
            self._model_plus = copy.deepcopy(self.model)
            self._model_minus = copy.deepcopy(self.model)

    @staticmethod
    def _top_k_sparsify(tensor: torch.Tensor, k_ratio: float) -> torch.Tensor:
        """Keep top-k absolute-value entries, zero the rest."""
        flat = tensor.flatten()
        k = max(1, int(k_ratio * flat.numel()))
        _, idx = torch.topk(flat.abs(), k)
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        return (flat * mask).reshape(tensor.shape)

    def _next_batch(self, iterator, loader):
        try:
            return next(iterator)
        except StopIteration:
            return next(iter(loader))

    def train(self) -> Optional[Dict[str, Any]]:
        train_loader = self.load_train_data()
        start_time = time.time()
        self.model.to(self.device)
        self.model.train()

        if self.hf:
            self._train_hf(train_loader)
        else:
            self._train_fo(train_loader)

        self.scheduler.step()
        if self.efficiency != "high":
            self.model.to("cpu")
        self.metrics["train_time"].append(time.time() - start_time)
        return None

    def _train_fo(self, train_loader) -> None:
        """First-order MAML: 2 mini-batches per outer step."""
        lr = self.learning_rate

        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)

                half = batch_x.size(0) // 2 or batch_x.size(0)

                # Save θ
                temp_params = [p.data.clone() for p in self.model.parameters()]

                # Inner step: θ' = θ - α∇f(θ; D^tr)
                self.optimizer.zero_grad()
                out = self.model(
                    batch_x[:half], x_mark=x_mark[:half], y_mark=y_mark[:half]
                )
                self.loss(out, batch_y[:half]).backward()
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.data.sub_(self.alpha * p.grad)

                # Meta-gradient at θ' on query set: ∇f(θ'; D^va)
                self.optimizer.zero_grad()
                x_q = batch_x[half:] if half < batch_x.size(0) else batch_x
                y_q = batch_y[half:] if half < batch_x.size(0) else batch_y
                xm_q = x_mark[half:] if half < batch_x.size(0) else x_mark
                ym_q = y_mark[half:] if half < batch_x.size(0) else y_mark
                out2 = self.model(x_q, x_mark=xm_q, y_mark=ym_q)
                self.loss(out2, y_q).backward()

                # Restore θ, apply outer step η
                with torch.no_grad():
                    for p, tp in zip(self.model.parameters(), temp_params):
                        p.data.copy_(tp)
                        if p.grad is not None:
                            p.data.sub_(lr * p.grad)

    def _train_hf(self, train_loader) -> None:
        """Hessian-free MAML: 3 mini-batches per outer step."""
        lr = self.learning_rate

        self._model_plus.to(self.device)
        self._model_minus.to(self.device)
        self._model_plus.train()
        self._model_minus.train()

        steps_per_epoch = max(1, len(train_loader) // 3)

        for _ in range(self.epochs):
            iterator = iter(train_loader)
            for _ in range(steps_per_epoch):
                # Batch 0: support set → inner step
                bx0, by0, bxm0, bym0 = self._next_batch(iterator, train_loader)
                bx0 = bx0.to(device=self.device, dtype=torch.float32)
                by0 = by0.to(device=self.device, dtype=torch.float32)
                bxm0 = bxm0.to(device=self.device, dtype=torch.float32)
                bym0 = bym0.to(device=self.device, dtype=torch.float32)

                frz_params = [p.data.clone() for p in self.model.parameters()]
                self.optimizer.zero_grad()
                self.loss(self.model(bx0, x_mark=bxm0, y_mark=bym0), by0).backward()
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.data.sub_(self.alpha * p.grad)

                # Batch 1: query set → meta-gradient g = ∇f(θ'; D^va)
                bx1, by1, bxm1, bym1 = self._next_batch(iterator, train_loader)
                bx1 = bx1.to(device=self.device, dtype=torch.float32)
                by1 = by1.to(device=self.device, dtype=torch.float32)
                bxm1 = bxm1.to(device=self.device, dtype=torch.float32)
                bym1 = bym1.to(device=self.device, dtype=torch.float32)

                self.optimizer.zero_grad()
                self.loss(self.model(bx1, x_mark=bxm1, y_mark=bym1), by1).backward()
                meta_grads = [
                    p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                    for p in self.model.parameters()
                ]

                # Batch 2: Hessian finite-difference at θ (frz_params)
                bx2, by2, bxm2, bym2 = self._next_batch(iterator, train_loader)
                bx2 = bx2.to(device=self.device, dtype=torch.float32)
                by2 = by2.to(device=self.device, dtype=torch.float32)
                bxm2 = bxm2.to(device=self.device, dtype=torch.float32)
                bym2 = bym2.to(device=self.device, dtype=torch.float32)

                with torch.no_grad():
                    for pp, pm, g, fp in zip(
                        self._model_plus.parameters(),
                        self._model_minus.parameters(),
                        meta_grads,
                        frz_params,
                    ):
                        pp.data.copy_(fp + self.delta * g)
                        pm.data.copy_(fp - self.delta * g)

                self._model_plus.zero_grad()
                self._model_minus.zero_grad()
                self.loss(
                    self._model_plus(bx2, x_mark=bxm2, y_mark=bym2), by2
                ).backward()
                self.loss(
                    self._model_minus(bx2, x_mark=bxm2, y_mark=bym2), by2
                ).backward()

                # g_hf = g - (α/2δ)·(∇f(θ+δg) - ∇f(θ-δg))
                hf_coef = self.alpha / (2.0 * self.delta)
                hf_grads = [
                    g
                    - hf_coef
                    * (
                        (pp.grad if pp.grad is not None else torch.zeros_like(g))
                        - (pm.grad if pm.grad is not None else torch.zeros_like(g))
                    )
                    for g, pp, pm in zip(
                        meta_grads,
                        self._model_plus.parameters(),
                        self._model_minus.parameters(),
                    )
                ]

                # Restore θ, apply outer step η with HF gradient
                with torch.no_grad():
                    for p, fp, g_hf in zip(
                        self.model.parameters(), frz_params, hf_grads
                    ):
                        p.data.copy_(fp - lr * g_hf)

        self._model_plus.to("cpu")
        self._model_minus.to("cpu")

    def variables_to_be_sent(self) -> Dict[str, Any]:
        data = super().variables_to_be_sent()

        if self.sparsity < 1.0:
            if self._memory is None:
                self._memory = [
                    torch.zeros_like(p.data.cpu()) for p in data["model"].parameters()
                ]
            model = data["model"]
            with torch.no_grad():
                new_mem = []
                for mem, param in zip(self._memory, model.parameters()):
                    combined = mem.to(param.device) + param.data
                    sparsified = self._top_k_sparsify(combined, self.sparsity)
                    new_mem.append((combined - sparsified).cpu())
                    param.data.copy_(sparsified)
                self._memory = new_mem

        return data
