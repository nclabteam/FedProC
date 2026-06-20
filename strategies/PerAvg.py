import copy
from argparse import Namespace

import torch

from .tFL import tFL, tFL_Client


class PerAvg(tFL):
    """
    Per-FedAvg: Personalized Federated Averaging via Model-Agnostic Meta-Learning.

    Trains a global initialization θ such that one local gradient step from θ
    yields a good personalized model for each client.

    Two variants controlled by --hf flag:
    - FO  (default): first-order MAML — uses 2 batches per meta-step.
    - HF  (--hf):    Hessian-Free correction via finite differences — uses 3
                     batches per meta-step.  g_hf = g - (α/2δ)·(∇f(w+δg) - ∇f(w-δg)).

    Reference: Fallah et al., "Personalized Federated Learning with Theoretical
    Guarantees: A Model-Agnostic Meta-Learning Approach", NeurIPS 2020.
    arXiv 2002.07948.
    """

    optional = {
        "beta": 1e-3,
        "hf": False,
        "delta": 1e-3,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--beta", type=float, default=None)
        parser.add_argument("--hf", action="store_true", default=None)
        parser.add_argument("--delta", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False


class PerAvg_Client(tFL_Client):
    """
    Client for PerAvg. Supports both FO and HF variants via self.hf flag.
    """

    hf: bool = False

    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        if self.hf:
            self._model_plus = copy.deepcopy(self.model)
            self._model_minus = copy.deepcopy(self.model)

    def _next_batch(self, iterator, loader):
        try:
            return next(iterator)
        except StopIteration:
            return next(iter(loader))

    def fit(self) -> None:
        self._set_worker_seed(self._loader_seed("train"))
        train_loader = self.load_train_data()

        self.model.to(self.device)
        self.model.train()

        if self.hf:
            self._train_hf(train_loader)
        else:
            self._train_fo(train_loader)

        self.scheduler.step()
        if self.efficiency != "high":
            self.model.to("cpu")

    def _train_fo(self, train_loader):
        """First-order MAML: 2 batches per meta-step."""
        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)

                half = batch_x.size(0) // 2
                if half == 0:
                    half = batch_x.size(0)

                # Save current params
                temp_params = [p.data.clone() for p in self.model.parameters()]

                # Inner step α on first half
                self.optimizer.zero_grad()
                out = self.model(
                    batch_x[:half], x_mark=x_mark[:half], y_mark=y_mark[:half]
                )
                loss = self.loss(out, batch_y[:half])
                loss.backward()
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.data.sub_(self.learning_rate * p.grad)

                # Meta gradient from second half at updated params
                self.optimizer.zero_grad()
                x_sec = batch_x[half:] if half < batch_x.size(0) else batch_x
                y_sec = batch_y[half:] if half < batch_x.size(0) else batch_y
                xm_sec = x_mark[half:] if half < batch_x.size(0) else x_mark
                ym_sec = y_mark[half:] if half < batch_x.size(0) else y_mark
                out2 = self.model(x_sec, x_mark=xm_sec, y_mark=ym_sec)
                loss2 = self.loss(out2, y_sec)
                loss2.backward()

                # Restore params, apply outer step β
                with torch.no_grad():
                    for p, tp in zip(self.model.parameters(), temp_params):
                        p.data.copy_(tp)
                        if p.grad is not None:
                            p.data.sub_(self.beta * p.grad)

    def _train_hf(self, train_loader):
        """Hessian-Free MAML: 3 batches per meta-step."""
        self._model_plus.to(self.device)
        self._model_minus.to(self.device)
        self._model_plus.train()
        self._model_minus.train()

        num_batches = len(train_loader)
        steps_per_epoch = max(1, num_batches // 3)

        for _ in range(self.epochs):
            iterator = iter(train_loader)
            for _ in range(steps_per_epoch):
                # Batch 0: inner step
                bx0, by0, bxm0, bym0 = self._next_batch(iterator, train_loader)
                bx0 = bx0.to(device=self.device, dtype=torch.float32)
                by0 = by0.to(device=self.device, dtype=torch.float32)
                bxm0 = bxm0.to(device=self.device, dtype=torch.float32)
                bym0 = bym0.to(device=self.device, dtype=torch.float32)

                frz_params = [p.data.clone() for p in self.model.parameters()]

                self.optimizer.zero_grad()
                out0 = self.model(bx0, x_mark=bxm0, y_mark=bym0)
                self.loss(out0, by0).backward()
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.data.sub_(self.learning_rate * p.grad)

                # Batch 1: meta gradient g
                bx1, by1, bxm1, bym1 = self._next_batch(iterator, train_loader)
                bx1 = bx1.to(device=self.device, dtype=torch.float32)
                by1 = by1.to(device=self.device, dtype=torch.float32)
                bxm1 = bxm1.to(device=self.device, dtype=torch.float32)
                bym1 = bym1.to(device=self.device, dtype=torch.float32)

                self.optimizer.zero_grad()
                out1 = self.model(bx1, x_mark=bxm1, y_mark=bym1)
                self.loss(out1, by1).backward()
                meta_grads = [
                    p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                    for p in self.model.parameters()
                ]

                # Batch 2: Hessian approximation
                bx2, by2, bxm2, bym2 = self._next_batch(iterator, train_loader)
                bx2 = bx2.to(device=self.device, dtype=torch.float32)
                by2 = by2.to(device=self.device, dtype=torch.float32)
                bxm2 = bxm2.to(device=self.device, dtype=torch.float32)
                bym2 = bym2.to(device=self.device, dtype=torch.float32)

                # model_plus = frz + δ·g, model_minus = frz - δ·g
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

                # g_hf = g - (α/2δ)·(grad_plus - grad_minus)
                hf_coef = self.learning_rate / (2.0 * self.delta)
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

                # Restore frz_params, apply outer step β
                with torch.no_grad():
                    for p, fp, g_hf in zip(
                        self.model.parameters(), frz_params, hf_grads
                    ):
                        p.data.copy_(fp - self.beta * g_hf)

        self._model_plus.to("cpu")
        self._model_minus.to("cpu")
