from typing import Any, Callable, Dict, Iterable

import torch

from .dFL import dFL, dFL_Client
from .tFL import tFL, tFL_Client


class FedProx(tFL):
    """FedAvg with the paper's proximal local objective."""

    optional = {"mu": 0.01}

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--mu", type=float, default=None)


class FedProx_Client(tFL_Client):
    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package=package)
        self._global_params = [
            parameter.detach().cpu().clone() for parameter in self.model.parameters()
        ]

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        dataloader: Iterable,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: Any,
        device: str | torch.device,
        offload_after: bool = True,
    ) -> None:
        if self.mu < 0:
            raise ValueError("mu must be non-negative")
        model.to(device=device)
        self._move_optimizer_state_to_param_devices(optimizer=optimizer)
        global_params = [
            parameter.to(device=device) for parameter in self._global_params
        ]
        model.train()
        for batch_x, batch_y, x_mark, y_mark in dataloader:
            optimizer.zero_grad(set_to_none=True)
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.float32)
            x_mark = x_mark.to(device=device)
            y_mark = y_mark.to(device=device)
            criterion(model(batch_x, x_mark=x_mark, y_mark=y_mark), batch_y).backward()

            pairs = [
                (parameter, anchor)
                for parameter, anchor in zip(model.parameters(), global_params)
                if parameter.grad is not None
            ]
            if pairs:
                # grad F_k(w) + mu * (w - w_t), equivalent to Eq. 2.
                torch._foreach_add_(
                    [parameter.grad for parameter, _ in pairs],
                    torch._foreach_sub(
                        [parameter.detach() for parameter, _ in pairs],
                        [anchor for _, anchor in pairs],
                    ),
                    alpha=self.mu,
                )
            optimizer.step()
            self.step_scheduler_batch(
                scheduler=scheduler,
                batch_data=batch_x,
            )

        self.step_scheduler_epoch(scheduler=scheduler)
        if offload_after:
            model.to(device="cpu")


class DFedProx(dFL):
    """Compose decentralized neighbor mixing with FedProx local training."""

    optional = {"mu": 0.01}

    @classmethod
    def args_update(cls, parser: Any) -> None:
        super().args_update(parser=parser)
        parser.add_argument("--mu", type=float, default=None)


class DFedProx_Client(FedProx_Client, dFL_Client):
    """Stateless decentralized FedProx worker."""
