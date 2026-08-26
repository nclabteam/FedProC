from typing import Any, Callable, Iterable

import torch

from .dFL import dFL, dFL_Client


class DFedSAM(dFL):
    """DFedSAM with optional multiple gossip steps (MGS)."""

    optional = {"use_mgs": False, "mgs_steps": 4, "rho": 0.01}

    @classmethod
    def args_update(cls, parser: Any) -> None:
        super().args_update(parser=parser)
        parser.add_argument("--rho", type=float, default=None)
        parser.add_argument(
            "--use_mgs", type=lambda value: value.lower() != "false", default=None
        )
        parser.add_argument("--mgs_steps", type=int, default=None)

    def _num_gossip_steps(self) -> int:
        if not self.use_mgs:
            return 1
        if self.mgs_steps < 1:
            raise ValueError("mgs_steps must be positive")
        return self.mgs_steps


class DFedSAM_Client(dFL_Client):
    """Local sharpness-aware training from paper Algorithm 1."""

    @staticmethod
    def _grad_norm(model: torch.nn.Module) -> torch.Tensor:
        gradients = [
            parameter.grad.detach()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            return torch.zeros((), device=next(model.parameters()).device)
        return torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(gradient) for gradient in gradients])
        )

    @staticmethod
    def _add_perturbation(
        model: torch.nn.Module,
        rho: float,
        grad_norm: torch.Tensor,
    ) -> tuple[list[torch.nn.Parameter], list[torch.Tensor]]:
        parameters = [
            parameter for parameter in model.parameters() if parameter.grad is not None
        ]
        scale = rho / grad_norm.clamp_min(torch.finfo(grad_norm.dtype).eps)
        perturbations = torch._foreach_mul(
            [parameter.grad.detach() for parameter in parameters], scale
        )
        with torch.no_grad():
            torch._foreach_add_(parameters, perturbations)
        return parameters, perturbations

    @staticmethod
    def _remove_perturbation(
        parameters: list[torch.nn.Parameter],
        perturbations: list[torch.Tensor],
    ) -> None:
        with torch.no_grad():
            torch._foreach_sub_(parameters, perturbations)

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
        if self.rho < 0:
            raise ValueError("rho must be non-negative")
        model.to(device=device)
        self._move_optimizer_state_to_param_devices(optimizer=optimizer)
        model.train()
        for batch_x, batch_y, x_mark, y_mark in dataloader:
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.float32)
            x_mark = x_mark.to(device=device)
            y_mark = y_mark.to(device=device)

            optimizer.zero_grad(set_to_none=True)
            criterion(model(batch_x, x_mark=x_mark, y_mark=y_mark), batch_y).backward()
            grad_norm = self._grad_norm(model=model)
            # Algorithm 1: delta = rho * g / ||g||_2.
            parameters, perturbations = self._add_perturbation(
                model=model,
                rho=self.rho,
                grad_norm=grad_norm,
            )
            optimizer.zero_grad(set_to_none=True)
            try:
                criterion(
                    model(batch_x, x_mark=x_mark, y_mark=y_mark), batch_y
                ).backward()
            finally:
                self._remove_perturbation(
                    parameters=parameters,
                    perturbations=perturbations,
                )
            optimizer.step()
            self.step_scheduler_batch(
                scheduler=scheduler,
                batch_data=batch_x,
            )

        self.step_scheduler_epoch(scheduler=scheduler)
        if offload_after:
            model.to(device="cpu")
