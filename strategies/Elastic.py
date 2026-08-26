from collections import OrderedDict
from typing import Any

import torch

from .tFL import tFL, tFL_Client


class Elastic(tFL):
    """Elastic aggregation (Chen et al., CVPR 2023)."""

    optional = {
        "tau": 0.5,
        "sample_ratio": 0.3,
        "mu": 0.95,
        "server_learning_rate": 1.0,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--tau", type=float, default=None)
        parser.add_argument("--sample_ratio", type=float, default=None)
        parser.add_argument("--mu", type=float, default=None)
        parser.add_argument("--server_learning_rate", type=float, default=None)

    def aggregate_client_updates(self, packages: Any) -> None:
        client_models = []
        client_sensitivities = []
        client_scores = []
        for package in packages.values():
            client_models.append(package["regular_model_params"])
            client_sensitivities.append(package["sensitivity"])
            client_scores.append(package["score"])
        scores = torch.tensor(
            client_scores,
            dtype=torch.float32,
        )
        weights = scores / scores.sum()
        new_global = OrderedDict()

        for name, server_parameter in self.public_model_params.items():
            sensitivities = torch.stack(
                [sensitivity[name].float() for sensitivity in client_sensitivities],
                dim=-1,
            )
            aggregated_sensitivity = torch.sum(
                sensitivities * weights.to(sensitivities.dtype), dim=-1
            )
            maximum = aggregated_sensitivity.max()
            if maximum > 0:
                zeta = 1 + self.tau - aggregated_sensitivity / maximum
            else:
                zeta = torch.ones_like(aggregated_sensitivity)

            client_parameters = torch.stack(
                [model[name].float() for model in client_models],
                dim=-1,
            )
            averaged_parameter = torch.sum(
                client_parameters * weights.to(client_parameters.dtype), dim=-1
            )
            updated = server_parameter.float() + self.server_learning_rate * zeta * (
                averaged_parameter - server_parameter.float()
            )
            new_global[name] = updated.to(server_parameter.dtype)

        self._commit_global(new_params=new_global)


class Elastic_Client(tFL_Client):
    def fit(self) -> None:
        if not 0 <= self.sample_ratio < 1:
            raise ValueError("Elastic sample_ratio must be in [0, 1)")

        seed = self._loader_seed(dataset_type="train")
        full_loader = self.load_data(
            file=self.train_file,
            sample_ratio=1.0,
            shuffle=False,
            scaler=self.scaler,
            batch_size=self.batch_size,
            seed=seed,
        )
        sample_count = len(full_loader.dataset)
        if sample_count < 2:
            raise ValueError("Elastic requires at least two local training samples")

        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(sample_count, generator=generator)
        sensitivity_count = (
            max(1, int(sample_count * self.sample_ratio))
            if self.sample_ratio > 0
            else 0
        )
        sensitivity_indices = indices[:sensitivity_count].tolist()
        training_indices = indices[sensitivity_count:].tolist()

        sensitivity_loader = self.load_data(
            file=self.train_file,
            sample_ratio=1.0,
            shuffle=False,
            scaler=self.scaler,
            batch_size=self.batch_size,
            seed=seed,
            indices=sensitivity_indices,
        )
        training_loader = self.load_data(
            file=self.train_file,
            sample_ratio=1.0,
            shuffle=True,
            scaler=self.scaler,
            batch_size=self.batch_size,
            seed=seed,
            indices=training_indices,
        )
        self.train_samples = len(training_loader.dataset)
        self.initialize_scheduler(steps_per_epoch=len(training_loader))

        self.model.to(self.device)
        self.model.eval()
        sensitivity = OrderedDict(
            (name, torch.zeros_like(parameter, device=self.device))
            for name, parameter in self.model.named_parameters()
        )
        for batch_x, _, x_mark, y_mark in sensitivity_loader:
            self.model.zero_grad(set_to_none=True)
            batch_x = batch_x.to(self.device, dtype=torch.float32)
            x_mark = x_mark.to(self.device, dtype=torch.float32)
            y_mark = y_mark.to(self.device, dtype=torch.float32)
            outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
            outputs.square().sum().backward()
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    sensitivity[name].mul_(self.mu).add_(
                        parameter.grad.detach().abs(), alpha=1 - self.mu
                    )
        self._sensitivity = OrderedDict(
            (name, value.cpu()) for name, value in sensitivity.items()
        )

        offload_after_epoch = self.efficiency == "low"
        for _ in range(self.epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=training_loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
                offload_after=offload_after_epoch,
            )
        if self.efficiency == "med":
            self.model.to("cpu")

    def package(self) -> dict:
        package = super().package()
        package["sensitivity"] = self._sensitivity
        package["__wire__"] = (*package["__wire__"], "sensitivity")
        return package
