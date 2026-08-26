import copy
from typing import Any

import numpy as np
import torch

from .pFL import pFL, pFL_Client


class mFL(pFL):
    """MAML-style personalized FL with a uniformly averaged initialization."""

    def aggregate_client_updates(self, packages: Any) -> None:
        self._commit_global(
            new_params=self.mean_models(
                models=[
                    package["regular_model_params"] for package in packages.values()
                ]
            )
        )


class mFL_Client(pFL_Client):
    """Shared FO/Hessian-free MAML worker and one-step personalization."""

    hf: bool = False

    def __init__(self, configs: Any, times: Any, device: Any) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self.validate_meta_hyperparameters(
            inner_rate=self._inner_learning_rate(),
            outer_rate=self._outer_learning_rate(),
            delta=self.delta,
            hf=self.hf,
        )
        if self.hf:
            self._model_plus = copy.deepcopy(self.model)
            self._model_minus = copy.deepcopy(self.model)

    def _inner_learning_rate(self) -> float:
        return self.learning_rate

    def _outer_learning_rate(self) -> float:
        return self.beta

    @staticmethod
    def validate_meta_hyperparameters(
        inner_rate: float,
        outer_rate: float,
        delta: float,
        hf: bool,
    ) -> None:
        if inner_rate <= 0 or outer_rate <= 0 or hf and delta <= 0:
            raise ValueError("meta-FL requires positive inner/outer rates and HF delta")

    @staticmethod
    def _next_batch(iterator: Any, loader: Any) -> Any:
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    @staticmethod
    def _move_batch(batch: Any, device: Any) -> Any:
        return tuple(value.to(device=device, dtype=torch.float32) for value in batch)

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        train_loader = self.load_train_data()
        self.model.to(self.device).train()

        if self.hf:
            self._train_hf(train_loader=train_loader)
        else:
            self._train_fo(train_loader=train_loader)

        if self.efficiency != "high":
            self.model.to("cpu")

    def _train_fo(self, train_loader: Any) -> None:
        iterator = iter(train_loader)
        for _ in range(self.epochs):
            first, iterator = self._next_batch(iterator=iterator, loader=train_loader)
            batch_x, batch_y, x_mark, y_mark = self._move_batch(
                batch=first, device=self.device
            )
            frozen = [
                parameter.detach().clone() for parameter in self.model.parameters()
            ]

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
            self.loss(output, batch_y).backward()
            with torch.no_grad():
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.add_(
                            parameter.grad, alpha=-self._inner_learning_rate()
                        )

            second, iterator = self._next_batch(iterator=iterator, loader=train_loader)
            batch_x, batch_y, x_mark, y_mark = self._move_batch(
                batch=second, device=self.device
            )
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
            self.loss(output, batch_y).backward()
            meta_grads = [
                (
                    parameter.grad.detach().clone()
                    if parameter.grad is not None
                    else torch.zeros_like(parameter)
                )
                for parameter in self.model.parameters()
            ]

            with torch.no_grad():
                for parameter, initial, gradient in zip(
                    self.model.parameters(), frozen, meta_grads
                ):
                    parameter.copy_(initial).add_(
                        gradient, alpha=-self._outer_learning_rate()
                    )

    def _train_hf(self, train_loader: Any) -> None:
        self._model_plus.to(self.device).train()
        self._model_minus.to(self.device).train()
        iterator = iter(train_loader)

        for _ in range(self.epochs):
            frozen_state = copy.deepcopy(self.model.state_dict())

            first, iterator = self._next_batch(iterator=iterator, loader=train_loader)
            batch_x, batch_y, x_mark, y_mark = self._move_batch(
                batch=first, device=self.device
            )
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
            self.loss(output, batch_y).backward()
            with torch.no_grad():
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.add_(
                            parameter.grad, alpha=-self._inner_learning_rate()
                        )

            second, iterator = self._next_batch(iterator=iterator, loader=train_loader)
            batch_x, batch_y, x_mark, y_mark = self._move_batch(
                batch=second, device=self.device
            )
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
            self.loss(output, batch_y).backward()
            meta_grads = [
                (
                    parameter.grad.detach().clone()
                    if parameter.grad is not None
                    else torch.zeros_like(parameter)
                )
                for parameter in self.model.parameters()
            ]

            third, iterator = self._next_batch(iterator=iterator, loader=train_loader)
            batch_x, batch_y, x_mark, y_mark = self._move_batch(
                batch=third, device=self.device
            )
            self._model_plus.load_state_dict(frozen_state)
            self._model_minus.load_state_dict(frozen_state)
            with torch.no_grad():
                for plus, minus, gradient in zip(
                    self._model_plus.parameters(),
                    self._model_minus.parameters(),
                    meta_grads,
                ):
                    plus.add_(gradient, alpha=self.delta)
                    minus.add_(gradient, alpha=-self.delta)

            self._model_plus.zero_grad(set_to_none=True)
            self._model_minus.zero_grad(set_to_none=True)
            self.loss(
                self._model_plus(batch_x, x_mark=x_mark, y_mark=y_mark), batch_y
            ).backward()
            self.loss(
                self._model_minus(batch_x, x_mark=x_mark, y_mark=y_mark), batch_y
            ).backward()
            coefficient = self._inner_learning_rate() / (2.0 * self.delta)
            hf_grads = [
                gradient
                - coefficient
                * (
                    (plus.grad if plus.grad is not None else torch.zeros_like(gradient))
                    - (
                        minus.grad
                        if minus.grad is not None
                        else torch.zeros_like(gradient)
                    )
                )
                for gradient, plus, minus in zip(
                    meta_grads,
                    self._model_plus.parameters(),
                    self._model_minus.parameters(),
                )
            ]

            self.model.load_state_dict(frozen_state)
            with torch.no_grad():
                for parameter, gradient in zip(self.model.parameters(), hf_grads):
                    parameter.add_(gradient, alpha=-self._outer_learning_rate())

        self._model_plus.to("cpu")
        self._model_minus.to("cpu")

    def evaluate_personalized(
        self,
        client_id: Any,
        global_params: Any,
        personal_params: Any,
        dataset_type: Any,
        current_iter: Any,
    ) -> float:
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id=client_id)
        adapted = copy.deepcopy(self.model)
        state = self.personalized_model_state(
            base_state=global_params,
            personal_params=personal_params,
            parameter_names=[name for name, _ in adapted.named_parameters()],
        )
        adapted.load_state_dict(state or global_params, strict=False)

        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        batch = next(iter(self.load_train_data()))
        batch_x, batch_y, x_mark, y_mark = self._move_batch(
            batch=batch, device=self.device
        )
        adapted.to(self.device).train()
        adapted.zero_grad(set_to_none=True)
        output = adapted(batch_x, x_mark=x_mark, y_mark=y_mark)
        self.loss(output, batch_y).backward()
        with torch.no_grad():
            for parameter in adapted.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-self._inner_learning_rate())

        loader = (
            self.load_test_data() if dataset_type == "test" else self.load_train_data()
        )
        return float(
            np.mean(
                self.calculate_loss(
                    model=adapted,
                    dataloader=loader,
                    criterion=self.loss,
                    device=self.device,
                    offload_after=True,
                )
            )
        )
