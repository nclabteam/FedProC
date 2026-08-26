import copy
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import ray
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor

from schedulers.BaseScheduler import BaseScheduler

from .dFL import dFL, dFL_Client
from .tFL import Trainer


class DFedHPO_Trainer(Trainer):
    """Run the two one-off HPO phases on reusable workers."""

    def run_hpo(
        self,
        packages: Mapping[int, dict[str, Any]],
        method: str,
    ) -> "OrderedDict[int, Any]":
        if not self.parallel:
            return OrderedDict(
                (
                    client_id,
                    getattr(self.worker, method)(package=package),
                )
                for client_id, package in packages.items()
            )

        client_ids = list(packages)
        futures = [
            getattr(self.workers[index % self.num_workers], method).remote(
                package=packages[client_id]
            )
            for index, client_id in enumerate(client_ids)
        ]
        return OrderedDict(zip(client_ids, ray.get(futures)))


class DFedHPO(dFL):
    """Single-pass neighbor HPO before decentralized training."""

    optional = {
        "trials": 10,
        "eval_epochs": 3,
        "aggregator": "FA",
        "top_k": 3,
        "lr_min": 1e-5,
        "lr_max": 1e-1,
    }
    compulsory = {"scheduler": "BaseScheduler"}

    @classmethod
    def args_update(cls, parser: Any) -> None:
        super().args_update(parser=parser)
        parser.add_argument("--trials", type=int, default=None)
        parser.add_argument("--eval_epochs", type=int, default=None)
        parser.add_argument(
            "--aggregator", type=str, default=None, choices=["CA", "FA", "MA"]
        )
        parser.add_argument("--top_k", type=int, default=None)
        parser.add_argument("--lr_min", type=float, default=None)
        parser.add_argument("--lr_max", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.client_learning_rates: dict[int, float] = {}

    def _make_trainer(self) -> DFedHPO_Trainer:
        return DFedHPO_Trainer(
            server=self,
            client_cls=self._client_cls(),
            configs=self.configs,
            times=self.times,
        )

    def package(self, client_id: int) -> dict[str, Any]:
        package = super().package(client_id=client_id)
        if client_id in self.client_learning_rates:
            package["learning_rate"] = self.client_learning_rates[client_id]
        return package

    def train(self) -> None:
        self.run_hpo()
        super().train()

    def run_hpo(self) -> None:
        client_ids = [
            client_id
            for client_id in range(self.num_clients)
            if not self.is_new[client_id]
        ]
        self.logger.info("--- DFed-HPO: starting HPO phase ---")
        local = self.trainer.run_hpo(
            packages=OrderedDict(
                (client_id, self.package(client_id=client_id))
                for client_id in client_ids
            ),
            method="run_local_hpo",
        )
        aggregate_packages = OrderedDict()
        for client_id in client_ids:
            peers = [
                client_id,
                *(peer for peer in self.topology[client_id] if peer in local),
            ]
            package = self.package(client_id=client_id)
            package["hpo_candidates"] = [
                candidate for peer in peers for candidate in local[peer]
            ]
            aggregate_packages[client_id] = package

        optimal = self.trainer.run_hpo(
            packages=aggregate_packages,
            method="aggregate_hpo",
        )
        self.client_learning_rates = {
            client_id: float(learning_rate)
            for client_id, learning_rate in optimal.items()
        }
        for client_id, learning_rate in self.client_learning_rates.items():
            self.logger.info(f"Client {client_id}: lr={learning_rate:.6f}")
        self.logger.info("--- DFed-HPO: HPO phase complete ---")


class DFedHPO_Client(dFL_Client):
    """Stateless local search and neighbor aggregation worker."""

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package=package)
        if "learning_rate" in package:
            for group in self.optimizer.param_groups:
                group["lr"] = package["learning_rate"]

    def run_local_hpo(self, package: Dict[str, Any]) -> list[dict[str, Any]]:
        self.set_parameters(package=package)
        if self.trials < 1 or self.eval_epochs < 1 or self.top_k < 1:
            raise ValueError("trials, eval_epochs, and top_k must be positive")
        if not 0 < self.lr_min < self.lr_max:
            raise ValueError("learning-rate bounds must satisfy 0 < min < max")

        rng = np.random.default_rng(self._loader_seed(dataset_type="train"))
        initial_state = copy.deepcopy(self.model.state_dict())
        candidates = []
        for _ in range(self.trials):
            config = self._sample_config(rng=rng)
            loss = self._evaluate_config(
                config=config,
                initial_state=initial_state,
            )
            candidate = {"config": config, "loss": loss}
            if self.aggregator == "CA":
                candidate["model_vector"] = (
                    torch.nn.utils.parameters_to_vector(self.model.parameters())
                    .detach()
                    .cpu()
                )
            candidates.append(candidate)

        self.model.load_state_dict(state_dict=initial_state)
        self.model.to(device="cpu")
        return candidates

    def _sample_config(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "lr": float(np.exp(rng.uniform(np.log(self.lr_min), np.log(self.lr_max))))
        }

    def _evaluate_config(
        self,
        config: Mapping[str, float],
        initial_state: Mapping[str, torch.Tensor],
    ) -> float:
        self.model.load_state_dict(state_dict=initial_state)
        trial_configs = copy.copy(self.configs)
        trial_configs.learning_rate = config["lr"]
        optimizer_cls = self._get_objective_function(
            func_type="optimizers", func_name=trial_configs.optimizer
        )
        optimizer = optimizer_cls(params=self.model.parameters(), configs=trial_configs)
        scheduler = BaseScheduler(optimizer=optimizer, configs=trial_configs)
        train_loader = self.load_train_data()
        for _ in range(self.eval_epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=self.loss,
                scheduler=scheduler,
                device=self.device,
                offload_after=False,
            )
        losses = self.calculate_loss(
            model=self.model,
            dataloader=self.load_test_data(),
            criterion=self.loss,
            device=self.device,
            offload_after=False,
        )
        return float(np.mean(losses))

    def aggregate_hpo(self, package: Dict[str, Any]) -> float:
        self.set_parameters(package=package)
        candidates = package["hpo_candidates"]
        initial_state = copy.deepcopy(self.model.state_dict())
        if self.aggregator == "CA":
            config = self._consensus_aggregator(candidates=candidates)
        elif self.aggregator == "FA":
            config = self._fusion_aggregator(
                candidates=candidates,
                initial_state=initial_state,
            )
        elif self.aggregator == "MA":
            config = self._metaregress_aggregator(
                candidates=candidates,
                initial_state=initial_state,
            )
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
        self.model.load_state_dict(state_dict=initial_state)
        self.model.to(device="cpu")
        return float(config["lr"])

    @staticmethod
    def _consensus_aggregator(
        candidates: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, float]:
        if not candidates:
            raise ValueError("at least one HPO candidate is required")
        vectors = [candidate.get("model_vector") for candidate in candidates]
        if any(vector is None for vector in vectors):
            return min(candidates, key=lambda candidate: candidate["loss"])["config"]
        normalized = F.normalize(torch.stack(vectors), dim=1)
        # MBA angular consensus: a_i = mean_j cos(theta_i, theta_j).
        agreement = torch.matmul(normalized, normalized.T).mean(dim=1)
        best = int(torch.argmax(agreement))
        return candidates[best]["config"]

    def _fusion_aggregator(
        self,
        candidates: Sequence[Mapping[str, Any]],
        initial_state: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        revalidated = [
            (
                candidate["config"],
                self._evaluate_config(
                    config=candidate["config"],
                    initial_state=initial_state,
                ),
            )
            for candidate in candidates
        ]
        top = sorted(revalidated, key=lambda item: item[1])[: self.top_k]
        return {"lr": float(np.mean([config["lr"] for config, _ in top]))}

    def _metaregress_aggregator(
        self,
        candidates: Sequence[Mapping[str, Any]],
        initial_state: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        if len(candidates) < 3:
            return self._fusion_aggregator(
                candidates=candidates,
                initial_state=initial_state,
            )

        # Paper MA: RF maps observed hyperparameters to local loss.
        features = np.array(
            [[np.log(candidate["config"]["lr"])] for candidate in candidates]
        )
        targets = np.array([candidate["loss"] for candidate in candidates])
        regressor = RandomForestRegressor(n_estimators=50, random_state=0)
        regressor.fit(features, targets)
        grid = np.linspace(np.log(self.lr_min), np.log(self.lr_max), 200)
        predicted = regressor.predict(grid.reshape(-1, 1))
        proposed = np.exp(grid[np.argsort(predicted)[: self.top_k]])

        evaluated = [
            (
                float(learning_rate),
                self._evaluate_config(
                    config={"lr": float(learning_rate)},
                    initial_state=initial_state,
                ),
            )
            for learning_rate in proposed
        ]
        return {"lr": min(evaluated, key=lambda item: item[1])[0]}
