"""PFMCP: FedAvg, local mixture-of-experts personalization, and conformal PI."""

import copy
import math
import time
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any, Dict, List

import numpy as np
import ray
import torch

from models.PFMCP import PFMCP as PFMCPModel

from .base import SharedMethods
from .pFL import pFL, pFL_Client


class PFMCPShared:
    """Conformal math shared by the PFMCP server and worker."""

    @staticmethod
    def conformal_quantile(scores: torch.Tensor, alpha: float) -> torch.Tensor:
        """Return the corrected conformal quantile."""
        if scores.ndim < 1 or scores.shape[0] == 0:
            raise ValueError("PFMCP requires at least one calibration score")
        if not 0.0 < alpha < 1.0:
            raise ValueError("pfmcp_alpha must be between 0 and 1")
        # Paper Eq. 10: finite-sample corrected quantile rank.
        rank = min(
            math.ceil((scores.shape[0] + 1) * (1.0 - alpha)),
            scores.shape[0],
        )
        return torch.sort(input=scores, dim=0).values[rank - 1]

    @staticmethod
    def dynamic_conformal_intervals(
        calibration_prediction: torch.Tensor,
        calibration_target: torch.Tensor,
        test_prediction: torch.Tensor,
        test_target: torch.Tensor,
        alpha: float,
        delay: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield ordered conformal intervals."""
        if delay < 1:
            raise ValueError("PFMCP conformal delay must be positive")
        if test_prediction.shape != test_target.shape:
            raise ValueError("PFMCP test prediction and target shapes must match")
        if calibration_prediction.shape != calibration_target.shape:
            raise ValueError(
                "PFMCP calibration prediction and target shapes must match"
            )

        scores = torch.abs(calibration_target - calibration_prediction)
        test_scores = torch.abs(test_target - test_prediction)
        # Paper Algorithm 1: reveal delayed residuals in temporal order.
        for test_index in range(test_prediction.shape[0]):
            quantile = PFMCPShared.conformal_quantile(
                scores=scores,
                alpha=alpha,
            )
            prediction = test_prediction[test_index]
            yield (
                prediction,
                prediction - quantile,
                prediction + quantile,
                test_target[test_index],
            )
            if test_index >= delay:
                scores = torch.cat(
                    tensors=(
                        scores[1:],
                        test_scores[test_index - delay].unsqueeze(dim=0),
                    ),
                    dim=0,
                )


# Compatibility names; implementations live on PFMCPShared.
conformal_quantile = PFMCPShared.conformal_quantile
dynamic_conformal_intervals = PFMCPShared.dynamic_conformal_intervals


class PFMCP(PFMCPShared, pFL):
    """Three-stage PFMCP server."""

    compulsory = {
        "model": "PFMCP",
        "optimizer": "SGD",
        "scheduler": "BaseScheduler",
        "learning_rate": 0.005,
        "loss": "MSE",
    }
    optional = {
        # The paper's 75:5:20 chronological split.  FedProC's train file is
        # train+calibration, so 0.05 / train_ratio is reserved locally.
        "pfmcp_calibration_ratio": 0.05,
        "pfmcp_alpha": 0.1,
        # Required only for the reported CWC metric; it does not affect PI.
        "pfmcp_cwc_lambda": 50.0,
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--pfmcp_calibration_ratio",
            type=float,
            default=None,
        )
        parser.add_argument("--pfmcp_alpha", type=float, default=None)
        parser.add_argument("--pfmcp_cwc_lambda", type=float, default=None)

    _client_csv_fields = (
        "uplink_mb",
        "train_loss",
        "test_loss",
        "pfmcp_picp",
        "pfmcp_nmpiw",
        "pfmcp_cwc",
    )

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        if not isinstance(self.model, PFMCPModel):
            raise TypeError("PFMCP strategy requires --model PFMCP")

        state = self.model.state_dict()
        self.public_model_params = OrderedDict(
            (name, state[name].detach().cpu().clone())
            for name in self.model.regular_parameter_names()
        )
        self.pfmcp_phase = "federated"
        self.pfmcp_personalized = False
        self.metrics.update(
            {
                "pfmcp_picp": [],
                "pfmcp_nmpiw": [],
                "pfmcp_cwc": [],
            }
        )

    def package(self, client_id: int) -> Dict[str, Any]:
        package = super().package(client_id=client_id)
        package["pfmcp_phase"] = self.pfmcp_phase
        # Personal state is never sent by the PFMCP protocol.  During the
        # one-shot personalization stage it is initialized from the received
        # final global decoder and retained locally.
        package["personal_model_params"] = {}
        package["__wire__"] = ("regular_model_params",)
        return package

    def _pre_eval_hook(self, dataset_type: str) -> None:
        if self.pfmcp_personalized:
            super()._pre_eval_hook(dataset_type=dataset_type)

    def _evaluate_conformal_clients(
        self,
        client_ids: List[int],
    ) -> List[Dict[str, float]]:
        """Run local CP instrumentation without adding protocol traffic."""
        if not self.parallel:
            return [
                self.trainer.worker.evaluate_conformal(
                    client_id,
                    self.public_model_params,
                    self.clients_personal_model_params[client_id],
                    self.current_iter,
                )
                for client_id in client_ids
            ]

        global_ref = ray.put(self.public_model_params)
        futures = [
            self.trainer.workers[
                index % self.trainer.num_workers
            ].evaluate_conformal.remote(
                client_id,
                global_ref,
                self.clients_personal_model_params[client_id],
                self.current_iter,
            )
            for index, client_id in enumerate(client_ids)
        ]
        return list(ray.get(futures))

    def _run_conformal_stage(self, client_ids: List[int]) -> None:
        self.logger.info("-------------PFMCP conformal prediction-------------")
        results = self._evaluate_conformal_clients(client_ids=client_ids)
        for metric in ("pfmcp_picp", "pfmcp_nmpiw", "pfmcp_cwc"):
            value = float(np.mean([result[metric] for result in results]))
            self.metrics[metric].append(value)
            self.logger.info(f"{metric}: {value:.6f}")
        for client_id, result in zip(client_ids, results):
            client_data = self._round_client_data.setdefault(client_id, {})
            client_data.update(result)

    def _run_personalization_stage(self) -> None:
        if self.pfmcp_personalized:
            return

        stage_start = time.time()
        self.pfmcp_phase = "personalization"
        self.current_iter = self.iterations
        incumbent = [
            client_id
            for client_id in range(self.num_clients)
            if not self.is_new[client_id]
        ]
        previous_selection = getattr(self, "selected_clients", [])
        self.selected_clients = incumbent

        self.logger.info("")
        self.logger.info("-------------PFMCP personalization-------------")
        packages = self.trainer.train(incumbent)
        uplink, downlink = self._compute_send_mb(packages=packages)
        self.metrics["downlink_mb"].append(downlink)
        for client_id, size_mb in uplink.items():
            self._round_client_data.setdefault(client_id, {})["uplink_mb"] = size_mb

        self.pfmcp_personalized = True
        for dataset_type in ("train", "test"):
            if dataset_type == "train" and self.skip_eval_train:
                continue
            self._pre_eval_hook(dataset_type=dataset_type)
        self._run_conformal_stage(client_ids=incumbent)

        elapsed = time.time() - stage_start
        self.metrics["time_per_iter"].append(elapsed)
        self.logger.info(f"PFMCP personalization and CP took {elapsed:.2f}s")
        self._flush_round()
        self.selected_clients = previous_selection

    def _save_personal_models(self, postfix: str) -> None:
        template = copy.deepcopy(self.model)
        for client_id, params in self.clients_personal_model_params.items():
            if not params:
                continue
            template.load_state_dict(self.public_model_params, strict=False)
            template.load_state_dict(params, strict=False)
            template.set_mode("personalized")
            personal_config = copy.deepcopy(self.configs)
            personal_config.__dict__["pfmcp_inference_mode"] = "personalized"
            SharedMethods.save_model(
                model=template,
                path=self.model_path,
                name=f"client_{client_id}",
                postfix=postfix,
                configs=personal_config,
                metadata={
                    "pfmcp_stage": "personalized_moe",
                    "pfmcp_local_state": True,
                },
                verbose=self.logger,
            )
        self.model.load_state_dict(self.public_model_params, strict=False)
        self.model.set_mode("global")

    def _save_last_hook(self) -> None:
        self._run_personalization_stage()
        super()._save_last_hook()


class PFMCP_Client(PFMCPShared, pFL_Client):
    """Reusable worker emulating a client with private PFMCP state."""

    def __init__(
        self,
        configs: Namespace,
        times: int,
        device: str,
    ) -> None:
        super().__init__(configs=configs, times=times, device=device)
        if not isinstance(self.model, PFMCPModel):
            raise TypeError("PFMCP client requires the PFMCP model")
        if not 0.0 < self.pfmcp_calibration_ratio < self.train_ratio:
            raise ValueError(
                "pfmcp_calibration_ratio must be positive and smaller than "
                "train_ratio"
            )
        if not 0.0 < self.pfmcp_alpha < 1.0:
            raise ValueError("pfmcp_alpha must be between 0 and 1")

        self.regular_params_name = self.model.regular_parameter_names()
        self.personal_params_name = self.model.personal_parameter_names()
        regular_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if name in self.regular_params_name
        ]
        optimizer_cls = self._build(kind="optimizers", name=configs.optimizer)
        self.optimizer = optimizer_cls(
            params=regular_parameters,
            configs=configs,
        )
        self._scheduler_base_lrs = [
            float(group["lr"]) for group in self.optimizer.param_groups
        ]
        self.initialize_scheduler()
        self.init_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.pfmcp_phase = "federated"

    def _split_indices(self) -> tuple[List[int], List[int]]:
        with np.load(self.train_file) as data:
            sample_count = len(data["x"])
        if sample_count < 2:
            raise ValueError("PFMCP needs at least two pre-test samples")

        fraction_within_train = self.pfmcp_calibration_ratio / self.train_ratio
        calibration_count = max(
            1,
            int(round(sample_count * fraction_within_train)),
        )
        calibration_count = min(calibration_count, sample_count - 1)
        split = sample_count - calibration_count
        return list(range(split)), list(range(split, sample_count))

    def load_train_data(self) -> Any:
        fit_indices, _ = self._split_indices()
        if self.sample_ratio < 1.0:
            sample_count = max(1, int(len(fit_indices) * self.sample_ratio))
            rng = np.random.default_rng(self._loader_seed(dataset_type="train"))
            fit_indices = rng.choice(
                fit_indices,
                size=sample_count,
                replace=False,
            ).tolist()
        loader = self.load_data(
            file=self.train_file,
            indices=fit_indices,
            shuffle=True,
            scaler=self.scaler,
            batch_size=self.batch_size,
            seed=self._loader_seed(dataset_type="train"),
        )
        self.train_samples = len(loader.dataset)
        return loader

    def load_calibration_data(self) -> Any:
        _, calibration_indices = self._split_indices()
        return self.load_data(
            file=self.train_file,
            indices=calibration_indices,
            shuffle=False,
            scaler=self.scaler,
            batch_size=self.batch_size,
            seed=self._loader_seed(dataset_type="valid"),
        )

    def set_parameters(self, package: Dict[str, Any]) -> None:
        self.pfmcp_phase = package["pfmcp_phase"]
        super().set_parameters(package=package)
        if self.pfmcp_phase == "personalization":
            # Algorithm 1 initializes the local decoder from the final global
            # decoder.  A per-client seed makes gate initialization independent
            # of which reusable worker happens to execute the client.
            self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
            self.model.initialize_personalization()
            self.model.gate.reset_parameters()
        self.model.set_trainable_phase(self.pfmcp_phase)

    def fit(self) -> None:
        if self.pfmcp_phase == "federated":
            super().fit()
            return
        self._fit_personalization()

    def _fit_personalization(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if name in self.personal_params_name
        ]
        optimizer = torch.optim.SGD(
            parameters,
            lr=self.learning_rate,
        )

        self.model.to(self.device)
        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in loader:
                batch_x = batch_x.to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                batch_y = batch_y.to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                x_mark = x_mark.to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                y_mark = y_mark.to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(
                    batch_x,
                    x_mark=x_mark,
                    y_mark=y_mark,
                )
                loss = self.loss(outputs, batch_y)
                loss.backward()
                optimizer.step()

        if self.efficiency != "high":
            self.model.to("cpu")

    def package(self) -> Dict[str, Any]:
        package = super().package()
        if self.pfmcp_phase == "federated":
            package["personal_model_params"] = {}
            package["__wire__"] = ("regular_model_params", "score")
        else:
            # FedProC writes this state back only to emulate client-local
            # persistence.  No personalized parameters are uploaded in PFMCP.
            package["regular_model_params"] = {}
            package["__wire__"] = ()
        return package

    def evaluate_global(
        self,
        client_id: int,
        global_params: OrderedDict,
        dataset_type: str,
        current_iter: int,
    ) -> float:
        self.model.set_mode("global")
        return super().evaluate_global(
            client_id=client_id,
            global_params=global_params,
            dataset_type=dataset_type,
            current_iter=current_iter,
        )

    def evaluate_personalized(
        self,
        client_id: int,
        global_params: OrderedDict,
        personal_params: Dict[str, torch.Tensor],
        dataset_type: str,
        current_iter: int,
    ) -> float:
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id=client_id)
        self.model.load_state_dict(global_params, strict=False)
        self.model.load_state_dict(personal_params, strict=False)
        self.model.set_mode("personalized")
        loader = (
            self.load_test_data() if dataset_type == "test" else self.load_train_data()
        )
        losses = self.calculate_loss(
            model=self.model,
            dataloader=loader,
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        return float(np.mean(losses))

    def _ordered_predictions(self, loader: Any) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        targets = []
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, x_mark, y_mark in loader:
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                x_mark = x_mark.to(self.device, dtype=torch.float32)
                y_mark = y_mark.to(self.device, dtype=torch.float32)
                prediction = self.model(
                    batch_x,
                    x_mark=x_mark,
                    y_mark=y_mark,
                )
                predictions.append(prediction.detach().cpu())
                targets.append(batch_y.detach().cpu().to(torch.float32))
        if self.efficiency != "high":
            self.model.to("cpu")

        prediction_np = self.scaler.inverse_transform(torch.cat(predictions).numpy())
        target_np = self.scaler.inverse_transform(torch.cat(targets).numpy())
        return (
            torch.as_tensor(prediction_np, dtype=torch.float32),
            torch.as_tensor(target_np, dtype=torch.float32),
        )

    def _target_range(self) -> torch.Tensor:
        minimum = None
        maximum = None
        for path in (self.train_file, self.test_file):
            with np.load(path) as data:
                targets = np.asarray(data["y"], dtype=np.float32)
                axes = tuple(range(targets.ndim - 1))
                current_min = np.nanmin(targets, axis=axes)
                current_max = np.nanmax(targets, axis=axes)
            minimum = (
                current_min if minimum is None else np.minimum(minimum, current_min)
            )
            maximum = (
                current_max if maximum is None else np.maximum(maximum, current_max)
            )
        target_range = np.maximum(maximum - minimum, np.finfo(np.float32).eps)
        return torch.as_tensor(target_range, dtype=torch.float32).view(1, -1)

    def evaluate_conformal(
        self,
        client_id: int,
        global_params: OrderedDict,
        personal_params: Dict[str, torch.Tensor],
        current_iter: int,
    ) -> Dict[str, float]:
        """Evaluate conformal coverage and width."""
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id=client_id)
        self.model.load_state_dict(global_params, strict=False)
        self.model.load_state_dict(personal_params, strict=False)
        self.model.set_mode("personalized")

        calibration_prediction, calibration_target = self._ordered_predictions(
            loader=self.load_calibration_data()
        )
        test_prediction, test_target = self._ordered_predictions(
            loader=self.load_test_data()
        )
        target_range = self._target_range()

        covered = 0.0
        element_count = 0
        normalized_width_sum = 0.0
        intervals = self.dynamic_conformal_intervals(
            calibration_prediction=calibration_prediction,
            calibration_target=calibration_target,
            test_prediction=test_prediction,
            test_target=test_target,
            alpha=self.pfmcp_alpha,
            delay=int(self.configs.output_len),
        )
        for _, lower, upper, target in intervals:
            covered += float(((target >= lower) & (target <= upper)).sum())
            element_count += target.numel()
            normalized_width_sum += float(((upper - lower) / target_range).sum())

        # Paper Eqs. 9-11: PICP, normalized MPIW, and CWC.
        picp = covered / element_count
        nmpiw = normalized_width_sum / element_count
        mu = 1.0 - self.pfmcp_alpha
        penalty = math.exp(-self.pfmcp_cwc_lambda * (picp - mu)) if picp < mu else 0.0
        cwc = nmpiw * (1.0 + penalty)
        return {
            "pfmcp_picp": float(picp),
            "pfmcp_nmpiw": float(nmpiw),
            "pfmcp_cwc": float(cwc),
        }
