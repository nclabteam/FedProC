import copy

import numpy as np
import torch

from .tFL import tFL, tFL_Client


class FedA2L(tFL):
    optional = {
        "swin": 10,
        "time_tunning": 1,
        "metric_tunning": "[wds,crs]",
        "start_tunning": 20,
        "kt": 0.3,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "--swin", type=int, default=None, help="Sliding window size"
        )
        parser.add_argument("--start_tunning", type=int, default=None)
        parser.add_argument("--time_tunning", type=int, default=None)
        parser.add_argument("--metric_tunning", type=str, default=None)
        parser.add_argument("--kt", type=float, default=None)


class FedA2L_Client(tFL_Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calculate_metric = ModelMetrics(self.metrics)
        self.count = 0
        self.metric_t = {}
        self.metric_a = {}
        self.metric_tunning = self.metric_tunning.strip("[]").split(",")
        self.ratio_tunning = [0.6, 0.4]
        self.iterations = 0

    def initialize_optimizer(self):
        self.layer_names = set()
        for name, param in self.model.named_parameters():
            layer_name = ".".join(name.split(".")[:-1])
            self.layer_names.add(layer_name)

        param_groups = []
        assigned_params = set()
        for name, module in self.model.named_modules():
            if name:
                params = list(module.parameters())
                if params:
                    unique_params = [p for p in params if p not in assigned_params]
                    if unique_params:
                        param_groups.append(
                            {
                                "params": unique_params,
                                "lr": self.learning_rate,
                                "name": name,
                            }
                        )
                        assigned_params.update(unique_params)

        self.optimizer = getattr(__import__("optimizers"), self.optimizer)(
            params=param_groups, configs=self.configs
        )

    def train(self):
        self.betrain_model = copy.deepcopy(self.model)
        result = super().train()
        self.metric_t = self.calculate_metric.calculate_metric(
            method_type=self.metric_tunning[0],
            model_before=self.betrain_model,
            model_after=self.model,
        )
        return result

    def receive_from_server(self, data):
        self.iterations += 1
        beagg_model = copy.deepcopy(self.model)
        super().receive_from_server(data=data)
        if self.iterations == 1:
            return

        self.metric_a = self.calculate_metric.calculate_metric(
            method_type=self.metric_tunning[1],
            model_before=beagg_model,
            model_after=self.model,
        )

        if (
            self.iterations > self.start_tunning
            and self.iterations % self.time_tunning == 0
        ):
            self.learning_rate_tunning()
            self.count += 1
        else:
            for group in self.optimizer.param_groups:
                self.metrics.setdefault(f"{group['name']}_lr", []).append(group["lr"])

        self.calculate_metric.update_metrics(
            metric_tunning=self.metric_tunning,
            metricts_t=self.metric_t,
            metricts_a=self.metric_a,
        )

    def learning_rate_tunning(self):
        layerwise_lr = {}
        epsilon = 1e-6

        for layer, group in zip(
            self.metric_t.keys(), self.optimizer.param_groups
        ):  # Iterate over all layers
            # Get the last `history_size` values for each metric
            metrics_keys = [
                f"{layer}_{metric}_{state}"
                for metric, state in zip(self.metric_tunning, ["t", "a"])
            ]
            metrics_swin = [
                (
                    self.metrics[key][-self.swin :]
                    if len(self.metrics[key]) >= self.swin
                    else self.metrics[key]
                )
                for key in metrics_keys
            ]
            metrics_tensor = torch.tensor(metrics_swin, dtype=torch.float32)

            metric_mean = torch.mean(metrics_tensor, dim=1)
            metric_std = torch.std(metrics_tensor, dim=1) + epsilon

            metric_score = (metrics_tensor[:, -1] - metric_mean) / metric_std
            lambda_l = torch.dot(
                torch.tensor(self.ratio_tunning, dtype=torch.float32),
                torch.exp(metric_score),
            )

            layer_lr = (
                self.learning_rate
                * (1 + torch.tanh(torch.log(lambda_l)))
                / torch.sqrt(
                    torch.tensor(1 + self.kt * self.count, dtype=torch.float32)
                )
            )

            layerwise_lr[layer] = group["lr"] = layer_lr.item()
            self.metrics.setdefault(f"{group['name']}_lr", []).append(layer_lr.item())


class ModelMetrics:
    def __init__(self, metrics):
        self.metrics = metrics
        self.methods = ["wds", "crs", "css", "wvs"]

    def softsign(self, x):
        return x / (1 + np.abs(x))

    def calculate_divergence(self, model_before, model_after):
        divergence_dict = {}
        for (name, param_before), (_, param_after) in zip(
            model_before.named_parameters(), model_after.named_parameters()
        ):
            param_before = param_before.to(param_after.device)
            param_diff = param_after - param_before
            divergence = (
                (torch.norm(param_diff) / torch.norm(param_before + 1e-8))
                .cpu()
                .detach()
                .numpy()
            )
            divergence_dict[name] = self.softsign(divergence.item())
        return self.remove_bias(divergence_dict)

    def calculate_consensus_ratio(self, model_before, model_after, threshold=1e-3):
        consensus_dict = {}
        for (name, param_before), (_, param_after) in zip(
            model_before.named_parameters(), model_after.named_parameters()
        ):
            param_before = param_before.to(param_after.device)
            param_diff = param_after - param_before
            consensus = (
                torch.sum(torch.abs(param_diff) < threshold).item() / param_diff.numel()
            )
            consensus_dict[name] = self.softsign(consensus)
        return self.remove_bias(consensus_dict)

    def calculate_convergence_speed(self, model_before, model_after):
        convergence_dict = {}
        for (name, param_before), (_, param_after) in zip(
            model_before.named_parameters(), model_after.named_parameters()
        ):
            param_before = param_before.to(param_after.device)
            param_diff = param_after - param_before
            convergence_speed = torch.mean(torch.abs(param_diff)).item()
            convergence_dict[name] = self.softsign(convergence_speed)
        return self.remove_bias(convergence_dict)

    def calculate_weight_variance(self, model_before, model_after):
        variance_dict = {}
        for (name, param_before), (_, param_after) in zip(
            model_before.named_parameters(), model_after.named_parameters()
        ):
            param_before = param_before.to(param_after.device)
            param_diff = param_after - param_before
            weight_variance = (
                (torch.norm(param_diff) / len(param_after)).cpu().detach().numpy()
            )
            variance_dict[name] = self.softsign(weight_variance.item())
        return self.remove_bias(variance_dict)

    def calculate_all_metrics(self, model_before, model_after):
        """
        Compute all metrics (Weight Divergence, Consensus Ratio, Convergence Speed, Weight Variance)
        """
        all_metrics = {
            "wds": self.calculate_divergence(model_before, model_after),
            "crs": self.calculate_consensus_ratio(model_before, model_after),
            "css": self.calculate_convergence_speed(model_before, model_after),
            "wvs": self.calculate_weight_variance(model_before, model_after),
        }
        self.save_all_metrics(all_metrics)

    def update_metrics(self, metric_tunning, metricts_t, metricts_a):
        for (layer, metrict_t), metrict_a in zip(
            metricts_t.items(), metricts_a.values()
        ):
            self.metrics.setdefault(f"{layer}_{metric_tunning[0]}_t", []).append(
                metrict_t
            )
            self.metrics.setdefault(f"{layer}_{metric_tunning[1]}_a", []).append(
                metrict_a
            )
            # self.metrics.setdefault(f"{layer}_{metric_tunning[2]}_r", []).append(metrict_r)

    def calculate_metric(self, method_type, model_before, model_after):
        method_map = {
            "css": self.calculate_convergence_speed,
            "wds": self.calculate_divergence,
            "crs": self.calculate_consensus_ratio,
            "wvs": self.calculate_weight_variance,
        }

        if method_type not in method_map:
            raise ValueError(
                f"Invalid method type: {method_type}. Choose from {list(method_map.keys())}"
            )

        return method_map[method_type](model_before, model_after)

    def save_all_metrics(self, metrics_dict, name="train"):
        for layer in metrics_dict["wds"].keys():
            for key in self.methods:
                self.metrics.setdefault(f"{layer}_{key}_{name}", []).append(
                    metrics_dict[key][layer]
                )

    def remove_bias(self, metrics):
        return {
            k.rsplit(".weight", 1)[0]: v
            for k, v in metrics.items()
            if k.endswith(".weight")
        }

    def get_metrics(self):
        return self.metrics
