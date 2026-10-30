import numpy as np
import torch

from topologies import TOPOLOGIES

from .DFL import DFL, DFL_Client

optional = {
    "topology": "FullyConnected",
    "max_lr_upper_bound": 0.02,
    "max_lr_lower_bound": 0.0001,
    "swin": 5,
    "time_tunning": 1,
    "metric_tunning": "[wds,crs]",
    "start_tunning": 10,
    "kt": 0.1,
}


compulsory = {
    "save_local_model": True,
    "exclude_server_model_processes": True,
}


def args_update(parser):
    parser.add_argument("--topology", type=str, default=None, choices=TOPOLOGIES)
    parser.add_argument("--swin", type=int, default=None, help="Sliding window size")
    parser.add_argument("--start_tunning", type=int, default=None)
    parser.add_argument("--time_tunning", type=int, default=None)
    parser.add_argument("--metric_tunning", type=str, default=None)
    parser.add_argument("--kt", type=float, default=None)
    parser.add_argument("--max_lr_upper_bound", type=float, default=None)
    parser.add_argument("--max_lr_lower_bound", type=float, default=None)


class _ParamSnapshot:
    """CPU snapshot of model parameters — provides named_parameters() without GPU memory."""

    def __init__(self, model):
        self._params = [(n, p.detach().cpu()) for n, p in model.named_parameters()]

    def named_parameters(self):
        return iter(self._params)


class FedA2L(DFL):
    pass


class FedA2L_Client(DFL_Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calculate_metric = ModelMetrics(self.metrics)
        self.count = 0
        self.metric_tunning = self.metric_tunning.strip("[]").split(",")
        self.metric_t = {}
        self.metric_a = {}
        self.ratio_tunning = [0.6, 0.4]
        self.iterations = 0

    @staticmethod
    def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        DFL_Client.train_one_epoch(
            model, dataloader, optimizer, criterion, scheduler, device
        )
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

    def initialize_optimizer(self):
        self.layer_names = set()
        for name, _ in self.model.named_parameters():
            if name.endswith(".weight"):
                layer_name = ".".join(name.split(".")[:-1])
                self.layer_names.add(layer_name)

        param_groups = []
        assigned_params = set()
        deep_model = len(self.layer_names) > 10
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                params = list(module.parameters())
                if params:
                    unique_params = [p for p in params if p not in assigned_params]
                    if unique_params:
                        if deep_model:
                            if "bn" in name.lower():
                                current_lr = (
                                    float(self.learning_rate) * float(self.kt) * 0.5
                                )
                            else:
                                current_lr = float(self.learning_rate) * float(self.kt)
                        else:
                            current_lr = float(self.learning_rate)
                        param_groups.append(
                            {
                                "params": unique_params,
                                "lr": current_lr,
                                "name": name,
                            }
                        )
                        assigned_params.update(unique_params)

        self.optimizer = getattr(__import__("optimizers"), self.optimizer)(
            params=param_groups, configs=self.configs
        )

    def train(self):
        # [§3.3 Step 1] Capture θ^{B,r} before local training for σ computation
        betrain_model = _ParamSnapshot(self.model)
        client_package = super().train()
        # [eq:sigma_wd] σ_{i,l}^r = ||θ^{T,r} - θ^{B,r}||_2 / (||θ^{B,r}||_2 + ε)
        # model_before=θ^B, model_after=θ^T (B→T transition: local training)
        self.metric_t = self.calculate_metric.calculate_metric(
            method_type=self.metric_tunning[0],
            model_before=betrain_model,
            model_after=self.model,
        )
        del betrain_model
        return client_package

    def aggregate_models(self):
        self.iterations += 1  # r — round counter [Algorithm 1]
        if self.iterations == 1:
            super().aggregate_models()
            return

        # [§3.3 Step 1] Capture θ^{T,r} before aggregation for ζ computation
        beagg_model = _ParamSnapshot(self.model)
        super().aggregate_models()

        # [eq:zeta_stability] ζ_{i,l}^r = (1/|θ|) Σ 𝟙[|θ^{A,r} - θ^{T,r}| < τ]
        # model_before=θ^T, model_after=θ^A (T→A transition: network consensus)
        self.metric_a = self.calculate_metric.calculate_metric(
            method_type=self.metric_tunning[1],
            model_before=beagg_model,
            model_after=self.model,
        )
        del beagg_model

        # [§3.3 Step 2] Warm-up: apply base LR for first R_warm=start_tunning rounds
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
            # [§3.3 Step 2] Sliding window of last ρ=swin rounds for Z-score
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

            # [eq:omega_zscore_window / eq:delta_zscore_window]
            # ω = (σ^r - mean_window) / (STD_window + ε)
            # δ = (ζ^r - mean_window) / (STD_window + ε)
            metric_mean = torch.mean(metrics_tensor, dim=1)
            metric_std = torch.std(metrics_tensor, dim=1) + epsilon
            metric_score = (metrics_tensor[:, -1] - metric_mean) / metric_std

            # [eq:lambda] λ_{i,l}^r = β·e^ω + (1-β)·e^δ
            lambda_l = torch.dot(
                torch.tensor(self.ratio_tunning, dtype=torch.float32),
                torch.exp(metric_score),
            )

            # [eq:final_lr_per_layer]
            # η_{i,l}^r = η^0 · (1 + tanh(log(λ))) · γ^r,  γ^r = (1 + ξ·r)^{-0.5}
            layer_lr = (
                self.learning_rate
                * (1 + torch.tanh(torch.log(lambda_l)))
                / torch.sqrt(
                    torch.tensor(1 + self.kt * self.count, dtype=torch.float32)
                )
            )
            layer_lr = torch.clamp(
                layer_lr, self.max_lr_lower_bound, self.max_lr_upper_bound
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
        # [eq:sigma_wd] σ_{i,l}^r = ||θ^T - θ^B||_2 / (||θ^B||_2 + ε)
        divergence_dict = {}
        with torch.no_grad():
            for (name, param_before), (_, param_after) in zip(
                model_before.named_parameters(), model_after.named_parameters()
            ):
                param_after = param_after.cpu()
                param_diff = param_after - param_before
                divergence = torch.norm(param_diff) / torch.norm(param_before + 1e-8)
                divergence_dict[name] = divergence.item()
        return self.remove_bias(divergence_dict)

    def calculate_consensus_ratio(self, model_before, model_after, threshold=1e-3):
        # [eq:zeta_stability] ζ_{i,l}^r = (1/|θ|) Σ 𝟙[|θ^A - θ^T| < τ]
        consensus_dict = {}
        with torch.no_grad():
            for (name, param_before), (_, param_after) in zip(
                model_before.named_parameters(), model_after.named_parameters()
            ):
                param_after = param_after.cpu()
                param_diff = param_after - param_before
                consensus = (
                    param_diff.abs_() < threshold
                ).sum().item() / param_diff.numel()
                consensus_dict[name] = consensus
        return self.remove_bias(consensus_dict)

    def calculate_convergence_speed(self, model_before, model_after):
        convergence_dict = {}
        with torch.no_grad():
            for (name, param_before), (_, param_after) in zip(
                model_before.named_parameters(), model_after.named_parameters()
            ):
                param_after = param_after.cpu()
                param_diff = param_after - param_before
                convergence_speed = param_diff.abs_().mean().item()
                convergence_dict[name] = self.softsign(convergence_speed)
        return self.remove_bias(convergence_dict)

    def calculate_weight_variance(self, model_before, model_after):
        variance_dict = {}
        with torch.no_grad():
            for (name, param_before), (_, param_after) in zip(
                model_before.named_parameters(), model_after.named_parameters()
            ):
                param_after = param_after.cpu()
                param_diff = param_after - param_before
                weight_variance = (torch.norm(param_diff) / len(param_after)).item()
                variance_dict[name] = self.softsign(weight_variance)
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
        return

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

    def save_all_metrics(self, metrics_dict, name):
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
