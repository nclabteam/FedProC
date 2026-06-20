import json
import os

import numpy as np

from .pFL import pFL, pFL_Client


class hFL(pFL):
    """
    Base class for federated learning with heterogeneous models.

    Clients can use different model architectures. Model assignment
    supports round-robin, wrap-around, and random modes with optional
    ratio specification.
    """

    optional = {
        "models": "DLinear,TSMixer",
        "model_config": "",
        "model_assign": "robin",
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--models", type=str, default=None)
        parser.add_argument("--model_config", type=str, default=None)
        parser.add_argument(
            "--model_assign",
            type=str,
            default=None,
            choices=["robin", "wrap", "random"],
        )

    def __init__(self, configs, times):
        self.set_configs(configs=configs, times=times)
        self.model_map = self._build_model_map(configs)
        configs._hfl_model_map = self.model_map
        super().__init__(configs, times)
        self._export_model_config()

    def _parse_models_str(self, models_str):
        """Parse 'DLinear:3,PatchTST:2,CMoS:1' -> {'DLinear': 3, 'PatchTST': 2, 'CMoS': 1}

        If no ratios given ('DLinear,PatchTST,CMoS'), each gets ratio 1.
        """
        result = {}
        for part in models_str.split(","):
            part = part.strip()
            if ":" in part:
                name, count = part.split(":", 1)
                result[name.strip()] = int(count.strip())
            else:
                result[part] = 1
        return result

    def _build_model_map(self, configs):
        """Build per-client model assignment. Returns list of dicts with 'client', 'model', 'params'."""
        if self.model_config:
            with open(self.model_config, encoding="utf-8") as f:
                return json.load(f)

        models_dict = self._parse_models_str(self.models)
        model_list = []
        for name, count in models_dict.items():
            model_list.extend([name] * count)

        n = configs.num_clients
        if self.model_assign == "robin":
            assignments = [model_list[i % len(model_list)] for i in range(n)]
        elif self.model_assign == "wrap":
            assignments = [model_list[i % len(model_list)] for i in range(n)]
        elif self.model_assign == "random":
            rng = np.random.default_rng(configs.seed)
            assignments = rng.choice(model_list, size=n).tolist()
        else:
            assignments = [configs.model] * n

        result = []
        for i, m in enumerate(assignments):
            model_cls = self._get_objective_function("models", m)
            params = dict(getattr(model_cls, "optional", {}))
            result.append({"client": i, "model": m, "params": params})
        return result

    def _export_model_config(self):
        """Save resolved model_config.json for reproducibility."""
        path = os.path.join(self.save_path, "model_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_map, f, indent=2)

    def aggregate_client_updates(self, packages) -> None:
        """No aggregation — heterogeneous models cannot be averaged."""

    def evaluate_generalization(self, *args, **kwargs) -> None:
        """No generalization eval — no shared server model."""

    def save_models(self, save_type: str) -> None:
        """Only save client models (no server model in hFL)."""
        if save_type not in ["last", "best"]:
            raise ValueError("save_type must be 'last' or 'best'")

        should_save = True
        if save_type == "best":
            metric_key = "personal_avg_test_loss"
            if metric_key not in self.metrics or not self.metrics[metric_key]:
                should_save = False
            else:
                metric_values = self.metrics[metric_key]
                if metric_values[-1] != min(metric_values):
                    should_save = False

        if not should_save:
            return

        for client in self.clients:
            client.save_model(
                model=client.model,
                path=client.model_path,
                name=client.name,
                postfix=save_type,
                configs=client.configs,
                metadata={"save_type": save_type, "owner": client.name},
                verbose=client.logger,
            )

    def early_stopping(self) -> bool:
        metric = self.metrics["personal_avg_test_loss"]
        if not self.patience or len(metric) < self.patience:
            return False
        best_so_far = min(metric)
        if best_so_far not in metric[-self.patience :]:
            self.logger.info("Early stopping activated.")
            return True
        return False

    def get_model_info(self):
        import gc

        import torch.nn as nn

        super().get_model_info()
        for client in self.clients:
            if isinstance(client.model, nn.Module):
                dl = client.load_train_data()
                client.summarize_model(dataloader=dl)
                del dl
                gc.collect()


class hFL_Client(pFL_Client):
    """Client that reads its model assignment from the hFL model map."""

    def __init__(self, *args, **kwargs):
        configs = kwargs.get("configs", args[1] if len(args) > 1 else None)
        client_id = kwargs.get("id", args[2] if len(args) > 2 else None)
        if hasattr(configs, "_hfl_model_map") and client_id is not None:
            client_cfg = configs._hfl_model_map[client_id]
            configs.model = client_cfg.get("model", configs.model)
            for k, v in client_cfg.get("params", {}).items():
                setattr(configs, k, v)
        super().__init__(*args, **kwargs)
        self.model_name = self.model.__class__.__name__
