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
        "models": "DLinear",
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

        return [
            {"client": i, "model": m, "params": {}} for i, m in enumerate(assignments)
        ]

    def _export_model_config(self):
        """Save resolved model_config.json for reproducibility."""
        path = os.path.join(self.save_path, "model_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_map, f, indent=2)


class hFL_Client(pFL_Client):
    """Client that reads its model assignment from the hFL model map."""

    def __init__(self, *args, **kwargs):
        configs = kwargs.get("configs") or args[1]
        client_id = kwargs.get("id") or args[2] if len(args) > 2 else None
        if hasattr(configs, "_hfl_model_map") and client_id is not None:
            client_cfg = configs._hfl_model_map[client_id]
            configs.model = client_cfg.get("model", configs.model)
            for k, v in client_cfg.get("params", {}).items():
                setattr(configs, k, v)
        super().__init__(*args, **kwargs)
        self.model_name = self.model.__class__.__name__
