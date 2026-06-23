import copy
import json
import os
from collections import OrderedDict

import numpy as np

from .base import SharedMethods
from .pFL import pFL, pFL_Client


class hFL_Trainer:
    """Per-client worker pool for heterogeneous-architecture FL.

    Unlike the standard Trainer (one reusable worker), hFL needs one worker per
    client because each client has a fixed model architecture that cannot be
    hot-swapped on a single worker. Workers are created once at __init__ with
    configs already patched to the correct model architecture.
    """

    def __init__(self, server, client_cls, configs, times) -> None:
        self.server = server
        self.workers: dict = {}
        for entry in server.model_map:
            cid = entry["client"]
            client_cfg = copy.deepcopy(configs)
            client_cfg.model = entry["model"]
            for k, v in entry.get("params", {}).items():
                setattr(client_cfg, k, v)
            self.workers[cid] = client_cls(
                configs=client_cfg, times=times, device=client_cfg.device
            )

    def train(self, selected) -> OrderedDict:
        packages: OrderedDict = OrderedDict()
        for cid in selected:
            pkg = self.server.package(cid)
            self.server._downlink_sizes[cid] = self.server.get_size(pkg)
            out = self.workers[cid].train(pkg)
            self.server._uplink_sizes[cid] = self.server.get_size(out)
            self._write_back(cid, out)
            packages[cid] = out
        return packages

    def evaluate(self, ids, global_params, dataset_type, current_iter):
        return [
            self.workers[cid].evaluate_global(
                cid, global_params, dataset_type, current_iter
            )
            for cid in ids
        ]

    def evaluate_personalized(self, ids, global_params, personal_map, dataset_type, current_iter):
        return [
            self.workers[cid].evaluate_personalized(
                cid, global_params, personal_map[cid], dataset_type, current_iter
            )
            for cid in ids
        ]

    def _write_back(self, cid, out) -> None:
        self.server.client_optimizer_states[cid] = out["optimizer_state"]
        self.server.client_scheduler_states[cid] = out["scheduler_state"]
        self.server.clients_personal_model_params[cid].update(out["personal_model_params"])


class hFL(pFL):
    """
    Base class for federated learning with heterogeneous models.

    Clients can use different model architectures. Model assignment
    supports round-robin, wrap-around, and random modes with optional
    ratio specification.

    No global aggregation — each client trains its own model independently
    (nFL-style per-client storage).
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
        # Replace standard single-worker Trainer with per-client-worker Trainer.
        # super().__init__() created a Trainer with one worker (wrong architecture for
        # non-default clients); we discard it here and rebuild with per-client configs.
        self.trainer = hFL_Trainer(self, self._client_cls(), configs, times)
        self._export_model_config()
        self.get_model_info()

    def _parse_models_str(self, models_str):
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
        path = os.path.join(self.save_path, "model_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_map, f, indent=2)

    def package(self, client_id: int) -> dict:
        result = super().package(client_id)
        personal = self.clients_personal_model_params[client_id]
        if personal:
            # Send client's own trained params; global model irrelevant (wrong arch)
            result["regular_model_params"] = dict(personal)
        else:
            # First round: send empty dict so worker uses its own random-initialized weights
            result["regular_model_params"] = {}
        return result

    def aggregate_client_updates(self, packages) -> None:
        # nFL-style: store each client's trained params; no cross-client aggregation
        for cid, pkg in packages.items():
            self.clients_personal_model_params[cid].update(pkg["regular_model_params"])

    def evaluate_generalization(self, *args, **kwargs) -> None:
        """No generalization eval — no shared server model in hFL."""

    def save_models(self, save_type: str) -> None:
        if save_type not in ["last", "best"]:
            raise ValueError("save_type must be 'last' or 'best'")

        if save_type == "best":
            metric_key = "personal_avg_test_loss"
            vals = self.metrics.get(metric_key, [])
            if not vals or vals[-1] != min(vals):
                return

        for cid, worker in self.trainer.workers.items():
            personal = self.clients_personal_model_params[cid]
            if not personal:
                continue
            worker.model.load_state_dict(personal, strict=False)
            self.save_model(
                model=worker.model,
                path=self.model_path,
                name=f"client_{cid}_{worker.model.__class__.__name__}",
                postfix=save_type,
                configs=worker.configs,
                metadata={"save_type": save_type, "owner": f"client_{cid}"},
                verbose=self.logger,
            )

    def _save_best_hook(self) -> None:
        self.save_models("best")

    def _save_last_hook(self) -> None:
        SharedMethods.save_model(
            self.model, self.model_path, self.name.strip(), "last",
            configs=self.configs, verbose=self.logger,
        )
        self.save_models("last")

    def early_stopping(self) -> bool:
        metric = self.metrics["personal_avg_test_loss"]
        if not self.patience or len(metric) < self.patience:
            return False
        if min(metric) not in metric[-self.patience:]:
            self.logger.info("Early stopping activated.")
            return True
        return False

    def get_model_info(self) -> None:
        if not isinstance(self.trainer, hFL_Trainer):
            return
        if not self.exclude_server_model_processes:
            first_cid = next(iter(self.trainer.workers))
            w = self.trainer.workers[first_cid]
            w._load_private(first_cid)
            w.id = first_cid
            w.current_iter = 0
            dl = w.load_train_data()
            self.summarize_model(dataloader=dl)
        seen_archs = set()
        for cid, worker in self.trainer.workers.items():
            arch = worker.model.__class__.__name__
            if arch in seen_archs:
                continue
            seen_archs.add(arch)
            worker.id = cid
            worker._load_private(cid)
            worker.current_iter = 0
            worker.name = f"client_{cid}_{arch}"
            worker.models_info_path = self.models_info_path
            dl = worker.load_train_data()
            worker.summarize_model(dataloader=dl)


class hFL_Client(pFL_Client):
    """Client pre-configured with its assigned model architecture.

    Architecture is injected via configs.model in hFL_Trainer before
    worker construction — no per-client logic needed here.
    """
    pass
