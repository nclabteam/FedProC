from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from layers import LoRALinear

from .base import SharedMethods
from .tFL import tFL, tFL_Client


class peftFLShared(SharedMethods):
    """Shared LoRA setup; FedProC stores paper ``BA`` as forward ``A @ B``."""

    lora_classes = {"Linear": nn.Linear}
    shared_lora_suffixes = (".lora_A", ".lora_B")

    @staticmethod
    def lora_params(
        params: Mapping[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (name, value)
            for name, value in params.items()
            if name.endswith((".lora_A", ".lora_B"))
        )

    @classmethod
    def shared_lora_params(
        cls, params: Mapping[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (name, value)
            for name, value in params.items()
            if name.endswith(cls.shared_lora_suffixes)
        )

    @staticmethod
    def lora_layers(
        params: Mapping[str, torch.Tensor],
    ) -> dict[str, dict[str, str]]:
        layers = {}
        for name in params:
            for suffix, key in ((".lora_A", "A"), (".lora_B", "B")):
                if name.endswith(suffix):
                    layers.setdefault(name[: -len(suffix)], {})[key] = name
        return layers

    @staticmethod
    def update_lora_params(
        model: nn.Module, params: Mapping[str, torch.Tensor]
    ) -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in params:
                    parameter.copy_(params[name].to(parameter))

    @staticmethod
    def _replace_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], replacement)

    def initialize_model(self) -> None:
        super().initialize_model()
        unknown = set(self.lora_target_modules) - self.lora_classes.keys()
        if unknown:
            raise ValueError(f"unsupported LoRA target modules: {sorted(unknown)}")
        targets = tuple(self.lora_classes[name] for name in self.lora_target_modules)
        modules = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, targets)
        ]
        if not modules:
            raise RuntimeError("no configured LoRA target modules found")
        for name, module in modules:
            self._replace_module(
                model=self.model,
                name=name,
                replacement=LoRALinear(
                    original_layer=module,
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                ),
            )
        self.setup_lora_training(model=self.model)

    @staticmethod
    def setup_lora_training(model: nn.Module) -> None:
        trainable = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.endswith((".lora_A", ".lora_B"))
            if parameter.requires_grad:
                trainable += parameter.numel()
        if not trainable:
            raise RuntimeError("no trainable LoRA parameters found")


class peftFL(peftFLShared, tFL):
    """Global parameter-efficient FL branch for LoRA adapters."""

    optional = {
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["Linear"],
    }
    compulsory = {"exclude_server_model_processes": True}

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--lora_r", default=None, type=int)
        parser.add_argument("--lora_alpha", default=None, type=int)
        parser.add_argument("--lora_dropout", default=None, type=float)
        parser.add_argument(
            "--lora_target_modules",
            default=None,
            nargs="+",
            help="LoRA target module classes (currently: Linear)",
        )

    def package(self, client_id: int) -> dict[str, Any]:
        package = super().package(client_id=client_id)
        package["lora_model_params"] = self.shared_lora_params(
            params=package["regular_model_params"]
        )
        package["__wire__"] = ("lora_model_params",)
        return package

    def aggregate_client_updates(self, packages: Mapping[int, dict[str, Any]]) -> None:
        # FedAvg adapters: theta = sum_i(n_i * theta_i) / sum_i(n_i).
        models, scores = self.extract_models_and_scores(
            packages=packages,
            model_key="lora_model_params",
        )
        averaged = self.mean_models(
            models=models,
            weights=scores,
        )
        self.update_lora_params(model=self.model, params=averaged)
        self._commit_global(
            new_params=OrderedDict(
                (name, value.detach().cpu().clone())
                for name, value in self.model.named_parameters()
            )
        )


class peftFL_Client(peftFLShared, tFL_Client):
    """Stateless worker that trains and transmits only shared LoRA factors."""

    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self.regular_params_name = [
            name
            for name, _ in self.model.named_parameters()
            if name.endswith(self.shared_lora_suffixes)
        ]

    def set_parameters(self, package: dict[str, Any]) -> None:
        super().set_parameters(package=package)
        self.update_lora_params(
            model=self.model,
            params=package.get("lora_model_params", {}),
        )
        self.setup_lora_training(model=self.model)

    def package(self) -> dict[str, Any]:
        package = super().package()
        package["lora_model_params"] = self.shared_lora_params(
            params=package["regular_model_params"]
        )
        package["__wire__"] = ("lora_model_params", "score")
        return package
