from collections import OrderedDict

import torch
import torch.nn as nn

from layers import LoRALinear

from .base import SharedMethods
from .pFL import pFL, pFL_Client

CLASS_MAPPING = {
    "Linear": nn.Linear,
    "Conv1d": nn.Conv1d,
    "Conv2d": nn.Conv2d,
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
}


class FedITShared(SharedMethods):
    @staticmethod
    def update_lora_params(model, new_lora_params):
        """Update the LoRA parameters of model with new LoRA parameters."""
        for name, param in model.named_parameters():
            if ("lora_A" in name or "lora_B" in name) and name in new_lora_params:
                param.data.copy_(new_lora_params[name].to(param.device))

    def initialize_model(self):
        # Get base model from parent class
        super().initialize_model()

        # Convert class names to actual classes
        target_classes = []
        for class_name in self.lora_target_modules:
            target_class = CLASS_MAPPING.get(class_name, None)
            if target_class is not None:
                target_classes.append(target_class)

        if not target_classes:
            raise RuntimeError(
                f"No valid target classes from: {self.lora_target_modules}"
            )

        # Find and replace modules by class type
        lora_applied = 0
        modules_to_replace = []

        for name, module in self.model.named_modules():
            # Check if module is instance of any target class
            if any(isinstance(module, target_class) for target_class in target_classes):
                modules_to_replace.append((name, module))

        if not modules_to_replace:
            raise RuntimeError(
                f"No modules of target classes found! Target classes: {[cls.__name__ for cls in target_classes]}"
            )

        # Replace modules with LoRA versions
        for name, module in modules_to_replace:
            if isinstance(module, nn.Linear):
                # Create LoRA version of the layer
                lora_layer = LoRALinear(
                    original_layer=module,
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                )

                # Replace the module in the model
                self._replace_module(self.model, name, lora_layer)
                lora_applied += 1

        if lora_applied == 0:
            raise RuntimeError("No LoRA layers were applied!")

        # Verify LoRA parameters were created
        self._verify_lora_parameters()

        # Setup training
        self.setup_lora_training(self.model)

    def _replace_module(self, model, module_name, new_module):
        """Replace a module in the model by name"""
        parts = module_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def _verify_lora_parameters(self):
        """Verify that LoRA parameters were created"""
        lora_A_count = 0
        lora_B_count = 0

        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                lora_A_count += 1
            elif "lora_B" in name:
                lora_B_count += 1

        if lora_A_count == 0 or lora_B_count == 0:
            raise RuntimeError("No LoRA parameters were created!")

    @staticmethod
    def setup_lora_training(model):
        """Ensure LoRA parameters are trainable and others are frozen"""
        trainable_params = 0
        frozen_params = 0

        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()

        if trainable_params == 0:
            raise RuntimeError("No trainable LoRA parameters found!")


class FedIT(pFL, FedITShared):
    """FedIT: Federated Instruction Tuning (Zhang et al., 2023).

    Applies LoRA (Low-Rank Adaptation) to selected model layers, then
    aggregates only the LoRA A and B matrices via FedAvg each round. Base
    model weights are frozen; only LoRA parameters are communicated.

    Note: FedAvg of A and B separately introduces aggregation bias
    (B̄·Ā ≠ mean(B_k·A_k)). Use FFA_LoRA to eliminate this bias.

    Default: r=8, α=32, dropout=0.1, target_modules=["Linear"].
    Reference: arXiv:2305.05644.
    """

    optional = {
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": ["Linear"],
    }

    compulsory = {
        "exclude_server_model_processes": True,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--lora_r", default=None, type=int)
        parser.add_argument("--lora_alpha", default=None, type=int)
        parser.add_argument("--lora_dropout", default=None, type=float)
        parser.add_argument(
            "--lora_target_modules",
            default=None,
            nargs="+",
            help="List of target module class names (e.g., Linear Conv1d)",
        )

    def aggregate_client_updates(self, packages) -> None:
        """FedIT: average A and B separately (biased aggregation ΔW' = B̄·Ā)."""
        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        cids = list(packages.keys())

        lora_names = [
            name
            for name in packages[cids[0]]["regular_model_params"]
            if "lora_" in name
        ]

        aggregated = {}
        for name in lora_names:
            stacked = torch.stack(
                [packages[cid]["regular_model_params"][name].float() for cid in cids],
                dim=-1,
            )
            w = torch.tensor([packages[cid]["score"] / total for cid in cids])
            aggregated[name] = torch.sum(stacked * w.to(stacked.dtype), dim=-1).to(
                packages[cids[0]]["regular_model_params"][name].dtype
            )

        self.update_lora_params(self.model, aggregated)
        self._commit_global(
            OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )


class FedIT_Client(pFL_Client, FedITShared):
    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        self.setup_lora_training(self.model)
