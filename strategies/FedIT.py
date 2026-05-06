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

    def variables_to_be_sent(self):
        """Send only LoRA parameters to reduce communication cost"""
        lora_params = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                lora_params[name] = param.data.clone()
        return {"lora_params": lora_params}

    def aggregate_models(self):
        """
        Server-side aggregation with bias (original FedIT)
        ΔW' = B̄ * Ā = (Σpk*Bk) * (Σpk*Ak)
        """
        # Initialize aggregated LoRA parameters
        aggregated_lora = {}
        first_client_lora = self.client_data[0]["lora_params"]

        # Initialize with zeros
        for name, param in first_client_lora.items():
            aggregated_lora[name] = torch.zeros_like(param)

        # Aggregate A and B matrices separately (creates bias)
        for client_data, weight in zip(self.client_data, self.weights):
            client_lora = client_data["lora_params"]
            for name, param in client_lora.items():
                if "lora_" in name:
                    aggregated_lora[name].add_(param.data, alpha=weight)

        # Update global model: This creates ΔW' = B̄ * Ā
        self.update_lora_params(self.model, aggregated_lora)


class FedIT_Client(pFL_Client, FedITShared):
    def variables_to_be_sent(self):
        """Send current LoRA parameters only"""
        lora_params = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                lora_params[name] = param.data.clone().to("cpu")
        return {"lora_params": lora_params, "score": self.train_samples}

    def receive_from_server(self, data):
        """Update local model with aggregated LoRA parameters from server"""
        if "lora_params" in data:
            self.update_lora_params(self.model, data["lora_params"])
            self.setup_lora_training(self.model)
