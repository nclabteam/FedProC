import torch

from .peftFL import peftFL, peftFL_Client


class FFA_LoRAShared:
    """Freeze random nonzero A; train and share zero-initialized B only."""

    shared_lora_suffixes = (".lora_B",)

    @staticmethod
    def setup_lora_training(model: torch.nn.Module) -> None:
        trainable = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.endswith(".lora_B")
            if parameter.requires_grad:
                trainable += parameter.numel()
        if not trainable:
            raise RuntimeError("no trainable LoRA-B parameters found")


class FFA_LoRA(FFA_LoRAShared, peftFL):
    """FFA-LoRA: mean(B_i) A0 = mean(B_i A0) (Sun et al., 2024)."""


class FFA_LoRA_Client(FFA_LoRAShared, peftFL_Client):
    """FFA-LoRA worker; A and the base model remain frozen."""
