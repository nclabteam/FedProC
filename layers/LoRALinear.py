import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, original_layer, r=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrices: W + BA (where B @ A approximates ΔW)
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # LoRA A matrix: [in_features, r] - initialized with Kaiming uniform
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        # LoRA B matrix: [r, out_features] - initialized with zeros
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        nn.init.zeros_(self.lora_B)

        # Dropout
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x):
        # Original forward: x @ W + b
        original_output = self.original_layer(x)

        # LoRA forward: x @ (A @ B) * scaling
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling

        return original_output + lora_output
