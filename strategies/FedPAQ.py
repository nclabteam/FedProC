import math
from typing import Any, Dict

import torch

from .qFL import qFL, qFL_Client


class FedPAQShared:
    """FedPAQ's norm-based stochastic quantization rules."""

    @staticmethod
    def quantize_tensor(tensor: torch.Tensor, levels: int) -> torch.Tensor:
        norm = torch.norm(tensor)
        if norm == 0:
            return tensor
        scaled = tensor.abs().div(norm).mul(levels)
        lower = scaled.floor()
        level = lower + (torch.rand_like(tensor) < scaled - lower)
        return norm * tensor.sign() * level.div(levels)

    @staticmethod
    def quantized_uplink_mb(
        model_params: Dict[str, torch.Tensor],
        levels: int,
    ) -> float:
        dimensions = sum(param.numel() for param in model_params.values())
        magnitude_bits = math.ceil(math.log2(max(levels, 2)))
        total_bits = dimensions * (magnitude_bits + 1) + 32
        return total_bits / 8 / (1024**2)


class FedPAQ(FedPAQShared, qFL):
    """FedPAQ: Federated Learning with Periodic Averaging and Quantization (Reisizadeh et al., 2020)."""

    optional = {
        "s": 8,
    }

    @classmethod
    def args_update(cls, parser: Any) -> Any:
        parser.add_argument(
            "-s",
            "--s_levels",
            dest="s",
            type=int,
            default=None,
            help="Number of quantization levels for FedPAQ (0 = disabled)",
        )
        return parser


class FedPAQ_Client(FedPAQShared, qFL_Client):
    s: int = 8
