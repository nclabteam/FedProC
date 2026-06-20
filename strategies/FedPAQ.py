import torch

from .tFL import tFL, tFL_Client


class FedPAQ(tFL):
    optional = {
        "s": 8,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "-s",
            "--s_levels",
            dest="s",
            type=int,
            default=None,
            help="Number of quantization levels for FedPAQ (0 = disabled)",
        )
        return parser


class FedPAQ_Client(tFL_Client):
    s: int = 8

    def package(self, train_time: float) -> dict:
        result = super().package(train_time)
        if self.s > 0:
            result["regular_model_params"] = {
                name: self.quantize_tensor(tensor, self.s)
                for name, tensor in result["regular_model_params"].items()
            }
        return result

    @staticmethod
    def quantize_tensor(tensor, s):
        norm = torch.norm(tensor)
        if norm == 0:
            return tensor
        abs_scaled = (torch.abs(tensor) / norm) * s
        l = torch.floor(abs_scaled)
        prob = abs_scaled - l
        rand = torch.rand_like(tensor)
        rounded = torch.where(rand < prob, l + 1.0, l)
        return norm * torch.sign(tensor) * (rounded / s)
