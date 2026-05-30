import torch

from .tFL import tFL, tFL_Client


class FedPAQ(tFL):
    optional = {
        "s": 8,
    }
    compulsory = {
        "return_diff": True,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "--s",
            type=int,
            default=None,
            help="Number of quantization levels for FedPAQ (set to 0 or leave empty to disable quantization)",
        )
        return parser


class FedPAQ_Client(tFL_Client):
    def variables_to_be_sent(self):
        # Retrieve base dictionary containing the model difference
        data = super().variables_to_be_sent()

        s = getattr(self, "s", None)
        if s is not None and s > 0:
            model = data["model"]
            with torch.no_grad():
                for param in model.parameters():
                    # Quantize each parameter tensor in-place
                    param.data.copy_(self.quantize_tensor(param.data, s))

        return data

    @staticmethod
    def quantize_tensor(tensor, s):
        # Unbiased stochastic low-precision quantization
        norm = torch.norm(tensor)
        if norm == 0:
            return tensor

        abs_scaled = (torch.abs(tensor) / norm) * s
        l = torch.floor(abs_scaled)
        prob = abs_scaled - l

        # Stochastic rounding
        rand = torch.rand_like(tensor)
        rounded = torch.where(rand < prob, l + 1.0, l)

        # Re-scale back using norm, sign, and rounded levels
        quantized = norm * torch.sign(tensor) * (rounded / s)
        return quantized
