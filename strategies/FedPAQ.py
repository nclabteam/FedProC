import torch

from .tFL import tFL, tFL_Client


class FedPAQ(tFL):
    """FedPAQ: Federated Learning with Periodic Averaging and Quantization (Reisizadeh et al., 2020).

    Reduces uplink communication by stochastically quantizing client update vectors
    before upload (paper §III-C). Quantizer (low-precision, QSGD-style):
      Q_s(v)_i = ||v|| * sign(v_i) * ξ_i / s
    where ξ_i ~ Bernoulli(|v_i|/||v|| * s - floor(|v_i|/||v|| * s)) + floor(|v_i|/||v|| * s).

    Client computes delta = x_τ - x_0, uploads Q(delta). Server aggregates
    x_{k+1} = x_k + Σ_i w_i · Q(delta_i), implemented by sending x_0 + Q(delta)
    so the existing FedAvg aggregation remains correct.

    Reference: arXiv:1909.13014.
    """

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

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        # Snapshot x_0 (params received from server before local training)
        self._init_params = {
            name: param.data.clone().cpu()
            for name, param in self.model.named_parameters()
        }

    def package(self, train_time: float) -> dict:
        result = super().package(train_time)
        if self.s > 0:
            # Quantize the update vector delta = x_τ - x_0, then send x_0 + Q(delta).
            # Server FedAvg then gives x_0 + Σ w_i·Q(delta_i) as required by paper.
            quantized = {}
            for name, x_tau in result["regular_model_params"].items():
                delta = x_tau.float() - self._init_params[name].float()
                q_delta = self.quantize_tensor(delta, self.s)
                quantized[name] = (self._init_params[name].float() + q_delta).to(x_tau.dtype)
            result["regular_model_params"] = quantized
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
