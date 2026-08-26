"""Transformer/MoE point forecaster from the PFMCP paper.

PFMCP has a shared Transformer encoder and global three-layer MLP decoder.
After federation, each client freezes both modules, clones the global decoder
into a local decoder, and learns the local decoder plus a one-layer sigmoid
gate.  Conformal prediction is implemented by :mod:`strategies.PFMCP` because
it is a post-training procedure rather than a model layer.
"""

import copy
import math
from argparse import ArgumentParser, Namespace
from typing import Any, List

import torch
from torch import nn


class _ThreeLayerDecoder(nn.Module):
    """Map one encoded history vector to a multi-step forecast."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.network(encoded)


class PFMCP(nn.Module):
    """Transformer forecaster with PFMCP global/local decoder mixing."""

    optional = {
        "pfmcp_d_model": 64,
        "pfmcp_n_heads": 4,
        "pfmcp_encoder_layers": 2,
        "pfmcp_ff_dim": 128,
        "pfmcp_decoder_hidden": 128,
        "pfmcp_dropout": 0.05,
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--pfmcp_d_model", type=int, default=None)
        parser.add_argument("--pfmcp_n_heads", type=int, default=None)
        parser.add_argument("--pfmcp_encoder_layers", type=int, default=None)
        parser.add_argument("--pfmcp_ff_dim", type=int, default=None)
        parser.add_argument("--pfmcp_decoder_hidden", type=int, default=None)
        parser.add_argument("--pfmcp_dropout", type=float, default=None)

    def __init__(self, configs: Namespace) -> None:
        super().__init__()
        self.input_len = int(configs.input_len)
        self.output_len = int(configs.output_len)
        self.input_channels = int(configs.input_channels)
        self.output_channels = int(configs.output_channels)
        self.d_model = int(configs.pfmcp_d_model)

        n_heads = int(configs.pfmcp_n_heads)
        if self.d_model % n_heads != 0:
            raise ValueError("pfmcp_d_model must be divisible by pfmcp_n_heads")

        dropout = float(configs.pfmcp_dropout)
        self.input_projection = nn.Linear(self.input_channels, self.d_model)
        positions = torch.arange(self.input_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / self.d_model)
        )
        positional_encoding = torch.zeros(1, self.input_len, self.d_model)
        positional_encoding[0, :, 0::2] = torch.sin(positions * frequencies)
        positional_encoding[0, :, 1::2] = torch.cos(
            positions * frequencies[: positional_encoding[:, :, 1::2].shape[-1]]
        )
        self.register_buffer(
            "positional_encoding",
            positional_encoding,
            persistent=True,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=int(configs.pfmcp_ff_dim),
            dropout=dropout,
            activation="relu",
            batch_first=True,
            # Figure 2 shows attention -> Add & Norm -> FFN -> Add & Norm.
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(configs.pfmcp_encoder_layers),
        )

        forecast_size = self.output_len * self.output_channels
        decoder_args = (
            self.d_model,
            int(configs.pfmcp_decoder_hidden),
            forecast_size,
        )
        self.global_decoder = _ThreeLayerDecoder(*decoder_args)
        self.local_decoder = copy.deepcopy(self.global_decoder)
        self.gate = nn.Linear(self.d_model, 1)

        default_mode = getattr(configs, "pfmcp_inference_mode", "global")
        self.set_mode(default_mode)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode history and retain the latest temporal representation."""
        if x.shape[1] != self.input_len:
            raise ValueError(
                f"PFMCP expected input length {self.input_len}, "
                f"received {x.shape[1]}"
            )
        encoded = self.input_projection(x) + self.positional_encoding
        encoded = self.encoder(encoded)
        return encoded[:, -1, :]

    def set_mode(self, mode: str) -> None:
        """Select global or personalized point-forecast inference."""
        if mode not in {"global", "personalized"}:
            raise ValueError(f"Unsupported PFMCP mode: {mode}")
        self.inference_mode = mode

    def initialize_personalization(self) -> None:
        """Initialize the client decoder from the final global decoder."""
        self.local_decoder.load_state_dict(self.global_decoder.state_dict())

    def set_trainable_phase(self, phase: str) -> None:
        """Freeze the modules that the paper keeps fixed in each phase."""
        if phase not in {"federated", "personalization"}:
            raise ValueError(f"Unsupported PFMCP training phase: {phase}")

        federated = phase == "federated"
        for parameter in self.input_projection.parameters():
            parameter.requires_grad_(federated)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(federated)
        for parameter in self.global_decoder.parameters():
            parameter.requires_grad_(federated)
        for parameter in self.local_decoder.parameters():
            parameter.requires_grad_(not federated)
        for parameter in self.gate.parameters():
            parameter.requires_grad_(not federated)
        self.set_mode("global" if federated else "personalized")

    def regular_parameter_names(self) -> List[str]:
        """Return parameters trained and aggregated during FedAvg."""
        personal_prefixes = ("local_decoder.", "gate.")
        return [
            name
            for name, _ in self.named_parameters()
            if not name.startswith(personal_prefixes)
        ]

    def personal_parameter_names(self) -> List[str]:
        """Return parameters retained by an individual client."""
        personal_prefixes = ("local_decoder.", "gate.")
        return [
            name
            for name, _ in self.named_parameters()
            if name.startswith(personal_prefixes)
        ]

    def forward(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
        y_mark: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del x_mark, y_mark, kwargs
        hidden = self.encode(x)
        global_output = self.global_decoder(hidden)

        if self.inference_mode == "personalized":
            local_output = self.local_decoder(hidden)
            local_weight = torch.sigmoid(self.gate(hidden))
            output = local_weight * local_output + (1.0 - local_weight) * global_output
        else:
            output = global_output

        return output.view(x.shape[0], self.output_len, self.output_channels)
