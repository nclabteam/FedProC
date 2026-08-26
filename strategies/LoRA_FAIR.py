from argparse import ArgumentParser
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .peftFL import peftFL, peftFL_Client


class LoRA_FAIR(peftFL):
    """LoRA-FAIR: refine averaged LoRA-B to match the averaged full update."""

    optional = {
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "lora_delta_steps": 1000,
        "lora_delta_lr": 1e-2,
        "lora_delta_reg": 1e-2,
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        super().args_update(parser=parser)
        parser.add_argument("--lora_delta_steps", default=None, type=int)
        parser.add_argument("--lora_delta_lr", default=None, type=float)
        parser.add_argument("--lora_delta_reg", default=None, type=float)

    def aggregate_client_updates(self, packages: Mapping[int, dict[str, Any]]) -> None:
        if (
            self.lora_delta_steps <= 0
            or self.lora_delta_lr <= 0
            or self.lora_delta_reg < 0
        ):
            raise ValueError("LoRA-FAIR refinement settings must be positive")

        client_params, client_scores = self.extract_models_and_scores(
            packages=packages,
            model_key="lora_model_params",
        )
        scores = torch.as_tensor(
            client_scores,
            dtype=torch.float32,
        )
        if not torch.isfinite(scores).all() or scores.sum() <= 0:
            raise ValueError("LoRA-FAIR client scores must have a positive sum")
        weights = scores / scores.sum()
        averaged = self.mean_models(models=client_params, weights=scores)
        layers = self.lora_layers(params=averaged)
        device = next(self.model.parameters()).device

        bars: dict[str, tuple[str, str, torch.Tensor, torch.Tensor]] = {}
        targets: dict[str, torch.Tensor] = {}
        deltas: dict[str, torch.Tensor] = {}
        for layer, names in layers.items():
            if set(names) != {"A", "B"}:
                continue
            a_name, b_name = names["A"], names["B"]
            client_a = torch.stack(
                [params[a_name].float() for params in client_params]
            ).to(device)
            client_b = torch.stack(
                [params[b_name].float() for params in client_params]
            ).to(device)
            a_bar = averaged[a_name].float().to(device)
            b_bar = averaged[b_name].float().to(device)
            bars[layer] = (a_name, b_name, a_bar, b_bar)
            targets[layer] = torch.tensordot(
                weights.to(device), torch.bmm(client_a, client_b), dims=([0], [0])
            )
            delta = torch.empty_like(b_bar)
            torch.nn.init.xavier_uniform_(delta)
            deltas[layer] = delta.requires_grad_()

        optimizer = torch.optim.SGD(list(deltas.values()), lr=self.lora_delta_lr)
        for _ in range(self.lora_delta_steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for layer, (_, _, a_bar, b_bar) in bars.items():
                # Paper Eq. 8, transposed to FedProC storage:
                # min_dB S(mean(A_i B_i), A_bar (B_bar + dB)) + lambda ||dB||^2.
                prediction = a_bar @ (b_bar + deltas[layer])
                discrepancy = 1 - F.cosine_similarity(
                    prediction.flatten(), targets[layer].flatten(), dim=0
                )
                losses.append(
                    discrepancy + self.lora_delta_reg * deltas[layer].square().sum()
                )
            torch.stack(losses).sum().backward()
            optimizer.step()

        corrected = OrderedDict(averaged)
        for layer, (_, b_name, _, b_bar) in bars.items():
            corrected[b_name] = (
                (b_bar + deltas[layer].detach()).cpu().to(averaged[b_name].dtype)
            )
        self.update_lora_params(model=self.model, params=corrected)
        self._commit_global(
            new_params=OrderedDict(
                (name, value.detach().cpu().clone())
                for name, value in self.model.named_parameters()
            )
        )


class LoRA_FAIR_Client(peftFL_Client):
    """LoRA-FAIR worker; local training is standard two-factor LoRA."""
