from collections import OrderedDict
from typing import Any

import torch

from .tFL import tFL, tFL_Client


class qFL(tFL):
    """Traditional FL with a quantized whole-update uplink."""

    def _compute_send_mb(self, packages: Any) -> tuple:
        uplink_mb = (
            self.quantized_uplink_mb(
                model_params=self.public_model_params,
                levels=self.s,
            )
            if self.s > 0
            else self.get_size(obj=self.public_model_params)
        )
        return (
            {cid: uplink_mb for cid in packages},
            sum(self._downlink_sizes.get(cid, 0.0) for cid in self.selected_clients),
        )

    def aggregate_client_updates(self, packages: Any) -> None:
        mean_delta = self.mean_models(
            models=[package["quantized_delta"] for package in packages.values()]
        )
        self._commit_global(
            new_params=OrderedDict(
                (
                    name,
                    parameter + mean_delta[name].to(parameter.device),
                )
                for name, parameter in self.public_model_params.items()
            )
        )


class qFL_Client(tFL_Client):
    """Shared whole-vector quantized-delta packaging."""

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package=package)
        self._init_params = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
        }

    def package(self) -> dict:
        result = super().package()
        deltas = OrderedDict(
            (
                name,
                local.float() - self._init_params[name].float(),
            )
            for name, local in result["regular_model_params"].items()
        )
        flat = torch.cat([delta.reshape(-1) for delta in deltas.values()])
        if self.s > 0:
            flat = self.quantize_tensor(tensor=flat, levels=self.s)

        offset = 0
        for name, delta in deltas.items():
            deltas[name] = flat[offset : offset + delta.numel()].view_as(delta)
            offset += delta.numel()

        result["regular_model_params"] = {}
        result["quantized_delta"] = deltas
        result["__wire__"] = ("quantized_delta",)
        return result
