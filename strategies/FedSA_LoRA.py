from argparse import Namespace
from typing import Any

from .peftFL import peftFL, peftFL_Client
from .pFL import pFL, pFL_Client


class FedSA_LoRA(pFL, peftFL):
    """FedSA-LoRA: aggregate general A while each client keeps personal B."""

    shared_lora_suffixes = (".lora_A",)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        initial_b = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if name.endswith(".lora_B")
        }
        for personal in self.clients_personal_model_params.values():
            personal.update(
                {name: parameter.clone() for name, parameter in initial_b.items()}
            )

    def package(self, client_id: int) -> dict[str, Any]:
        package = super().package(client_id=client_id)
        package["__wire__"] = ("lora_model_params",)
        return package


class FedSA_LoRA_Client(pFL_Client, peftFL_Client):
    """FedSA-LoRA worker; only A crosses the wire."""

    shared_lora_suffixes = (".lora_A",)

    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self.personal_params_name = [
            name
            for name, _ in self.model.named_parameters()
            if name.endswith(".lora_B")
        ]

    def package(self) -> dict[str, Any]:
        package = super().package()
        package["__wire__"] = ("lora_model_params", "score")
        return package
