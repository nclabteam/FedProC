from collections import OrderedDict

import torch

from .FedIT import FedIT, FedIT_Client


class FedSA_LoRA(FedIT):
    """FedSA-LoRA: Selective Aggregation for Low-Rank Adaptation (Su et al., ICLR 2025).

    Only aggregates A matrices server-side; B stays local per client.
    Per round: clients train both A and B, send only A to server (Ā = FedAvg(A_k)),
    receive Ā back; each client computes ΔW_i = B_i · Ā (local B, global A).

    Reference: arXiv:2410.01463. ICLR 2025.
    """

    def __init__(self, configs, times):
        super().__init__(configs=configs, times=times)
        lora_B_init = {
            name: param.data.cpu().clone()
            for name, param in self.model.named_parameters()
            if "lora_B" in name
        }
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid].update(
                {name: t.clone() for name, t in lora_B_init.items()}
            )

    def aggregate_client_updates(self, packages) -> None:
        """Aggregate only A matrices; B stays per-client in personal_model_params."""
        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        cids = list(packages.keys())

        # regular_model_params from client excludes lora_B (moved to personal).
        # Keys present in regular_model_params are non-B params.
        ref_keys = set(packages[cids[0]]["regular_model_params"].keys())

        new_params = OrderedDict()
        for name in self.public_model_params:
            if name in ref_keys:
                stacked = torch.stack(
                    [packages[cid]["regular_model_params"][name].float() for cid in cids],
                    dim=-1,
                )
                w = torch.tensor([packages[cid]["score"] / total for cid in cids])
                new_params[name] = torch.sum(stacked * w.to(stacked.dtype), dim=-1).to(
                    self.public_model_params[name].dtype
                )
            else:
                # lora_B: keep server's current value (clients maintain personal B)
                new_params[name] = self.public_model_params[name]

        self._commit_global(new_params)


class FedSA_LoRA_Client(FedIT_Client):
    """FedSA-LoRA Client: trains both A and B, sends only A, keeps B local.

    On each round: set_parameters restores personal lora_B (overriding the global
    placeholder B in regular_model_params), then client trains both A and B.
    package() moves lora_B to personal_model_params so the server stores but
    does not aggregate it.
    """

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        lora_B = {
            name: tensor
            for name, tensor in package["personal_model_params"].items()
            if "lora_B" in name
        }
        if lora_B:
            self.update_lora_params(self.model, lora_B)

    def package(self) -> dict:
        result = super().package()
        # Move lora_B from regular to personal so server only aggregates A
        new_regular = OrderedDict()
        for name, tensor in result["regular_model_params"].items():
            if "lora_B" in name:
                result["personal_model_params"][name] = tensor
            else:
                new_regular[name] = tensor
        result["regular_model_params"] = new_regular
        return result
