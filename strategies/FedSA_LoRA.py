from collections import OrderedDict

import torch

from .FedIT import FedIT, FedIT_Client


class FedSA_LoRA(FedIT):
    """
    FedSA-LoRA Server: Only aggregates A matrices, B stays local.

    Aggregation:
        1) Receive only A_k from each client (via regular_model_params)
        2) Compute weighted average: Ā = Σ p_k A_k
        3) Broadcast Ā back to all clients (as part of public_model_params)
        4) Clients compute: W_i = W_0 + B_i Ā (local B, global A)
    """

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
    """
    FedSA-LoRA Client: Only sends A, keeps B local across all rounds.

    Local training:
        - Both A and B are trainable
        - After training, only A is sent to server (B goes to personal_model_params)
        - B remains local (personalized component)

    Receive from server:
        - Only updates local A with aggregated Ā (via regular_model_params)
        - Local B is preserved across rounds (via personal_model_params)
    """

    def package(self, train_time: float) -> dict:
        result = super().package(train_time)
        # Move lora_B from regular to personal so server only aggregates A
        new_regular = OrderedDict()
        for name, tensor in result["regular_model_params"].items():
            if "lora_B" in name:
                result["personal_model_params"][name] = tensor
            else:
                new_regular[name] = tensor
        result["regular_model_params"] = new_regular
        return result
