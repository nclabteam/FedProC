"""
FlexLoRA: Heterogeneous LoRA Ranks for Resource-Aware Federated Learning

Key insight: Clients have different computational resources, so uniform LoRA ranks
waste resources (under-utilize rich clients) or create bottlenecks (constrained by
poor clients). FlexLoRA lets each client use their own rank r_i based on resources.

Algorithm:
    1. Each client trains with their own rank r_i LoRA (A_i ∈ ℝ^(r_i × n), B_i ∈ ℝ^(m × r_i))
    2. Clients send (A_i, B_i) to server
    3. Server aggregates: W_g = Σ p_i * B_i @ A_i (full product, heterogeneous ranks)
    4. Server SVD: U, Σ, V = SVD(W_g)
    5. For each client i, server sends top r_i singular components:
       W_g^i = U[:, :r_i] @ Σ[:r_i, :r_i] @ V[:r_i, :]^T
    6. Client i decomposes: B_i ← U[:, :r_i] @ Σ[:r_i, :r_i] / α
                           A_i ← V[:r_i, :]^T
    7. Clients continue training with their rank-customized LoRA

Benefits:
    + Heterogeneous ranks naturally supported
    + Larger ranks improve generalization (empirically shown)
    + Clients fully utilize available resources
    + No bottleneck from least-resourced client
    - SVD overhead per round (but negligible vs LLM training)
    - Higher communication in first round (send tailored components)
"""

from collections import OrderedDict

import torch

from .FedIT import FedIT, FedIT_Client


class FlexLoRA(FedIT):
    """
    FlexLoRA Server: Aggregate heterogeneous client ranks via SVD decomposition.

    Aggregation:
        1. Receive (A_k, B_k) from each client (may have different ranks r_k)
        2. Aggregate: W_global = Σ p_k (A_k @ B_k)  (full product matrix)
        3. Decompose: U, Σ, V = SVD(W_global)
        4. For each client i, prepare tailored components based on their rank r_i:
           W_global_i = U[:, :r_i] @ Σ[:r_i, :r_i] @ V[:r_i, :]^T
        5. Send W_global_i back to client i for decomposition

    Result: Each client can have different rank while maintaining aggregation correctness.
    """

    optional = {
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": ["Linear"],
        "client_ranks": None,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--lora_r", default=None, type=int)
        parser.add_argument("--lora_alpha", default=None, type=int)
        parser.add_argument("--lora_dropout", default=None, type=float)
        parser.add_argument(
            "--lora_target_modules",
            default=None,
            nargs="+",
            help="List of target module class names (e.g., Linear Conv1d)",
        )
        parser.add_argument(
            "--client_ranks",
            default=None,
            type=str,
            help="JSON dict mapping client_id to LoRA rank (e.g., '{0: 4, 1: 8, 2: 16}')",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Map client_id to their LoRA rank (set during aggregation)
        self.client_ranks = {}
        # Per-client tailored LoRA from SVD (set during aggregation)
        self.tailored_lora_params = {}

    def _resolve_client_rank(self, lora_params, client_rank_hint, lora_layers):
        """Resolve a valid positive int rank for a client."""
        if client_rank_hint is not None:
            try:
                rank_int = int(client_rank_hint)
                if rank_int > 0:
                    return rank_int
            except (TypeError, ValueError):
                pass

        for layer_info in lora_layers.values():
            a_key = layer_info.get("A_name")
            if a_key and a_key in lora_params:
                a_tensor = lora_params[a_key]
                if hasattr(a_tensor, "shape") and len(a_tensor.shape) >= 2:
                    rank_int = int(a_tensor.shape[1])
                    if rank_int > 0:
                        return rank_int

        return max(1, int(self.lora_r))

    def package(self, client_id: int) -> dict:
        result = super().package(client_id)
        if self.tailored_lora_params:
            result["tailored_lora_params"] = self.tailored_lora_params
            result["client_ranks"] = self.client_ranks
        return result

    def aggregate_client_updates(self, packages) -> None:
        """Aggregate heterogeneous LoRA ranks via SVD."""
        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        cids = list(packages.keys())
        weights = [packages[cid]["score"] / total for cid in cids]

        device = next(self.model.parameters()).device

        # Build lora layer map from first client
        first_params = packages[cids[0]]["regular_model_params"]
        lora_layers = {}
        for key in first_params.keys():
            if key.endswith(".lora_A"):
                layer = key[: -len(".lora_A")]
                lora_layers.setdefault(layer, {})["A_name"] = key
            elif key.endswith(".lora_B"):
                layer = key[: -len(".lora_B")]
                lora_layers.setdefault(layer, {})["B_name"] = key

        # Resolve client ranks
        self.client_ranks = {}
        for cid in cids:
            cparams = packages[cid]["regular_model_params"]
            hint = packages[cid].get("client_rank", None)
            self.client_ranks[cid] = self._resolve_client_rank(cparams, hint, lora_layers)

        self.tailored_lora_params = {}

        lora_params_averaged = {}
        for key in first_params:
            if "lora_" in key:
                lora_params_averaged[key] = torch.zeros_like(first_params[key], device=device)

        for layer, info in lora_layers.items():
            A_name = info.get("A_name")
            B_name = info.get("B_name")
            if not (A_name and B_name):
                continue

            W_global = None
            for cid, w in zip(cids, weights):
                cparams = packages[cid]["regular_model_params"]
                if A_name in cparams and B_name in cparams:
                    A_k = cparams[A_name].to(device)
                    B_k = cparams[B_name].to(device)
                    W_k = A_k @ B_k
                    if W_global is None:
                        W_global = W_k * w
                    else:
                        W_global.add_(W_k, alpha=w)
                    # Accumulate for server model update
                    if A_name in lora_params_averaged and lora_params_averaged[A_name].shape == A_k.shape:
                        lora_params_averaged[A_name].add_(A_k, alpha=w)
                    if B_name in lora_params_averaged and lora_params_averaged[B_name].shape == B_k.shape:
                        lora_params_averaged[B_name].add_(B_k, alpha=w)

            if W_global is None:
                continue

            U, Sigma, Vh = torch.linalg.svd(W_global, full_matrices=False)

            self.tailored_lora_params[layer] = {}
            for cid, r_i in self.client_ranks.items():
                r_i = min(r_i, Sigma.shape[0])

                U_i = U[:, :r_i]
                Sigma_i = Sigma[:r_i]
                Vh_i = Vh[:r_i, :]

                if self.lora_alpha is None or float(self.lora_alpha) == 0.0:
                    scale = 1.0
                else:
                    scale = float(r_i) / float(self.lora_alpha)

                A_i = U_i
                B_i = (torch.diag(Sigma_i) @ Vh_i) * scale

                self.tailored_lora_params[layer][cid] = {
                    "B": B_i.detach().cpu(),
                    "A": A_i.detach().cpu(),
                }

        # Update server model with averaged LoRA params (for reference)
        self.update_lora_params(self.model, lora_params_averaged)
        self._commit_global(
            OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )


class FlexLoRA_Client(FedIT_Client):
    """
    FlexLoRA Client: Accept heterogeneous LoRA rank from server.

    Each client can have a different rank r_i based on available resources.
    After receiving SVD-tailored components, reconstruct (A, B) for next training.
    """

    def __init__(self, configs, times, device) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self.client_rank = None

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        # Apply tailored lora params from server's SVD decomposition
        if "tailored_lora_params" in package:
            tailored = package["tailored_lora_params"]
            reconstructed_lora = {}
            for name, param in self.model.named_parameters():
                if "lora_B" in name:
                    layer = name[: -len(".lora_B")]
                    if layer in tailored and self.id in tailored[layer]:
                        reconstructed_lora[name] = tailored[layer][self.id]["B"]
                elif "lora_A" in name:
                    layer = name[: -len(".lora_A")]
                    if layer in tailored and self.id in tailored[layer]:
                        reconstructed_lora[name] = tailored[layer][self.id]["A"]
            if reconstructed_lora:
                self.update_lora_params(self.model, reconstructed_lora)
                if "client_ranks" in package and self.id in package["client_ranks"]:
                    self.client_rank = package["client_ranks"][self.id]
        self.setup_lora_training(self.model)

    def package(self, train_time: float) -> dict:
        result = super().package(train_time)
        result["client_rank"] = self.client_rank
        return result
