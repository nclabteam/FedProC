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

import torch

from .FedIT import FedIT, FedIT_Client


class FlexLoRA(FedIT):
    """
    FlexLoRA Server: Aggregate heterogeneous client ranks via SVD decomposition.

    Aggregation:
        1. Receive (A_k, B_k) from each client (may have different ranks r_k)
        2. Aggregate: W_global = Σ p_k (B_k @ A_k)  (full product matrix)
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

    compulsory = {
        "exclude_server_model_processes": True,
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

    def _resolve_client_rank(self, client_data, lora_layers):
        """Resolve a valid positive int rank for a client.

        Priority:
        1) client_data['client_rank'] if valid
        2) infer from any available LoRA A tensor shape
        3) fallback to configured self.lora_r
        """
        raw_rank = client_data.get("client_rank", None)
        if raw_rank is not None:
            try:
                rank_int = int(raw_rank)
                if rank_int > 0:
                    return rank_int
            except (TypeError, ValueError):
                pass

        for layer_info in lora_layers.values():
            a_key = layer_info.get("A_name")
            if a_key and a_key in client_data.get("lora_params", {}):
                a_tensor = client_data["lora_params"][a_key]
                if hasattr(a_tensor, "shape") and len(a_tensor.shape) >= 2:
                    # LoRALinear convention in this repo: lora_A shape is [in_features, r]
                    rank_int = int(a_tensor.shape[1])
                    if rank_int > 0:
                        return rank_int

        fallback_rank = getattr(self, "lora_r", FlexLoRA.optional["lora_r"])
        return max(1, int(fallback_rank))

    def aggregate_models(self):
        """
        Aggregate heterogeneous LoRA ranks via SVD.

        Key difference from LoRA_FAIR:
            - Aggregates to full W_global matrix (like LoRA_FAIR)
            - Then SVD decomposes and tailors back to each client's rank
            - Stores tailored components for sending back
        """
        device = next(self.model.parameters()).device

        # Build lora layer map from first client
        first_client = self.client_data[0]["lora_params"]
        lora_layers = {}
        for key in first_client.keys():
            if key.endswith(".lora_A"):
                layer = key[: -len(".lora_A")]
                lora_layers.setdefault(layer, {})["A_name"] = key
            elif key.endswith(".lora_B"):
                layer = key[: -len(".lora_B")]
                lora_layers.setdefault(layer, {})["B_name"] = key

        # Collect client ranks keyed by actual client.id (not enumeration index).
        # self.selected_clients[i] aligns with self.client_data[i].
        self.client_ranks = {}
        for i, client_data in enumerate(self.client_data):
            actual_id = self.selected_clients[i].id
            self.client_ranks[actual_id] = self._resolve_client_rank(
                client_data, lora_layers
            )

        # Store tailored LoRA for each client (computed from SVD)
        self.tailored_lora_params = {}

        # Aggregate and SVD decompose for each layer
        for layer, info in lora_layers.items():
            A_name = info.get("A_name")
            B_name = info.get("B_name")
            if not (A_name and B_name):
                continue

            # Step 1: Aggregate to full product W_global = Σ p_k (A_k @ B_k)
            # LoRALinear in this repo uses x @ A @ B, so A:[in,r], B:[r,out].
            W_global = None
            for client_data, weight in zip(self.client_data, self.weights):
                client_lora = client_data["lora_params"]
                if A_name in client_lora and B_name in client_lora:
                    A_k = client_lora[A_name].to(device)
                    B_k = client_lora[B_name].to(device)
                    W_k = A_k @ B_k
                    if W_global is None:
                        W_global = W_k * weight
                    else:
                        W_global.add_(W_k, alpha=weight)

            if W_global is None:
                continue

            # Step 2: SVD decompose
            U, Sigma, Vh = torch.linalg.svd(W_global, full_matrices=False)
            # U: [m, min(m,n)]
            # Sigma: [min(m,n)]
            # Vh: [min(m,n), n]

            # Step 3: Tailor components for each client based on their rank
            self.tailored_lora_params[layer] = {}
            for client_id, r_i in self.client_ranks.items():
                # Ensure r_i doesn't exceed SVD rank
                r_i = min(r_i, Sigma.shape[0])

                # Extract top r_i components
                U_i = U[:, :r_i]  # [m, r_i]
                Sigma_i = Sigma[:r_i]  # [r_i]
                Vh_i = Vh[:r_i, :]  # [r_i, n]

                # Reconstruct for this client's LoRA parameter convention:
                # A:[in,r], B:[r,out], forward uses x @ A @ B * (alpha/r).
                # Choose A_i = U_i and B_i = diag(Sigma_i) @ Vh_i scaled by (r_i/alpha)
                # so A_i @ B_i * (alpha/r_i) = U_i @ diag(Sigma_i) @ Vh_i.

                alpha = getattr(self, "lora_alpha", FlexLoRA.optional["lora_alpha"])
                if alpha is None or float(alpha) == 0.0:
                    scale = 1.0
                else:
                    scale = float(r_i) / float(alpha)

                A_i = U_i
                B_i = (torch.diag(Sigma_i) @ Vh_i) * scale

                # Store for later transmission
                if client_id not in self.tailored_lora_params[layer]:
                    self.tailored_lora_params[layer][client_id] = {}

                self.tailored_lora_params[layer][client_id]["B"] = B_i.detach().cpu()
                self.tailored_lora_params[layer][client_id]["A"] = A_i.detach().cpu()

        # Update server model (keep same rank as initialization)
        # Average the full W_global and decompose back to original rank
        self._update_server_model_from_w_global(device)

    def _update_server_model_from_w_global(self, device):
        """
        After SVD, update server model with averaged components (using original rank).
        This keeps server model consistent for reference.
        """
        # For simplicity, just use FedIT's regular averaging on the server side
        # (Server maintains original rank, clients get heterogeneous ranks)
        lora_params_averaged = {}

        first_client = self.client_data[0]["lora_params"]
        for key in first_client.keys():
            if "lora_" in key:
                lora_params_averaged[key] = torch.zeros_like(
                    first_client[key], device=device
                )

        for client_data, weight in zip(self.client_data, self.weights):
            for key in client_data["lora_params"].keys():
                if "lora_" in key:
                    # Simple average of original client LoRA (ignoring heterogeneous ranks)
                    # This is just for server model consistency
                    local_tensor = client_data["lora_params"][key].to(device)
                    if (
                        key in lora_params_averaged
                        and lora_params_averaged[key].shape == local_tensor.shape
                    ):
                        lora_params_averaged[key].add_(local_tensor, alpha=weight)

        # Update server model (as backup/reference)
        self.update_lora_params(self.model, lora_params_averaged)

    def variables_to_be_sent(self):
        """
        Return dictionary with per-client tailored LoRA parameters.
        """
        if not self.tailored_lora_params:
            return super().variables_to_be_sent()

        return {
            "tailored_lora_params": self.tailored_lora_params,
            "client_ranks": self.client_ranks,
        }


class FlexLoRA_Client(FedIT_Client):
    """
    FlexLoRA Client: Accept heterogeneous LoRA rank from server.

    Each client can have a different rank r_i based on available resources.
    After receiving SVD-tailored components, reconstruct (A, B) for next training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_rank = None  # Will be set during setup

    def variables_to_be_sent(self):
        """Send current LoRA parameters (A, B) to server."""
        lora_params = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                lora_params[name] = param.data.clone().to("cpu")

        # Also send client's rank for server to know
        return {
            "lora_params": lora_params,
            "score": self.train_samples,
            "client_rank": getattr(self, "client_rank", None),
        }

    def receive_from_server(self, data):
        """
        Receive SVD-tailored LoRA components from server.

        Reconstruct (A_i, B_i) from U[:, :r_i] @ diag(Σ_i) @ V[:r_i, :]^T
        """
        if "tailored_lora_params" in data:
            # data["tailored_lora_params"][layer][client_id] = {"B": B_i, "A": A_i}
            # But we need to know client_id...
            # For now, assume single-threaded and use first (only) tailored entry
            tailored = data["tailored_lora_params"]

            # Map layers to parameter names using this client's actual ID
            reconstructed_lora = {}
            for name, param in self.model.named_parameters():
                if "lora_B" in name:
                    layer = name[: -len(".lora_B")]
                    if layer in tailored and self.id in tailored[layer]:
                        B_new = tailored[layer][self.id]["B"]
                        reconstructed_lora[name] = B_new
                elif "lora_A" in name:
                    layer = name[: -len(".lora_A")]
                    if layer in tailored and self.id in tailored[layer]:
                        A_new = tailored[layer][self.id]["A"]
                        reconstructed_lora[name] = A_new

            if reconstructed_lora:
                self.update_lora_params(self.model, reconstructed_lora)
                # Update client rank if provided
                if "client_ranks" in data and self.id in data["client_ranks"]:
                    self.client_rank = data["client_ranks"][self.id]
                self.setup_lora_training(self.model)
        else:
            # Fallback to regular LoRA update
            super().receive_from_server(data)
