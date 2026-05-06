import torch
import torch.nn.functional as F

from .FedIT import FedIT, FedIT_Client


class LoRA_FAIR(FedIT):

    optional = {
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_delta_steps": 1000,
        "lora_delta_lr": 1e-2,
        "lora_delta_reg": 1e-2,
        "sim_metric": "cosine",
        "lora_target_modules": ["Linear"],
    }

    compulsory = {}

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--lora_r", default=None, type=int)
        parser.add_argument("--lora_alpha", default=None, type=int)
        parser.add_argument("--lora_dropout", default=None, type=float)
        parser.add_argument("--lora_delta_steps", default=None, type=int)
        parser.add_argument("--lora_delta_lr", default=None, type=float)
        parser.add_argument("--lora_delta_reg", default=None, type=float)
        parser.add_argument(
            "--sim_metric", default=None, type=str, choices=["cosine", "l2"]
        )
        parser.add_argument(
            "--lora_target_modules",
            default=None,
            nargs="+",
            help="List of target module class names (e.g., Linear Conv1d)",
        )

    def _similarity_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        metric = getattr(self, "sim_metric", LoRA_FAIR.optional["sim_metric"])
        if metric == "cosine":
            p = pred.flatten().unsqueeze(0)
            t = target.flatten().unsqueeze(0)
            cos = F.cosine_similarity(p, t, dim=1)  # shape [1]
            return 1.0 - cos.mean()
        elif metric == "l2":
            return F.mse_loss(pred, target)
        else:
            raise ValueError(f"Unknown sim_metric: {metric}. Choose 'cosine' or 'l2'.")

    def aggregate_models(self):
        """
        Override FedIT.aggregate_models:
        1) compute A_bar, B_bar (weighted average)
        2) compute ideal W_target = Σ p_k (A_k @ B_k)
        3) optimize small ΔB per layer to minimize S(A_bar @ (B_bar+ΔB), W_target) + λ||ΔB||^2
        4) update global model with A_bar and B_bar' = B_bar + ΔB
        """
        device = next(self.model.parameters()).device

        # Build lora layer map from first client's lora_params keys
        first_client = self.client_data[0]["lora_params"]
        lora_layers = {}
        for key in first_client.keys():
            if key.endswith(".lora_A"):
                layer = key[: -len(".lora_A")]
                lora_layers.setdefault(layer, {})["A_name"] = key
            elif key.endswith(".lora_B"):
                layer = key[: -len(".lora_B")]
                lora_layers.setdefault(layer, {})["B_name"] = key

        # Prepare aggregated containers
        A_bar = {}
        B_bar = {}
        W_target = {}

        # Initialize zeros using shapes from first client
        for layer, info in lora_layers.items():
            A_name = info.get("A_name")
            B_name = info.get("B_name")
            if not (A_name and B_name):
                continue
            sample_A = first_client[A_name]
            sample_B = first_client[B_name]
            A_bar[A_name] = torch.zeros_like(sample_A, device=device)
            B_bar[B_name] = torch.zeros_like(sample_B, device=device)
            # target full product shape = [in_features, out_features] => sample_A @ sample_B
            in_features = sample_A.shape[0]
            out_features = sample_B.shape[1]
            W_target[layer] = torch.zeros((in_features, out_features), device=device)

        # Weighted aggregation for A_bar, B_bar and ideal product W_target
        for client_data, weight in zip(self.client_data, self.weights):
            client_lora = client_data["lora_params"]
            for layer, info in lora_layers.items():
                A_name = info.get("A_name")
                B_name = info.get("B_name")
                if A_name in client_lora and B_name in client_lora:
                    cA = client_lora[A_name].to(device)
                    cB = client_lora[B_name].to(device)
                    A_bar[A_name].add_(cA, alpha=weight)
                    B_bar[B_name].add_(cB, alpha=weight)
                    W_target[layer].add_(cA @ cB, alpha=weight)

        aggregated_lora = {}

        # Per-layer small optimization for ΔB
        for layer, info in lora_layers.items():
            A_name = info.get("A_name")
            B_name = info.get("B_name")
            if A_name not in A_bar or B_name not in B_bar:
                continue

            A_t = A_bar[A_name]  # [in, r]
            B_t = B_bar[B_name]  # [r, out]
            W_tgt = W_target[layer]  # [in, out]

            # delta_B parameter initialized to zero
            delta_B = torch.zeros_like(B_t, device=device, requires_grad=True)
            optimizer = torch.optim.SGD([delta_B], lr=self.lora_delta_lr)

            for _ in range(max(1, self.lora_delta_steps)):
                optimizer.zero_grad()
                pred = A_t @ (B_t + delta_B)  # [in, out]
                loss_sim = self._similarity_loss(pred, W_tgt)
                loss_reg = self.lora_delta_reg * (delta_B.pow(2).sum())
                loss = loss_sim + loss_reg
                loss.backward()
                optimizer.step()

            # finalize aggregated params (move to cpu for storage/transmission)
            aggregated_lora[A_name] = A_t.detach().cpu().clone()
            aggregated_lora[B_name] = (B_t + delta_B.detach()).cpu().clone()

        # Update global model with corrected LoRA params
        self.update_lora_params(self.model, aggregated_lora)


class LoRA_FAIR_Client(FedIT_Client):
    pass
