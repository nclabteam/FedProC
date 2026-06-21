from collections import OrderedDict

import torch

from .FedIT import FedIT, FedIT_Client


class FFA_LoRAShared:
    """Shared methods for FFA-LoRA: Freeze A, train only B."""

    @staticmethod
    def setup_lora_training(model):
        """
        FFA-LoRA training setup: Only B is trainable, A is frozen.

        Override FedIT's setup to ensure A matrices remain frozen throughout training.
        """
        trainable_params = 0
        frozen_params = 0

        for name, param in model.named_parameters():
            if "lora_B" in name:
                # Only B matrices are trainable
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                # Everything else frozen (including lora_A)
                param.requires_grad = False
                frozen_params += param.numel()

        if trainable_params == 0:
            raise RuntimeError("No trainable LoRA-B parameters found!")


class FFA_LoRA(FedIT, FFA_LoRAShared):
    """FFA-LoRA: Improving LoRA in Privacy-Preserving Federated Learning (Sun et al., ICLR 2024).

    A matrices are initialized once and never updated (frozen globally).
    This eliminates aggregation bias since:
        W̄ = W_0 + B̄*A_0 = W_0 + (Σ p_k B_k)*A_0 = W_0 + Σ p_k (B_k*A_0) ✓

    Unlike FedIT where both A and B are aggregated:
        W̄ = W_0 + B̄*Ā ≠ W_0 + Σ p_k (B_k*A_k) ✗ (aggregation bias)

    Reference: arXiv:2403.12313. ICLR 2024.
    """

    # Bug fix (MRO): FedITShared appears before FFA_LoRAShared in the MRO, so
    # without this explicit binding Python would resolve setup_lora_training to
    # FedITShared's version (which makes A trainable, defeating the freeze-A design).
    setup_lora_training = staticmethod(FFA_LoRAShared.setup_lora_training)

    def initialize_model(self):
        """
        Initialize model with LoRA layers, then freeze A matrices.

        A matrices get random Gaussian init (standard LoRA), then frozen.
        B matrices get zero init and remain trainable.
        """
        # Use parent's initialization (creates LoRA layers)
        super().initialize_model()

        # Override training setup to freeze A
        self.setup_lora_training(self.model)

    def aggregate_client_updates(self, packages) -> None:
        """Aggregate only B matrices (A stays frozen)."""
        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        cids = list(packages.keys())

        # Only aggregate lora_B params
        lora_B_names = [
            name
            for name in packages[cids[0]]["regular_model_params"]
            if "lora_B" in name
        ]

        aggregated = {}
        for name in lora_B_names:
            stacked = torch.stack(
                [packages[cid]["regular_model_params"][name].float() for cid in cids],
                dim=-1,
            )
            w = torch.tensor([packages[cid]["score"] / total for cid in cids])
            aggregated[name] = torch.sum(stacked * w.to(stacked.dtype), dim=-1).to(
                packages[cids[0]]["regular_model_params"][name].dtype
            )

        self.update_lora_params(self.model, aggregated)
        self._commit_global(
            OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )


class FFA_LoRA_Client(FedIT_Client, FFA_LoRAShared):
    """
    FFA-LoRA Client: Trains only B, sends only B.

    A matrices are frozen to the server's A₀ (received each round) and never
    updated locally. Only B matrices are trained and communicated.
    """

    # Bug fix (MRO): same issue as FFA_LoRA — bind explicitly so Python doesn't
    # resolve to FedITShared.setup_lora_training (which makes A trainable).
    setup_lora_training = staticmethod(FFA_LoRAShared.setup_lora_training)

    def initialize_model(self):
        """Initialize model and freeze A matrices."""
        super().initialize_model()
        # Override to freeze A
        self.setup_lora_training(self.model)
