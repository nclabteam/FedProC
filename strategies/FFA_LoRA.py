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
    """
    FFA-LoRA Server: Aggregates only B matrices.

    A matrices are initialized once and never updated (frozen globally).
    This eliminates aggregation bias since:
        W̄ = W_0 + B̄*A_0 = W_0 + (Σ p_k B_k)*A_0 = W_0 + Σ p_k (B_k*A_0) ✓

    Unlike FedIT where both A and B are aggregated:
        W̄ = W_0 + B̄*Ā ≠ W_0 + Σ p_k (B_k*A_k) ✗ (aggregation bias)
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

    def variables_to_be_sent(self):
        """
        Send aggregated B matrices AND the server's frozen A₀ to all clients.

        Bug fix (shared A₀): Without broadcasting A₀, each client independently
        initializes its own random A_k, so ΣB_k trained against different A_k
        bases can't be coherently combined. Sending the server's A₀ every round
        ensures all clients freeze the same basis before training.
        """
        lora_params = {}
        for name, param in self.model.named_parameters():
            if "lora_B" in name or "lora_A" in name:
                lora_params[name] = param.data.clone()
        return {"lora_params": lora_params}

    def aggregate_models(self):
        """
        Aggregate only B matrices (A stays frozen).

        B̄ = Σ p_k B_k  (weighted average of trainable B matrices)
        A remains at its initial random values (never aggregated or updated)
        """
        device = next(self.model.parameters()).device

        # Build lora layer map for B matrices only
        first_client = self.client_data[0]["lora_params"]
        lora_B_names = [key for key in first_client.keys() if "lora_B" in key]

        # Initialize aggregated B parameters
        aggregated_lora = {}
        for B_name in lora_B_names:
            aggregated_lora[B_name] = torch.zeros_like(
                first_client[B_name], device=device
            )

        # Weighted aggregation for B matrices only
        for client_data, weight in zip(self.client_data, self.weights):
            client_lora = client_data["lora_params"]
            for B_name in lora_B_names:
                if B_name in client_lora:
                    aggregated_lora[B_name].add_(
                        client_lora[B_name].to(device), alpha=weight
                    )

        # Move to CPU for storage
        for B_name in lora_B_names:
            aggregated_lora[B_name] = aggregated_lora[B_name].detach().cpu().clone()

        # Update global model (only B matrices, A unchanged)
        self.update_lora_params(self.model, aggregated_lora)


class FFA_LoRA_Client(FedIT_Client, FFA_LoRAShared):
    """
    FFA-LoRA Client: Trains only B, sends only B.

    A matrices are frozen at the server's A₀ (received each round) and never
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

    def variables_to_be_sent(self):
        """
        Send only B matrices to server.

        A is frozen to the server's A₀ (received and applied in receive_from_server),
        so it is identical across all clients and does not need to be transmitted.
        """
        lora_B_params = {}
        for name, param in self.model.named_parameters():
            if "lora_B" in name:
                lora_B_params[name] = param.data.clone().to("cpu")
        return {"lora_params": lora_B_params, "score": self.train_samples}

    def receive_from_server(self, data):
        """
        Receive aggregated B from server, update local B.

        A matrices remain frozen at their initial values (never updated).
        """
        if "lora_params" in data:
            # Update only B parameters
            self.update_lora_params(self.model, data["lora_params"])
            # Ensure A stays frozen, B trainable
            self.setup_lora_training(self.model)
