import torch

from .FedIT import FedIT, FedIT_Client


class FedSA_LoRA(FedIT):
    """
    FedSA-LoRA Server: Only aggregates A matrices, B stays local.

    Aggregation:
        1) Receive only A_k from each client
        2) Compute weighted average: Ā = Σ p_k A_k
        3) Broadcast Ā back to all clients
        4) Clients compute: W_i = W_0 + B_i Ā (local B, global A)
    """

    def aggregate_models(self):
        """
        Override FedIT aggregation: only aggregate A matrices.
        B matrices are never sent to server and remain client-specific.
        """
        device = next(self.model.parameters()).device

        # Build lora layer map from first client's lora_params keys
        first_client = self.client_data[0]["lora_params"]
        lora_layers = {}
        for key in first_client.keys():
            if key.endswith(".lora_A"):
                layer = key[: -len(".lora_A")]
                lora_layers.setdefault(layer, {})["A_name"] = key

        # Prepare aggregated A containers
        A_bar = {}

        # Initialize zeros using shapes from first client
        for layer, info in lora_layers.items():
            A_name = info.get("A_name")
            if not A_name:
                continue
            sample_A = first_client[A_name]
            A_bar[A_name] = torch.zeros_like(sample_A, device=device)

        # Weighted aggregation for A matrices only
        for client_data, weight in zip(self.client_data, self.weights):
            client_lora = client_data["lora_params"]
            for layer, info in lora_layers.items():
                A_name = info.get("A_name")
                if A_name in client_lora:
                    cA = client_lora[A_name].to(device)
                    A_bar[A_name].add_(cA, alpha=weight)

        # Prepare aggregated params: only A (B is not aggregated)
        aggregated_lora = {}
        for A_name, A_val in A_bar.items():
            aggregated_lora[A_name] = A_val.detach().cpu().clone()

        # Update global model with aggregated A only
        # Note: Server's B matrices remain unchanged (not used in practice)
        self.update_lora_params(self.model, aggregated_lora)

    def variables_to_be_sent(self):
        """Send only aggregated A matrices to clients (FedSA-LoRA protocol)."""
        lora_A_params = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                lora_A_params[name] = param.data.clone()
        return {"lora_params": lora_A_params}


class FedSA_LoRA_Client(FedIT_Client):
    """
    FedSA-LoRA Client: Only sends A, keeps B local across all rounds.

    Local training:
        - Both A and B are trainable
        - After training, only A is sent to server
        - B remains local (personalized component)

    Receive from server:
        - Only updates local A with aggregated Ā
        - Local B is preserved across rounds
    """

    def variables_to_be_sent(self):
        """Send only A matrices to server, keep B local."""
        lora_A_params = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name:  # Only send A matrices
                lora_A_params[name] = param.data.clone().to("cpu")
        return {"lora_params": lora_A_params, "score": self.train_samples}

    def receive_from_server(self, data):
        """
        Update local model with aggregated A from server.
        Preserve local B matrices (client-specific knowledge).
        """
        if "lora_params" in data:
            # Only update A parameters, B remains unchanged
            for name, new_param in data["lora_params"].items():
                if "lora_A" in name:  # Only update A
                    for param_name, param in self.model.named_parameters():
                        if param_name == name:
                            param.data.copy_(new_param.to(param.device))
                            break

            # Ensure training setup (A trainable, others frozen)
            self.setup_lora_training(self.model)
