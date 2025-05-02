import torch
import torch.nn as nn

from .base import Server

optional = {
    "server_epochs": 1,
    "reg_distance": "cos",
    "server_lr": 0.01,
    "server_optimizer": "Adam",
    "gamma": 1.0,
}


# Argument parser update function
def args_update(parser):
    parser.add_argument("--server_epochs", type=int, default=None)
    parser.add_argument(
        "--reg_distance", type=str, default=None, choices=["cos", "euc"]
    )
    parser.add_argument("--server_lr", type=float, default=None)
    parser.add_argument(
        "--server_optimizer", type=str, default=None, choices=["SGD", "Adam"]
    )
    parser.add_argument("--gamma", type=float, default=None)


class FedAWA(Server):
    """
    Paper: https://arxiv.org/abs/2503.15842
    Source: https://github.com/ChanglongShi/FedAWA/blob/main/server_funct.py
    """

    awa_weights = None

    @staticmethod
    def _flatten_params(model):
        """Helper function to flatten model parameters into a single vector."""
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    @staticmethod
    def _cost_matrix(x, y, dis="cos", p=2):
        """
        Calculates the cost matrix between representations x and y.
        Adapted from the original FedAWA code.

        Args:
            x: Tensor (e.g., global model flat params), shape [..., features]
            y: Tensor (e.g., client model flat params), shape [..., features]
            dis: Distance metric ('cos' or 'euc').
            p: Power for Euclidean distance.

        Returns:
            Tensor: Cost matrix.
        """
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if dis == "cos":
            # Cosine distance: 1 - cosine similarity
            d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
            C = 1 - d_cosine(x_col, y_lin)
        elif dis == "euc":
            # Average L_p distance across the feature dimension
            C = torch.mean((torch.abs(x_col - y_lin)) ** p, -1)
        else:
            raise ValueError(f"Unsupported distance type: {dis}")
        return C

    def calculate_aggregation_weights(self):
        """
        Learns aggregation weights using the FedAWA server optimization process.
        Updates self.weights with the learned, scaled, and normalized weights.
        Updates self.awa_weights with the raw learned logits for the next round.
        """

        num_clients = len(self.client_data)

        # 1. Initialize or retrieve learnable weights (logits)
        if self.awa_weights is None or self.awa_weights.shape[0] != num_clients:
            # Initialize based on sample counts for the first time or if client set changes
            ts = torch.tensor(
                [client["train_samples"] for client in self.client_data],
                dtype=torch.float32,
                device=self.device,
            )
            # Initialize logits (e.g., log of counts) for stability. Add epsilon for non-zero counts.
            self.awa_weights = (
                torch.log(ts + 1e-9).clone().detach().requires_grad_(True)
            )

            obj = self._get_objective_function("optimizers", self.server_optimizer)
            self.optimizer = obj(
                params=[self.awa_weights],
                configs=type(
                    "Config",
                    (),
                    {
                        "learning_rate": self.server_lr,
                        "momentum": 0.9,
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "epsilon": 1e-8,
                        "weight_decay": 0,
                        "amsgrad": False,
                    },
                )(),
            )
        else:
            # Use weights (logits) from the previous round
            self.awa_weights = self.awa_weights.clone().detach().requires_grad_(True)

        # Ensure weights are on the correct device and require gradients
        self.awa_weights = self.awa_weights.to(self.device).requires_grad_(True)

        # 2. Prepare models and flatten parameters
        # Detach parameters as they are constants during weight optimization
        global_flat = self._flatten_params(self.model).detach()
        client_models = [client["model"] for client in self.client_data]
        client_flats = torch.stack(
            [self._flatten_params(m).detach() for m in client_models]
        )

        # 4. Server optimization loop
        for i in range(self.server_epochs):
            self.optimizer.zero_grad()

            # Current probabilities from logits
            probability_train = torch.nn.functional.softmax(self.awa_weights, dim=0)

            # --- Calculate Regularization Loss (Distance-based) ---
            # Cost matrix between global model and each client model (flattened)
            # Input shapes: global_flat [flat_dim], client_flats [num_clients, flat_dim]
            # Need global_flat unsqueezed to [1, flat_dim] for cost_matrix function
            C = self._cost_matrix(
                x=global_flat.unsqueeze(0),
                y=client_flats,
                dis=self.reg_distance,
            )
            # C shape is likely [1, num_clients] after mean over features in _cost_matrix
            # Ensure dimensions match for multiplication with probability_train [num_clients]
            reg_loss = torch.sum(
                probability_train * C.squeeze(0)
            )  # Squeeze C if needed

            # --- Calculate Similarity Loss (Update direction-based) ---
            # Client updates relative to the global model
            # tau_local = theta_local - theta_global (equation 1)
            client_updates = client_flats - global_flat  # Shape [num_clients, flat_dim]
            # Weighted average of client updates
            # tau_global = sum of trainable_coffeicients * tau_local (equation 1)
            weighted_avg_update = torch.sum(
                client_updates * probability_train.unsqueeze(1), dim=0, keepdim=True
            )  # Shape [1, flat_dim]
            # L2 distance between each client update and the weighted average update
            # equation 3 - first half
            l2_distance = torch.norm(
                client_updates - weighted_avg_update, p=2, dim=1
            )  # Shape [num_clients]
            # equation 3 - second half
            sim_loss = torch.sum(probability_train * l2_distance)

            # --- Total Loss ---
            # equation 3 - full loss
            total_loss = sim_loss + reg_loss

            # Backpropagate gradients w.r.t awa_weights
            total_loss.backward()
            self.optimizer.step()

        # 5. Store final weights
        # Detach the learned weights (logits) for storage and use in aggregation
        final_awa_weights_logits = self.awa_weights.detach().clone()
        # Store raw weights (logits) for potential use in the next round
        self.awa_weights = final_awa_weights_logits

        # Calculate final aggregation probabilities
        final_probabilities = torch.nn.functional.softmax(
            final_awa_weights_logits, dim=0
        )

        # Apply gamma scaling
        scaled_weights = final_probabilities * self.gamma

        # Renormalize weights to sum to 1 for standard aggregation framework
        # This ensures the aggregation behaves like a weighted average.
        # The original FedAWA implementation might handle gamma differently,
        # but renormalization is safer for general use.
        sum_scaled_weights = scaled_weights.sum()
        if sum_scaled_weights > 1e-8:  # Avoid division by zero
            self.weights = scaled_weights / sum_scaled_weights
        else:
            # Fallback to uniform weights if learned weights are too small/zero
            self.weights = torch.ones_like(final_probabilities) / num_clients
