import copy

import higher
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from .base import Client, Server

optional = {
    "L_ct": 10,  # Interval for updating client-side synthetic data
    "synthetic_data_size_ct": 100,  # Size of client-side synthetic data
    "L_gt": 10,  # Interval for updating server-side synthetic data
    "synthetic_data_size_gt": 100,  # Size of server-side synthetic data
    "synthetic_epochs": 300,  # Outer-loop epochs for synthetic data generation
    "synthetic_inner_epochs": 1,  # Inner-loop epochs for training on synthetic data
    "synthetic_lr": 3e-4,  # Learning rate for synthetic data optimization
    "refine_epochs": 1,  # Epochs to refine global model on D_gt after aggregation
}


def args_update(parser):
    # --- Client-side Synthetic Data (D_ct) Parameters ---
    parser.add_argument(
        "--L_ct",
        type=int,
        default=None,
        help="Interval in rounds for updating the client-side synthetic data (D_ct).",
    )
    parser.add_argument(
        "--synthetic_data_size_ct",
        type=int,
        default=None,
        help="Number of synthetic data samples to create for D_ct.",
    )

    # --- Server-side Synthetic Data (D_gt) Parameters ---
    parser.add_argument(
        "--L_gt",
        type=int,
        default=None,
        help="Interval in rounds for updating the server-side synthetic data (D_gt).",
    )
    parser.add_argument(
        "--synthetic_data_size_gt",
        type=int,
        default=None,
        help="Number of synthetic data samples to create for D_gt.",
    )

    # --- Synthetic Data Generation Parameters (for both D_ct and D_gt) ---
    parser.add_argument(
        "--synthetic_epochs",
        type=int,
        default=None,
        help="Outer-loop training epochs for generating the synthetic datasets.",
    )
    parser.add_argument(
        "--synthetic_inner_epochs",
        type=int,
        default=None,
        help="Inner-loop epochs for training a temporary model on synthetic data during its generation.",
    )
    parser.add_argument(
        "--synthetic_lr",
        type=float,
        default=None,
        help="Learning rate for optimizing the synthetic data itself.",
    )

    # --- Global Model Refinement Parameter ---
    parser.add_argument(
        "--refine_epochs",
        type=int,
        default=None,
        help="Number of epochs to refine the global model on D_gt after aggregation.",
    )

    return parser


class FedTrend(Server):
    """
    Paper: https://arxiv.org/abs/2411.15716
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize FedTrend specific attributes
        self.D_ct = None  # Client-side synthetic data
        self.D_gt = None  # Global-side synthetic data

        # Trajectory banks
        # T_ct format: {client_id: {'start': model, 'end': model, 'prev_delta': {name: tensor}}}
        self.T_ct = {client.id: {} for client in self.clients}
        self.T_gt = []  # Stores global models

        # Snapshot of the global model before sending to clients
        self.model_snapshot = None

        self.initialize_loss()
        self.initialize_optimizer()
        self.initialize_scheduler()

    def variables_to_be_sent(self):
        """Override to include D_ct in the package sent to clients."""
        # Snapshot the model state before sending it
        self.model_snapshot = copy.deepcopy(self.model)

        base_vars = super().variables_to_be_sent()
        base_vars["D_ct"] = self.D_ct
        return base_vars

    def _data_construction(
        self,
        trajectories,
        data_size,
        input_shape,
        output_shape,
        synthetic_epochs,
        synthetic_inner_epochs,
        synthetic_lr,
        is_client_trajectory=False,
    ):
        """
        The core function for constructing synthetic data (D_syn) from model trajectories.
        This implements Algorithm 1, lines 29-36, using the `higher` library for
        model-agnostic differentiable inner-loop optimization.
        """
        # 1. Initialize synthetic dataset D_syn
        synthetic_x = torch.randn(
            data_size,
            *input_shape,
            requires_grad=True,
            device="cpu",
        )
        synthetic_y = torch.randn(
            data_size,
            *output_shape,
            requires_grad=True,
            device="cpu",
        )
        synthetic_data = TensorDataset(synthetic_x, synthetic_y)
        optimizer_data = torch.optim.Adam([synthetic_x, synthetic_y], lr=synthetic_lr)

        # 2. Outer loop for training D_syn
        for s_epoch in range(synthetic_epochs):
            # 3. Sample a trajectory
            if is_client_trajectory:
                client_id = np.random.choice(list(trajectories.keys()))
                trajectory = trajectories[client_id]
                model_start_state = trajectory["start"].state_dict()
                model_end_state = trajectory["end"].state_dict()
            else:
                start_idx = np.random.randint(0, len(trajectories) - 1)
                model_start_state = trajectories[start_idx].state_dict()
                model_end_state = trajectories[start_idx + 1].state_dict()

            # --- INNER LOOP ---
            temp_model = copy.deepcopy(self.model).to("cpu")
            temp_model.load_state_dict(model_start_state)

            # Use a standard optimizer, `higher` will make it differentiable
            optimizer_model = torch.optim.SGD(
                temp_model.parameters(), lr=self.learning_rate
            )

            # Create a differentiable copy of the model and optimizer
            # `track_higher_grads=False` is important for memory efficiency here, as we
            # only need gradients w.r.t. the synthetic data, not the model's own params.
            with higher.innerloop_ctx(
                temp_model, optimizer_model, track_higher_grads=False
            ) as (fmodel, diffopt):
                # fmodel is the differentiable ("functional") model
                # diffopt is the differentiable optimizer
                for _ in range(synthetic_inner_epochs):
                    for sx, sy in DataLoader(
                        synthetic_data, batch_size=self.batch_size, shuffle=True
                    ):
                        outputs = fmodel(sx)  # Use the functional model
                        loss = self.loss(outputs, sy)
                        diffopt.step(
                            loss
                        )  # Use the differentiable optimizer to take a step

                # After the inner loop, `fmodel` contains the updated parameters
                # that are still connected to the computation graph.

                # 5. Compute distance loss and update D_syn
                optimizer_data.zero_grad()
                distance_loss = 0

                mask = {}

                # --- Calculate Distance using the `fmodel` parameters ---
                # `fmodel.parameters()` gives you the updated, differentiable parameters.
                for (name_end, param_end), param_tilde in zip(
                    model_end_state.items(), fmodel.parameters()
                ):
                    if name_end not in dict(fmodel.named_parameters()):
                        continue  # Should not happen if models are identical

                    param_dist = torch.sum((param_end.to("cpu") - param_tilde) ** 2)

                    if is_client_trajectory and name_end in mask:
                        param_dist = torch.sum(
                            mask[name_end] * (param_end.to("cpu") - param_tilde) ** 2
                        )
                    distance_loss += param_dist

            # The `distance_loss` was computed using `fmodel`'s parameters, which
            # `higher` has kept attached to the graph. So this will work.
            self.logger.info(
                f"SynEpoch {s_epoch+1}/{synthetic_epochs}, Distance Loss: {distance_loss.item():.4f}"
            )
            distance_loss.backward()
            optimizer_data.step()

        return TensorDataset(synthetic_x.detach().cpu(), synthetic_y.detach().cpu())

    def _update_D_ct(self):
        # Get data shape from a client
        input_shape = (self.input_len, self.input_channels)
        output_shape = (self.output_len, self.output_channels)

        # Filter trajectories that have a start and end
        valid_trajectories = {
            cid: t for cid, t in self.T_ct.items() if "start" in t and "end" in t
        }

        self.D_ct = self._data_construction(
            trajectories=valid_trajectories,
            data_size=self.synthetic_data_size_ct,
            input_shape=input_shape,
            output_shape=output_shape,
            synthetic_epochs=self.synthetic_epochs,
            synthetic_inner_epochs=self.synthetic_inner_epochs,
            synthetic_lr=self.synthetic_lr,
            is_client_trajectory=True,
        )
        # Clear the client trajectory bank after use
        self.T_ct = {client.id: {} for client in self.clients}

    def _update_D_gt(self):
        """Handles the logic for updating the global-side synthetic data D_gt."""
        self.logger.info(f"Updating D_gt at round {self.current_iter}.")

        input_shape = (self.input_len, self.input_channels)
        output_shape = (self.output_len, self.output_channels)

        if len(self.T_gt) < 2:
            self.logger.warning(
                "Not enough global models in trajectory bank to update D_gt."
            )
            return

        self.D_gt = self._data_construction(
            trajectories=self.T_gt,
            data_size=self.synthetic_data_size_gt,
            input_shape=input_shape,
            output_shape=output_shape,
            synthetic_epochs=self.synthetic_epochs,
            synthetic_inner_epochs=self.synthetic_inner_epochs,
            synthetic_lr=self.synthetic_lr,
            is_client_trajectory=False,
        )
        # Keep only the last model for the next trajectory segment
        self.T_gt = [self.T_gt[-1]]

    def _refine_global_model(self):
        """Refines the aggregated global model by training it on D_gt."""
        if self.D_gt is None:
            return

        self.logger.info("Refining global model on D_gt...")
        dataloader = DataLoader(self.D_gt, batch_size=self.batch_size, shuffle=True)
        # Use existing server optimizer and scheduler for refinement
        for _ in range(self.refine_epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=dataloader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
            )
        self.logger.info("Global model refined.")

    def receive_from_clients(self):
        super().receive_from_clients()
        # --- FedTrend: Store Client Trajectories ---
        for client_data in self.client_data:
            client_id = client_data["id"]
            # Store the start model (which was the global model snapshot) and the end model
            self.T_ct[client_id]["start"] = self.model_snapshot
            self.T_ct[client_id]["end"] = client_data["model"]

            # Calculate and store delta for consistency check in the NEXT round
            with torch.no_grad():
                start_params = self.model_snapshot.state_dict()
                end_params = client_data["model"].state_dict()
                delta = {
                    name: end_params[name] - start_params[name] for name in end_params
                }
                self.T_ct[client_id]["prev_delta"] = delta

    def aggregate_models(self):
        super().aggregate_models()
        # --- FedTrend: Store Global Trajectory & Refine Model ---
        self.T_gt.append(copy.deepcopy(self.model.to(self.device)))
        self._refine_global_model()
        self.model.to("cpu")

        # --- FedTrend: Update Synthetic Data ---
        if self.current_iter > 0 and self.current_iter % self.L_ct == 0:
            self._update_D_ct()
        if self.current_iter > 0 and self.current_iter % self.L_gt == 0:
            self._update_D_gt()


class FedTrend_Client(Client):
    D_ct = None  # To store the received synthetic data

    def receive_from_server(self, data):
        """Override to handle the received D_ct."""
        self.D_ct = data.pop("D_ct", None)  # Safely pop D_ct
        super().receive_from_server(data)  # Handle the rest (e.g., model)

    def variables_to_be_sent(self):
        """Override to include client ID in the returned package."""
        to_be_sent = super().variables_to_be_sent()
        to_be_sent["id"] = self.id
        return to_be_sent

    def load_train_data(self, *args, **kwargs):
        """
        Override to mix local data with synthetic data D_ct.
        This implements Algorithm 1, line 26, in a more encapsulated way.
        """
        # Get the original dataloader for local data
        local_dataloader = super().load_train_data(*args, **kwargs)
        local_dataset = local_dataloader.dataset

        # Check if synthetic data D_ct is available to be mixed in
        if self.D_ct is not None and len(self.D_ct) > 0:
            # Combine the local dataset with the synthetic one
            final_dataset = ConcatDataset([local_dataset, self.D_ct])
        else:
            # If no synthetic data, just use the local dataset
            final_dataset = local_dataset

        # The 'train_samples' attribute is used by the server for aggregation weights.
        # It's crucial to update it to reflect the size of the dataset being used for training.
        self.train_samples = len(final_dataset)

        # Create and return the final DataLoader that the train() method will use
        final_dataloader = DataLoader(
            dataset=final_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Always shuffle for training
        )

        return final_dataloader
