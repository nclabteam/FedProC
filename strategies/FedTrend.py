import copy

import higher
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from .tFL import tFL, tFL_Client


class FedTrend(tFL):

    optional = {
        "L_ct": 10,
        "synthetic_data_size_ct": 100,
        "L_gt": 10,
        "synthetic_data_size_gt": 100,
        "synthetic_epochs": 300,
        "synthetic_inner_epochs": 1,
        "synthetic_lr": 3e-4,
        "refine_epochs": 1,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--L_ct", type=int, default=None)
        parser.add_argument("--synthetic_data_size_ct", type=int, default=None)
        parser.add_argument("--L_gt", type=int, default=None)
        parser.add_argument("--synthetic_data_size_gt", type=int, default=None)
        parser.add_argument("--synthetic_epochs", type=int, default=None)
        parser.add_argument("--synthetic_inner_epochs", type=int, default=None)
        parser.add_argument("--synthetic_lr", type=float, default=None)
        parser.add_argument("--refine_epochs", type=int, default=None)
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.D_ct = None
        self.D_gt = None

        # T_ct: {client_id: {'start': model, 'end': model, 'mask': {name: BoolTensor}|None}}
        # Populated at each L_ct boundary; cleared after D_ct is built.
        self.T_ct = {}

        # Start of the current L_ct interval per client — initialized to W^0.
        self.T_ct_start = {
            client.id: copy.deepcopy(self.model) for client in self.clients
        }
        # Delta of the previous L_ct interval per client (for consistency masking).
        self.T_ct_prev_delta = {}

        # T_gt: list of aggregated global models [W^0, W^1, ...].
        self.T_gt = []

        self.initialize_loss()
        self.initialize_optimizer()
        self.initialize_scheduler()

    # ------------------------------------------------------------------ #
    # Server → clients                                                     #
    # ------------------------------------------------------------------ #

    def variables_to_be_sent(self):
        base_vars = super().variables_to_be_sent()
        base_vars["D_ct"] = self.D_ct
        return base_vars

    # ------------------------------------------------------------------ #
    # Trajectory management                                                #
    # ------------------------------------------------------------------ #

    def _update_T_ct(self):
        """Record one L_ct-length trajectory per client with consistency mask.

        Called at round t when t % L_ct == 0 (after receiving client models).
        Per paper §3.2:
          - start  = W_ci^{k*L_ct}  (stored from previous boundary)
          - end    = W_ci^{(k+1)*L_ct}  (current client model)
          - mask   = {name: sign(prev_delta)==sign(curr_delta)} (zero out
                     params whose sign flipped vs the prior interval)
        """
        for client_data in self.client_data:
            client_id = client_data["id"]
            end_model = client_data["model"]
            start_model = self.T_ct_start[client_id]

            # Current interval delta: W_end - W_start
            curr_delta = {}
            with torch.no_grad():
                end_params = dict(end_model.named_parameters())
                start_params = dict(start_model.named_parameters())
                for name in end_params:
                    if name in start_params:
                        curr_delta[name] = (
                            end_params[name].data - start_params[name].data
                        ).cpu().clone()

            # Consistency mask: keep only params whose sign matches the previous interval.
            # First interval has no prior delta → no masking (full distance used).
            mask = None
            prev_delta = self.T_ct_prev_delta.get(client_id)
            if prev_delta is not None:
                mask = {
                    name: (torch.sign(prev_delta[name]) == torch.sign(cd))
                    for name, cd in curr_delta.items()
                    if name in prev_delta
                }

            self.T_ct[client_id] = {"start": start_model, "end": end_model, "mask": mask}

            # Advance interval boundary.
            self.T_ct_start[client_id] = copy.deepcopy(end_model)
            self.T_ct_prev_delta[client_id] = curr_delta

    # ------------------------------------------------------------------ #
    # Synthetic data construction (Algorithm 1, lines 29-36)              #
    # ------------------------------------------------------------------ #

    def _data_construction(
        self,
        trajectories,
        data_size,
        input_shape,
        output_shape,
        synthetic_epochs,
        synthetic_inner_epochs,
        synthetic_lr,
        x_mark_shape=None,
        y_mark_shape=None,
        is_client_trajectory=False,
    ):
        synthetic_x = torch.randn(data_size, *input_shape, requires_grad=True, device="cpu")
        synthetic_y = torch.randn(data_size, *output_shape, requires_grad=True, device="cpu")
        synthetic_data = TensorDataset(synthetic_x, synthetic_y)
        optimizer_data = torch.optim.Adam([synthetic_x, synthetic_y], lr=synthetic_lr)

        for s_epoch in range(synthetic_epochs):
            # Sample a trajectory segment.
            if is_client_trajectory:
                client_id = np.random.choice(list(trajectories.keys()))
                traj = trajectories[client_id]
                model_start_state = traj["start"].state_dict()
                model_end_state = traj["end"].state_dict()
                mask_dict = traj.get("mask") or {}
            else:
                start_idx = np.random.randint(0, len(trajectories) - 1)
                model_start_state = trajectories[start_idx].state_dict()
                model_end_state = trajectories[start_idx + 1].state_dict()
                mask_dict = {}

            # Inner loop: train a temporary model on D_syn from W_start.
            temp_model = copy.deepcopy(self.model).to("cpu")
            temp_model.load_state_dict(model_start_state)
            optimizer_model = torch.optim.SGD(temp_model.parameters(), lr=self.learning_rate)

            with higher.innerloop_ctx(
                temp_model, optimizer_model, track_higher_grads=False
            ) as (fmodel, diffopt):
                for _ in range(synthetic_inner_epochs):
                    for batch in DataLoader(
                        synthetic_data, batch_size=self.batch_size, shuffle=True
                    ):
                        sx, sy = batch[0], batch[1]
                        x_mark = batch[2] if len(batch) > 2 else None
                        y_mark = batch[3] if len(batch) > 3 else None
                        outputs = fmodel(sx, x_mark=x_mark, y_mark=y_mark)
                        loss = self.loss(outputs, sy)
                        diffopt.step(loss)

                # Outer-loop: distance loss between fmodel and W_end.
                # For D_ct: zero out params that updated inconsistently (mask_dict).
                optimizer_data.zero_grad()
                distance_loss = 0
                named_end = dict(model_end_state)
                for (name, param_end), param_tilde in zip(
                    named_end.items(), fmodel.parameters()
                ):
                    if name not in dict(fmodel.named_parameters()):
                        continue
                    diff = (param_end.to("cpu") - param_tilde) ** 2
                    if name in mask_dict:
                        # mask: True (1) = consistent, False (0) = masked out
                        diff = diff * mask_dict[name].float().to("cpu")
                    distance_loss += torch.sum(diff)

            self.logger.info(
                f"SynEpoch {s_epoch+1}/{synthetic_epochs}, "
                f"Distance Loss: {distance_loss.item():.4f}"
            )
            distance_loss.backward()
            optimizer_data.step()

        sx = synthetic_x.detach().cpu()
        sy = synthetic_y.detach().cpu()
        if x_mark_shape is not None and y_mark_shape is not None:
            return TensorDataset(
                sx, sy,
                torch.zeros(data_size, *x_mark_shape),
                torch.zeros(data_size, *y_mark_shape),
            )
        return TensorDataset(sx, sy)

    def _get_mark_shapes(self):
        sample = next(iter(self.clients[0].load_train_data()))
        return tuple(sample[2].shape[1:]), tuple(sample[3].shape[1:])

    def _update_D_ct(self):
        input_shape = (self.input_len, self.input_channels)
        output_shape = (self.output_len, self.output_channels)
        x_mark_shape, y_mark_shape = self._get_mark_shapes()

        valid = {cid: t for cid, t in self.T_ct.items() if t}
        if not valid:
            return

        self.D_ct = self._data_construction(
            trajectories=valid,
            data_size=self.synthetic_data_size_ct,
            input_shape=input_shape,
            output_shape=output_shape,
            synthetic_epochs=self.synthetic_epochs,
            synthetic_inner_epochs=self.synthetic_inner_epochs,
            synthetic_lr=self.synthetic_lr,
            x_mark_shape=x_mark_shape,
            y_mark_shape=y_mark_shape,
            is_client_trajectory=True,
        )
        # T_ct_start already updated in _update_T_ct; clear the used trajectories.
        self.T_ct = {}

    def _update_D_gt(self):
        self.logger.info(f"Updating D_gt at round {self.current_iter}.")
        if len(self.T_gt) < 2:
            self.logger.warning("Not enough global models in T_gt to update D_gt.")
            return

        input_shape = (self.input_len, self.input_channels)
        output_shape = (self.output_len, self.output_channels)
        x_mark_shape, y_mark_shape = self._get_mark_shapes()

        self.D_gt = self._data_construction(
            trajectories=self.T_gt,
            data_size=self.synthetic_data_size_gt,
            input_shape=input_shape,
            output_shape=output_shape,
            synthetic_epochs=self.synthetic_epochs,
            synthetic_inner_epochs=self.synthetic_inner_epochs,
            synthetic_lr=self.synthetic_lr,
            x_mark_shape=x_mark_shape,
            y_mark_shape=y_mark_shape,
            is_client_trajectory=False,
        )
        self.T_gt = [self.T_gt[-1]]

    def _refine_global_model(self):
        if self.D_gt is None:
            return
        self.logger.info("Refining global model on D_gt...")
        dataloader = DataLoader(self.D_gt, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.refine_epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=dataloader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
            )

    # ------------------------------------------------------------------ #
    # FL hooks                                                             #
    # ------------------------------------------------------------------ #

    def aggregate_models(self):
        super().aggregate_models()

        # Store aggregated global model in T_gt, then refine on D_gt.
        self.T_gt.append(copy.deepcopy(self.model.to(self.device)))
        self._refine_global_model()
        self.model.to("cpu")

        # Update T_ct and D_ct at L_ct boundaries (t > 0).
        if self.current_iter > 0 and self.current_iter % self.L_ct == 0:
            self._update_T_ct()
            self._update_D_ct()

        # Update D_gt at L_gt boundaries (t > 0).
        if self.current_iter > 0 and self.current_iter % self.L_gt == 0:
            self._update_D_gt()


class FedTrend_Client(tFL_Client):
    D_ct = None

    def receive_from_server(self, data):
        self.D_ct = data.pop("D_ct", None)
        super().receive_from_server(data)

    def variables_to_be_sent(self):
        to_be_sent = super().variables_to_be_sent()
        to_be_sent["id"] = self.id
        return to_be_sent

    def load_train_data(self, *args, **kwargs):
        local_dataloader = super().load_train_data(*args, **kwargs)
        local_dataset = local_dataloader.dataset

        if self.D_ct is not None and len(self.D_ct) > 0:
            final_dataset = ConcatDataset([local_dataset, self.D_ct])
        else:
            final_dataset = local_dataset

        self.train_samples = len(final_dataset)
        return DataLoader(dataset=final_dataset, batch_size=self.batch_size, shuffle=True)
