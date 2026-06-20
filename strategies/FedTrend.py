import copy
from collections import OrderedDict

import higher
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from .tFL import tFL, tFL_Client


class FedTrend(tFL):
    """Fed-TREND: Federated Time Series Forecasting with Synthetic Data (Bai et al., 2025).

    Generates two types of synthetic data on the server to address heterogeneity:
    - D_ct: derived from per-client model-update trajectory pairs; distributed to clients
      for local training augmentation to reduce cross-client heterogeneity.
    - D_gt: derived from the global model trajectory; used server-side to refine the
      aggregated global model and correct for model drift.

    Both datasets are updated every L_ct / L_gt rounds via bi-level optimization (MTT
    data condensation: Adam on synthetic (X, Y) pairs, inner SGD for trajectory replay).

    Default hyperparameters (from paper §5.4): L_ct=L_gt=10, synthetic_epochs=300,
    synthetic_lr=3e-4. Effective D_ct/D_gt sizes explored in paper: 5–40 samples.
    Reference: arXiv:2411.15716.
    """

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

        # T_ct: {client_id: {'start': params_dict, 'end': params_dict, 'mask': dict|None}}
        self.T_ct = {}

        # Start of current L_ct interval per client — initialized to W^0.
        self.T_ct_start = {
            i: copy.deepcopy(self.public_model_params)
            for i in range(self.num_clients)
        }
        self.T_ct_prev_delta = {}

        # T_gt: list of aggregated global model state dicts [W^0, W^1, ...].
        self.T_gt = []

        self.initialize_loss()
        self.initialize_optimizer()
        self.initialize_scheduler()

    # ------------------------------------------------------------------ #
    # Server → clients                                                     #
    # ------------------------------------------------------------------ #

    def package(self, client_id: int) -> dict:
        result = super().package(client_id)
        result["D_ct"] = self.D_ct
        return result

    # ------------------------------------------------------------------ #
    # Trajectory management                                                #
    # ------------------------------------------------------------------ #

    def _update_T_ct(self, packages) -> None:
        """Record one L_ct-length trajectory per client with consistency mask."""
        for cid, pkg in packages.items():
            end_params = pkg["regular_model_params"]
            start_params = self.T_ct_start[cid]

            curr_delta = {}
            with torch.no_grad():
                for name in end_params:
                    if name in start_params:
                        curr_delta[name] = (
                            end_params[name] - start_params[name]
                        ).cpu().clone()

            mask = None
            prev_delta = self.T_ct_prev_delta.get(cid)
            if prev_delta is not None:
                mask = {
                    name: (torch.sign(prev_delta[name]) == torch.sign(cd))
                    for name, cd in curr_delta.items()
                    if name in prev_delta
                }

            self.T_ct[cid] = {
                "start": start_params,
                "end": end_params,
                "mask": mask,
            }

            self.T_ct_start[cid] = copy.deepcopy(end_params)
            self.T_ct_prev_delta[cid] = curr_delta

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
        synthetic_x = torch.randn(
            data_size, *input_shape, requires_grad=True, device="cpu"
        )
        synthetic_y = torch.randn(
            data_size, *output_shape, requires_grad=True, device="cpu"
        )
        synthetic_data = TensorDataset(synthetic_x, synthetic_y)
        optimizer_data = torch.optim.Adam([synthetic_x, synthetic_y], lr=synthetic_lr)

        for s_epoch in range(synthetic_epochs):
            if is_client_trajectory:
                client_id = np.random.choice(list(trajectories.keys()))
                traj = trajectories[client_id]
                model_start_state = traj["start"]
                model_end_state = traj["end"]
                mask_dict = traj.get("mask") or {}
            else:
                start_idx = np.random.randint(0, len(trajectories) - 1)
                model_start_state = trajectories[start_idx]
                model_end_state = trajectories[start_idx + 1]
                mask_dict = {}

            temp_model = copy.deepcopy(self.model).to("cpu")
            temp_model.load_state_dict(model_start_state, strict=False)
            optimizer_model = torch.optim.SGD(
                temp_model.parameters(), lr=self.learning_rate
            )

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
                sx,
                sy,
                torch.zeros(data_size, *x_mark_shape),
                torch.zeros(data_size, *y_mark_shape),
            )
        return TensorDataset(sx, sy)

    def _get_mark_shapes(self):
        sample = next(iter(self.trainer.worker.load_train_data()))
        return tuple(sample[2].shape[1:]), tuple(sample[3].shape[1:])

    def _update_D_ct(self, packages) -> None:
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
        self.T_ct = {}

    def _update_D_gt(self) -> None:
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

    def _refine_global_model(self) -> None:
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
    # FL aggregation hook                                                  #
    # ------------------------------------------------------------------ #

    def aggregate_client_updates(self, packages) -> None:
        # Standard FedAvg aggregation
        super().aggregate_client_updates(packages)

        # Store aggregated global model state in T_gt, then refine on D_gt.
        self.T_gt.append(copy.deepcopy(self.public_model_params))
        self._refine_global_model()
        # Commit refined model after refinement
        self._commit_global(
            OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )

        # Update T_ct and D_ct at L_ct boundaries (t > 0).
        if self.current_iter > 0 and self.current_iter % self.L_ct == 0:
            self._update_T_ct(packages)
            self._update_D_ct(packages)

        # Update D_gt at L_gt boundaries (t > 0).
        if self.current_iter > 0 and self.current_iter % self.L_gt == 0:
            self._update_D_gt()


class FedTrend_Client(tFL_Client):
    D_ct = None

    def set_parameters(self, package: dict) -> None:
        self.D_ct = package.pop("D_ct", None)
        super().set_parameters(package)

    def load_train_data(self, *args, **kwargs):
        local_dataloader = super().load_train_data(*args, **kwargs)
        local_dataset = local_dataloader.dataset

        if self.D_ct is not None and len(self.D_ct) > 0:
            final_dataset = ConcatDataset([local_dataset, self.D_ct])
        else:
            final_dataset = local_dataset

        self.train_samples = len(final_dataset)
        return DataLoader(
            dataset=final_dataset, batch_size=self.batch_size, shuffle=True
        )
