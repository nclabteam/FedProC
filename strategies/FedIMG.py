import copy

import torch
import torch.nn.functional as F

from .base import Client, Server

optional = {
    "multi_granularity_scale": [2],
    "alpha": 0.1,
}

compulsory = {
    "save_local_model": True,
}


def args_update(parser):
    parser.add_argument(
        "--multi_granularity_scale",
        type=int,
        default=None,
        action="append",
        help="Scale factor for multi-granularity time series.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
    )


class FedIMG(Server):
    pass


class FedIMG_Client(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 1 not in self.configs.multi_granularity_scale:
            self.configs.multi_granularity_scale.append(1)
        self.configs.multi_granularity_scale.sort()

    @staticmethod
    def _xicor_1d(x: torch.Tensor, y: torch.Tensor, ties: bool = True):
        """
        Calculates Xicor correlation coefficient for two 1D tensors.
        This is the exact function you provided.
        """
        # Ensure inputs are float for calculation
        x = x.float()
        y = y.float()
        n = x.size(0)

        # The argsort logic is slightly different from your original, but more standard in PyTorch
        order = torch.argsort(x)
        y_ordered = y[order]

        if ties:
            # Vectorized rank calculation
            l = (y_ordered.unsqueeze(0) >= y_ordered.unsqueeze(1)).sum(dim=1).float()
            r = l.clone()

            # Tie-breaking logic from your function
            unique_r, counts = torch.unique(r, return_counts=True)
            tied_values = unique_r[counts > 1]
            for val in tied_values:
                tie_indices = (r == val).nonzero(as_tuple=True)[0]
                count = len(tie_indices)
                new_vals = val - torch.arange(count, device=r.device, dtype=r.dtype)
                perm = torch.randperm(count, device=r.device)
                r[tie_indices] = new_vals[perm]

            numerator = n * torch.sum(torch.abs(r[1:] - r[:-1]))
            denominator = 2 * torch.sum(l * (n - l))

            if denominator == 0:
                return 0.0
            # Return a tensor, not a float, to keep it on the device
            return 1.0 - (numerator / denominator)
        else:
            # The simplified version without tie handling
            r = (y.unsqueeze(0) >= y_ordered.unsqueeze(1)).sum(dim=1).float()
            numerator = 3 * torch.sum(torch.abs(r[1:] - r[:-1]))
            denominator = float(n**2 - 1)
            if denominator == 0:
                return 0.0
            return 1.0 - (numerator / denominator)

    @torch.no_grad()
    def _get_xicor_infonce_value(self, z_local, z_global, temperature=0.1):
        """
        FORWARD FACE: Calculates the loss value by building a similarity matrix
        using the provided _xicor_1d method.
        """
        batch_size = z_local.size(0)
        sim_matrix = torch.zeros(batch_size, batch_size, device=self.device)

        # Loop to build the all-pairs similarity matrix
        for i in range(batch_size):
            for j in range(batch_size):
                # Use your provided function for each pair
                sim_matrix[i, j] = self._xicor_1d(z_local[i], z_global[j])

        sim_matrix /= temperature
        labels = torch.arange(batch_size, device=self.device)
        return F.cross_entropy(sim_matrix, labels)

    def receive_from_server(self, data):
        global_model = copy.deepcopy(data["model"])
        self.model.to(self.device)
        global_model.to(self.device)
        self.model.train()
        global_model.eval()

        dataloader = self.load_train_data(sample_ratio=0.3, shuffle=True)
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            self.optimizer.zero_grad()

            local_output = self.model(batch_x)
            with torch.no_grad():
                global_output = global_model(batch_x)

            sup_loss = self.loss(local_output, batch_y)

            total_contrastive_loss_value = 0.0
            total_weight = 0

            for scale in self.multi_granularity_scale:
                val = int(self.output_len / scale)
                if val == 0:
                    continue

                l_output_g = local_output[:, :val, :]
                g_output_g = global_output[:, :val, :]

                # The pooling step converts the time-series predictions into vectors
                z_local = F.max_pool1d(
                    l_output_g.transpose(1, 2), kernel_size=val
                ).squeeze(2)
                z_global = F.max_pool1d(
                    g_output_g.transpose(1, 2), kernel_size=val
                ).squeeze(2)

                # Calculate the loss VALUE using your Xicor function
                c_loss_val = self.loss(z_local, z_global)
                # c_loss_val = self._get_xicor_infonce_value(z_local, z_global)

                weight = val
                total_contrastive_loss_value += weight * c_loss_val
                total_weight += weight

            norm_con_loss_val = (
                total_contrastive_loss_value / total_weight if total_weight > 0 else 0
            )

            # This is the loss that will be used for the actual backpropagation
            loss_for_backward_pass = (
                1 - self.configs.alpha
            ) * sup_loss + self.configs.alpha * norm_con_loss_val

            loss_for_backward_pass.backward()
            self.optimizer.step()

        self.model.to("cpu")
