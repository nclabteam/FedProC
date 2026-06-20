import copy

import numpy as np
import torch

from .pFL import pFL, pFL_Client


class FedCAC(pFL):
    optional = {"tau": 0.5, "beta": 170}

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--tau", type=float, default=None)
        parser.add_argument("--beta", type=int, default=None)

    def aggregate_client_updates(self, packages) -> None:
        # Standard FedAvg of regular_model_params
        super().aggregate_client_updates(packages)

        cids = list(packages.keys())

        # Skip customization if any client lacks critical_parameter (first round)
        for cid in cids:
            if "critical_parameter" not in self.clients_personal_model_params[cid]:
                return

        num_clients = len(cids)
        overlap_buffer = [[] for _ in range(num_clients)]

        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    continue
                cp_i = self.clients_personal_model_params[cids[i]]["critical_parameter"].to(self.device)
                cp_j = self.clients_personal_model_params[cids[j]]["critical_parameter"].to(self.device)
                overlap_rate = 1 - torch.sum(torch.abs(cp_i - cp_j)) / float(
                    torch.sum(cp_i).cpu() * 2
                )
                overlap_buffer[i].append(overlap_rate.item())

        overlap_buffer_tensor = torch.tensor(overlap_buffer)
        overlap_sum = overlap_buffer_tensor.sum()
        overlap_avg = overlap_sum / ((num_clients - 1) * num_clients)
        overlap_max = overlap_buffer_tensor.max()
        threshold = overlap_avg + (self.current_iter + 1) / self.beta * (
            overlap_max - overlap_avg
        )

        for i, cid in enumerate(cids):
            w_customized = copy.deepcopy(packages[cid]["regular_model_params"])
            collaboration_cids = [cid]
            index = 0
            for j in range(num_clients):
                if i == j:
                    continue
                if overlap_buffer[i][index] >= threshold:
                    collaboration_cids.append(cids[j])
                index += 1

            for name in w_customized:
                for collab_cid in collaboration_cids:
                    if collab_cid == cid:
                        continue
                    w_customized[name] = (
                        w_customized[name]
                        + packages[collab_cid]["regular_model_params"][name]
                    )
                w_customized[name] = torch.div(
                    w_customized[name], float(len(collaboration_cids))
                )

            self.clients_personal_model_params[cid]["customized_model_state"] = w_customized


class FedCAC_Client(pFL_Client):
    def set_parameters(self, package: dict) -> None:
        self.id = package["client_id"]
        self.current_iter = package["current_iter"]
        self._load_private(self.id)
        self.model.load_state_dict(package["regular_model_params"])

        personal = package["personal_model_params"]
        if personal:
            if personal.get("local_mask") is not None and "customized_model_state" in personal:
                # Replicate original FedCAC behavior: load customized model
                self.model.load_state_dict(personal["customized_model_state"])
            else:
                self.model.load_state_dict(personal, strict=False)

        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
            self._move_optimizer_state_to_param_devices(self.optimizer)
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)
        if package["scheduler_state"]:
            self.scheduler.load_state_dict(package["scheduler_state"])
        else:
            self.scheduler.load_state_dict(self.init_scheduler_state)

    def fit(self) -> None:
        prev_params = [p.data.clone() for p in self.model.parameters()]
        super().fit()
        cp, gm, lm = self._evaluate_critical_parameter(
            prev_params, list(self.model.parameters())
        )
        self._critical_parameter = cp
        self._global_mask = gm
        self._local_mask = lm

    def package(self, train_time: float) -> dict:
        result = super().package(train_time)
        result["personal_model_params"]["critical_parameter"] = self._critical_parameter
        result["personal_model_params"]["local_mask"] = self._local_mask
        result["personal_model_params"]["global_mask"] = self._global_mask
        return result

    def _evaluate_critical_parameter(self, prev_params, curr_params):
        device = curr_params[0].device
        global_mask = []
        local_mask = []
        critical_parameter = []

        for prevparam, param in zip(prev_params, curr_params):
            prevparam = prevparam.to(device)
            g = param.data - prevparam
            v = param.data
            c = torch.abs(g * v)

            metric = c.view(-1)
            num_params = metric.size(0)
            nz = int(self.tau * num_params)
            top_values, _ = torch.topk(metric, nz)
            thresh = top_values[-1] if len(top_values) > 0 else np.inf
            if thresh <= 1e-10:
                new_metric = metric[metric > 1e-20]
                if len(new_metric) > 0:
                    thresh = new_metric.sort()[0][0]

            mask = (c >= thresh).int().to("cpu")
            global_mask.append((c < thresh).int().to("cpu"))
            local_mask.append(mask)
            critical_parameter.append(mask.view(-1))

        for param in curr_params:
            param.grad = None
        critical_parameter = torch.cat(critical_parameter)

        return critical_parameter, global_mask, local_mask
