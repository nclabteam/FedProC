import copy

import numpy as np
import torch

from .pFL import pFL, pFL_Client


class FedCAC(pFL):
    """FedCAC: Federated Learning with Critical-parameter-Aware Collaboration (Wu et al., ICCV 2023).

    Identifies per-client critical parameters (sensitivity-based) each round,
    then selects each client's collaboration set based on overlap similarity
    of critical parameter masks. Non-critical parameters use FedAvg; critical
    parameters are averaged only among collaborating clients.

    Collaboration threshold increases over rounds (controlled by β):
      Ω^t = avg(O) + (t+1)/β * (max(O) - avg(O))
    where O_{i,j} = 1 - ||M_i - M_j||_1 / (2n) (1 = identical, 0 = disjoint).

    Default τ=0.5 (critical-param fraction), β=170 (threshold growth rate).
    Reference: arXiv:2309.11103. ICCV 2023.
    """

    optional = {"tau": 0.5, "beta": 170}

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--tau", type=float, default=None)
        parser.add_argument("--beta", type=int, default=None)

    def __init__(self, configs, times):
        super().__init__(configs=configs, times=times)
        total_numel = sum(p.numel() for p in self.model.parameters())
        cp_init = torch.zeros(total_numel, dtype=torch.int32)
        mask_init = [
            torch.zeros(p.shape, dtype=torch.int32)
            for p in self.model.parameters()
        ]
        global_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid].update({
                "critical_parameter": cp_init.clone(),
                "local_mask": [m.clone() for m in mask_init],
                "global_mask": [m.clone() for m in mask_init],
                "customized_model_state": {k: v.clone() for k, v in global_model_state.items()},
            })

    def aggregate_client_updates(self, packages) -> None:
        # Standard FedAvg of regular_model_params
        super().aggregate_client_updates(packages)

        cids = list(packages.keys())
        num_clients = len(cids)
        overlap_buffer = [[] for _ in range(num_clients)]
        total_params = float(
            self.clients_personal_model_params[cids[0]]["critical_parameter"].numel()
        )

        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    continue
                cp_i = self.clients_personal_model_params[cids[i]]["critical_parameter"].to(self.device)
                cp_j = self.clients_personal_model_params[cids[j]]["critical_parameter"].to(self.device)
                overlap_rate = 1.0 - torch.sum(torch.abs(cp_i - cp_j)).item() / (total_params * 2.0)
                overlap_buffer[i].append(overlap_rate)

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

            # Paper Alg. 1 step 14: ŵ_i = w̄ ⊙ (J-M_i) + u_i ⊙ M_i
            # critical params (mask=1): collab avg; non-critical (mask=0): FedAvg
            local_mask_list = self.clients_personal_model_params[cid]["local_mask"]
            param_names = [name for name, _ in self.model.named_parameters()]
            for idx, pname in enumerate(param_names):
                if pname not in w_customized:
                    continue
                m = local_mask_list[idx].float()
                w_customized[pname] = (
                    self.public_model_params[pname].cpu() * (1.0 - m)
                    + w_customized[pname].cpu() * m
                )
            self.clients_personal_model_params[cid]["customized_model_state"] = w_customized


class FedCAC_Client(pFL_Client):
    def set_parameters(self, package: dict) -> None:
        self.id = package["client_id"]
        self.current_iter = package["current_iter"]
        self._load_private(self.id)
        self.model.load_state_dict(
            package["personal_model_params"]["customized_model_state"]
        )
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
