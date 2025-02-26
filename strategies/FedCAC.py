import copy

import numpy as np
import torch

from .base import Client, Server

optional = {"tau": 0.5, "beta": 170}


# Argument parser update function
def args_update(parser):
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--beta", type=int, default=None)


class FedCAC(Server):
    def get_customized_global_models(self):
        num_clients = len(self.client_data)
        overlap_buffer = [[] for i in range(num_clients)]

        # calculate overlap rate between client i and client j in the selected clients
        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    continue
                overlap_rate = 1 - torch.sum(
                    torch.abs(
                        self.client_data[i]["critical_parameter"].to(self.device)
                        - self.client_data[j]["critical_parameter"].to(self.device)
                    )
                ) / float(
                    torch.sum(
                        self.client_data[i]["critical_parameter"].to(self.device)
                    ).cpu()
                    * 2
                )
                overlap_buffer[i].append(overlap_rate)

        # calculate the global threshold
        overlap_buffer_tensor = torch.tensor(overlap_buffer)
        overlap_sum = overlap_buffer_tensor.sum()
        overlap_avg = overlap_sum / ((num_clients - 1) * num_clients)
        overlap_max = overlap_buffer_tensor.max()
        threshold = overlap_avg + (self.current_iter + 1) / self.beta * (
            overlap_max - overlap_avg
        )

        # calculate the customized global model for each client
        for i in range(num_clients):
            w_customized_global = copy.deepcopy(
                self.client_data[i]["model"].state_dict()
            )
            collaboration_clients = [i]
            # find clients whose critical parameter locations are similar to client i
            index = 0
            for j in range(num_clients):
                if i == j:
                    continue
                if overlap_buffer[i][index] >= threshold:
                    collaboration_clients.append(j)
                index += 1

            for key in w_customized_global.keys():
                for client in collaboration_clients:
                    if client == i:
                        continue
                    w_customized_global[key] += self.client_data[client][
                        "model"
                    ].state_dict()[key]
                w_customized_global[key] = torch.div(
                    w_customized_global[key], float(len(collaboration_clients))
                )
            # send the customized global model to client i
            self.client_data[i]["customized_model"] = copy.deepcopy(self.model)
            self.client_data[i]["customized_model"].load_state_dict(w_customized_global)

    def variables_to_be_sent(self):
        if self.current_iter == 0:
            return super().variables_to_be_sent()
        self.get_customized_global_models()
        customized_models = []
        for client in self.clients:
            if self.client_data[client.id].get("customized_model") is not None:
                customized_models.append(
                    self.client_data[client.id]["customized_model"]
                )
            else:
                customized_models.append(None)
        return {**super().variables_to_be_sent(), "customized_model": customized_models}


class FedCAC_Client(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critical_parameter = (
            None  # record the critical parameter positions in FedCAC
        )
        self.customized_model = copy.deepcopy(self.model)  # customized global model
        self.critical_parameter = None
        self.global_mask = None
        self.local_mask = None

    def train(self):
        self.snapshot = copy.deepcopy(self.model)
        super().train()
        (
            self.critical_parameter,
            self.global_mask,
            self.local_mask,
        ) = self.evaluate_critical_parameter(
            prevModel=self.snapshot, model=self.model, tau=self.tau
        )

    def evaluate_critical_parameter(self, prevModel, model, tau):
        r"""
        Overview:
            Implement critical parameter selection.
        """
        global_mask = []  # mark non-critical parameter
        local_mask = []  # mark critical parameter
        critical_parameter = []

        # select critical parameters in each layer
        for prevparam, param in zip(prevModel.parameters(), model.parameters()):
            g = param.data - prevparam.data
            v = param.data
            c = torch.abs(g * v)

            metric = c.view(-1)
            num_params = metric.size(0)
            nz = int(tau * num_params)
            top_values, _ = torch.topk(metric, nz)
            thresh = top_values[-1] if len(top_values) > 0 else np.inf
            # if threshold equals 0, select minimal nonzero element as threshold
            if thresh <= 1e-10:
                new_metric = metric[metric > 1e-20]
                if len(new_metric) == 0:  # this means all items in metric are zero
                    print(f"Abnormal!!! metric:{metric}")
                else:
                    thresh = new_metric.sort()[0][0]

            # Get the local mask and global mask
            mask = (c >= thresh).int().to("cpu")
            global_mask.append((c < thresh).int().to("cpu"))
            local_mask.append(mask)
            critical_parameter.append(mask.view(-1))
        model.zero_grad()
        critical_parameter = torch.cat(critical_parameter)

        return critical_parameter, global_mask, local_mask

    def receive_from_server(self, data):
        if self.local_mask is not None:
            self.customized_model = data["customized_model"]
            index = 0
            for param1, param2, param3 in zip(
                self.model.parameters(),
                self.customized_model.parameters(),
                self.customized_model.parameters(),
            ):
                param1.data = (
                    self.local_mask[index].to(self.device).float() * param3.data
                    + self.global_mask[index].to(self.device).float() * param2.data
                )
                index += 1

        else:
            self.update_model_params(data["model"])

    def variables_to_be_sent(self):
        return {
            **super().variables_to_be_sent(),
            "critical_parameter": self.critical_parameter,
            "id": self.id,
        }
