import copy

import torch
import torch.nn.functional as F

from .base import Server

optional = {
    "first_stage_bound": 0.0,
    "cross_alpha": 0.99,
    "collaborative_model_select_strategy": 1,
}


# Argument parser update function
def args_update(parser):
    parser.add_argument("--first_stage_bound", type=float, default=None)
    parser.add_argument("--cross_alpha", type=float, default=None)
    parser.add_argument(
        "-cmss",
        "--collaborative_model_select_strategy",
        type=int,
        default=None,
        choices=[0, 1, 2],
    )


class FedCross(Server):
    """
    Paper: https://arxiv.org/abs/2210.08285
    Source: https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/servercross.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.w_locals = []
        self.w_locals_num = self.num_join_clients
        for i in range(self.w_locals_num):
            self.w_locals.append(copy.deepcopy(self.model))

    def variables_to_be_sent(self):
        return {"model": self.w_locals}

    def receive_from_clients(self):
        super().receive_from_clients()
        for w_local, client in zip(self.w_locals, self.client_data):
            self.update_model_params(old=w_local, new=client["model"])

    def aggregate_models(self):
        super().aggregate_models()

        # Calculate similarity between models
        sim_tab, sim_value = self.calculate_similarity()

        # Update global model
        if self.current_iter >= self.first_stage_bound:
            # Cross aggregation
            self.w_locals = self.cross_aggregation(self.current_iter, sim_tab)
        else:
            for i in range(len(self.w_locals)):
                for param, global_param in zip(
                    self.w_locals[i].parameters(), self.model.parameters()
                ):
                    param.data = global_param.data.clone()

    def calculate_similarity(self):
        model_num = len(self.w_locals)
        sim_tab = [[0 for _ in range(model_num)] for _ in range(model_num)]
        sum_sim = 0.0

        w_locals_dict = [model.state_dict() for model in self.w_locals]

        for k in range(model_num):
            for j in range(k):
                s = 0.0
                dict_a = torch.Tensor(0)
                dict_b = torch.Tensor(0)
                cnt = 0

                for p in w_locals_dict[k].keys():
                    a = w_locals_dict[k][p]
                    b = w_locals_dict[j][p]
                    a = a.view(-1)
                    b = b.view(-1)

                    if cnt == 0:
                        dict_a = a
                        dict_b = b
                    else:
                        dict_a = torch.cat((dict_a, a), dim=0)
                        dict_b = torch.cat((dict_b, b), dim=0)

                    if cnt % 2 == 0:
                        sub_a = a
                        sub_b = b
                    else:
                        sub_a = torch.cat((sub_a, a), dim=0)
                        sub_b = torch.cat((sub_b, b), dim=0)

                    if cnt % 2 == 1:
                        s += F.cosine_similarity(sub_a, sub_b, dim=0)
                    cnt += 1

                s += F.cosine_similarity(sub_a, sub_b, dim=0)
                sim_tab[k][j] = s
                sim_tab[j][k] = s
                sum_sim += copy.deepcopy(s)

        l = int(len(w_locals_dict[0].keys()) / 5) + 1.0
        sum_sim /= l * self.num_clients * (self.num_clients - 1) / 2.0

        return sim_tab, sum_sim

    def cross_aggregation(self, iter, sim_tab):
        w_locals_new = copy.deepcopy(self.w_locals)
        crosslist = []

        for j in range(self.w_locals_num):
            maxtag = 0
            submax = 1
            mintag = (j + 1) % self.w_locals_num

            for p in range(self.w_locals_num):
                if sim_tab[j][p] > sim_tab[j][maxtag]:
                    submax = maxtag
                    maxtag = p
                elif sim_tab[j][p] > sim_tab[j][submax]:
                    submax = p
                if sim_tab[j][p] < sim_tab[j][mintag] and p != j:
                    mintag = p

            rlist = []
            offset = iter % (self.w_locals_num - 1) + 1
            sub_list = []

            for k in range(self.w_locals_num):
                if k == j:
                    rlist.append(self.cross_alpha)
                    sub_list.append(copy.deepcopy(self.w_locals[j]))

                if self.collaborative_model_select_strategy == 0:
                    if (j + offset) % self.w_locals_num == k:
                        rlist.append(1.0 - self.cross_alpha)
                        sub_list.append(copy.deepcopy(self.w_locals[k]))
                elif self.collaborative_model_select_strategy == 1:
                    if mintag == k:
                        rlist.append(1.0 - self.cross_alpha)
                        sub_list.append(copy.deepcopy(self.w_locals[mintag]))
                elif self.collaborative_model_select_strategy == 2:
                    if maxtag == k:
                        rlist.append(1.0 - self.cross_alpha)
                        sub_list.append(copy.deepcopy(self.w_locals[maxtag]))

            # Aggregate selected models
            w_cc = self.aggregate_parameters_cross(sub_list, rlist)
            crosslist.append(w_cc)

        for k in range(self.w_locals_num):
            w_locals_new[k] = crosslist[k]

        return w_locals_new

    @staticmethod
    def aggregate_parameters_cross(models, weights):
        aggregated_model = copy.deepcopy(models[0])
        for param in aggregated_model.parameters():
            param.data.zero_()

        total_count = sum(weights)
        for w, client_model in zip(weights, models):
            for aggregated_model_param, client_param in zip(
                aggregated_model.parameters(), client_model.parameters()
            ):
                aggregated_model_param.data += (
                    client_param.data.clone() * w / total_count
                )

        return aggregated_model
