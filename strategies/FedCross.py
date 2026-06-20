import copy
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn.functional as F

from .tFL import tFL, tFL_Client


class FedCross(tFL):

    optional = {
        "first_stage_bound": 0.0,
        "cross_alpha": 0.99,
        "collaborative_model_select_strategy": 1,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--first_stage_bound", type=float, default=None)
        parser.add_argument("--cross_alpha", type=float, default=None)
        parser.add_argument(
            "-cmss",
            "--collaborative_model_select_strategy",
            type=int,
            default=None,
            choices=[0, 1, 2],
        )

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id)
        # Send per-client w_local if available, otherwise global (first round)
        personal = self.clients_personal_model_params.get(client_id, {})
        if personal:
            pkg["regular_model_params"] = copy.deepcopy(personal)
            pkg["personal_model_params"] = {}
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        cids = list(packages.keys())
        scores = [packages[cid]["score"] for cid in cids]
        total = float(sum(scores))
        w_locals = [dict(packages[cid]["regular_model_params"]) for cid in cids]

        # Store trained models as per-client local models
        for cid, w in zip(cids, w_locals):
            self.clients_personal_model_params[cid].update(w)

        # Global FedAvg
        param_names = list(self.public_model_params.keys())
        new_global = OrderedDict()
        weights_t = torch.tensor([s / total for s in scores], dtype=torch.float32)
        for name in param_names:
            stacked = torch.stack([w[name].float() for w in w_locals], dim=-1)
            new_global[name] = torch.sum(stacked * weights_t, dim=-1).to(
                self.public_model_params[name].dtype
            )
        self._commit_global(new_global)

        # Cross-aggregation between w_locals
        if len(w_locals) >= 2:
            if self.current_iter >= self.first_stage_bound:
                sim_tab, _ = self._calculate_similarity(w_locals)
                new_w_locals = self._cross_aggregation(w_locals, sim_tab)
                for cid, new_w in zip(cids, new_w_locals):
                    self.clients_personal_model_params[cid].update(new_w)
            else:
                # Early rounds: sync all local models to global
                for cid in cids:
                    self.clients_personal_model_params[cid] = dict(self.public_model_params)

    def _calculate_similarity(self, w_locals: List[Dict[str, torch.Tensor]]):
        n = len(w_locals)
        sim_tab = [[0.0] * n for _ in range(n)]
        sum_sim = 0.0

        for k in range(n):
            for j in range(k):
                s = 0.0
                dict_a = torch.Tensor(0)
                dict_b = torch.Tensor(0)
                cnt = 0
                sub_a = sub_b = None

                for p in w_locals[k]:
                    a = w_locals[k][p].float().view(-1)
                    b = w_locals[j][p].float().view(-1)

                    dict_a = a if cnt == 0 else torch.cat((dict_a, a), dim=0)
                    dict_b = b if cnt == 0 else torch.cat((dict_b, b), dim=0)

                    if cnt % 2 == 0:
                        sub_a = a
                        sub_b = b
                    else:
                        sub_a = torch.cat((sub_a, a), dim=0)
                        sub_b = torch.cat((sub_b, b), dim=0)

                    if cnt % 2 == 1:
                        s += F.cosine_similarity(sub_a, sub_b, dim=0).item()
                    cnt += 1

                if sub_a is not None and sub_b is not None:
                    s += F.cosine_similarity(sub_a, sub_b, dim=0).item()

                sim_tab[k][j] = s
                sim_tab[j][k] = s
                sum_sim += s

        l = int(len(w_locals[0]) / 5) + 1.0
        denom = l * self.num_clients * (self.num_clients - 1) / 2.0
        sum_sim /= denom if denom > 0 else 1.0

        return sim_tab, sum_sim

    def _cross_aggregation(
        self,
        w_locals: List[Dict[str, torch.Tensor]],
        sim_tab: List[List[float]],
    ) -> List[Dict[str, torch.Tensor]]:
        n = len(w_locals)
        crosslist = []

        for j in range(n):
            maxtag = 0
            submax = 1
            mintag = (j + 1) % n

            for p in range(n):
                if sim_tab[j][p] > sim_tab[j][maxtag]:
                    submax = maxtag
                    maxtag = p
                elif sim_tab[j][p] > sim_tab[j][submax]:
                    submax = p
                if p != j and sim_tab[j][p] < sim_tab[j][mintag]:
                    mintag = p

            rlist = []
            sub_list = []
            offset = int(self.current_iter) % (n - 1) + 1

            for k in range(n):
                if k == j:
                    rlist.append(self.cross_alpha)
                    sub_list.append(copy.deepcopy(w_locals[j]))

                if self.collaborative_model_select_strategy == 0:
                    if (j + offset) % n == k:
                        rlist.append(1.0 - self.cross_alpha)
                        sub_list.append(copy.deepcopy(w_locals[k]))
                elif self.collaborative_model_select_strategy == 1:
                    if mintag == k:
                        rlist.append(1.0 - self.cross_alpha)
                        sub_list.append(copy.deepcopy(w_locals[mintag]))
                elif self.collaborative_model_select_strategy == 2:
                    if maxtag == k:
                        rlist.append(1.0 - self.cross_alpha)
                        sub_list.append(copy.deepcopy(w_locals[maxtag]))

            crosslist.append(self._aggregate_parameters_cross(sub_list, rlist))

        return crosslist

    @staticmethod
    def _aggregate_parameters_cross(
        state_dicts: List[Dict[str, torch.Tensor]],
        weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        result = {name: torch.zeros_like(p.float()) for name, p in state_dicts[0].items()}
        total = sum(weights)
        for w, sd in zip(weights, state_dicts):
            for name in result:
                result[name] += sd[name].float() * (w / total)
        return {
            name: result[name].to(state_dicts[0][name].dtype)
            for name in result
        }


class FedCross_Client(tFL_Client):
    pass
