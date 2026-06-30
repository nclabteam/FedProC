# -*- coding: utf-8 -*-
"""FedRTS - Federated Robust Pruning via Combinatorial Thompson Sampling.

Paper: https://arxiv.org/abs/2501.19122  |  NeurIPS '25
Ref:   https://github.com/FedPruning/FedPruning/tree/main/api/distributed/fedrts/

Server maintains per-weight Beta(alpha, beta) distributions.
Each adj round: clients vote on which active weights are "core" (top-kappa by magnitude)
  and which inactive weights have highest gradient -> server updates Beta params and
  re-samples a new mask via Thompson Sampling.
"""
from typing import Any, Dict

import torch

from .spFL import spFL, spFL_Client


class FedRTS(spFL):
    """FedRTS server - Thompson Sampling mask via per-weight Beta distributions."""

    optional = {
        **spFL.optional,
        "aggregated_gamma": 0.5,
        "ts_ratio": 0.1,
    }

    def __init__(self, configs, times):
        super().__init__(configs, times)
        self._ts_alpha: Dict[str, torch.Tensor] = {}
        self._ts_beta: Dict[str, torch.Tensor] = {}

    def _sp_init_mask(self) -> None:
        super()._sp_init_mask()
        for name, param in self.model.named_parameters():
            self._ts_alpha[name] = torch.ones_like(param.data.cpu())
            self._ts_beta[name] = torch.ones_like(param.data.cpu())

    def _sp_update_mask(self, packages: Dict[int, Any]) -> None:
        total = sum(p["train_samples"] for p in packages.values())
        t = self._sp_t
        gamma = self.aggregated_gamma
        ratio = self.ts_ratio

        # Phase 1: update Beta params for active weights (prune signal)
        global_params = {n: p.data.cpu() for n, p in self.model.named_parameters()}
        for name in self._sp_mask_dict:
            mask = self._sp_mask_dict[name]
            K = int((mask == 1).sum().item())
            k = int(self.f_decay(t, self.adjust_alpha, self.T_end) * K)
            kappa = K - k  # core active count

            active_idx = (mask.view(-1) == 1).nonzero(as_tuple=False).view(-1)

            _, glb_top = torch.topk(global_params[name].abs().view(-1)[active_idx], kappa, largest=True)
            glb_outcome = torch.zeros(mask.numel())
            glb_outcome[active_idx[glb_top]] = 1.0

            client_outcome = torch.zeros(mask.numel())
            for pkg in packages.values():
                w = pkg["train_samples"] / total
                local_top_idx = pkg["_sp_extra"].get(name + ".active_topk")
                if local_top_idx is None:
                    continue
                lo = torch.zeros(mask.numel())
                lo[active_idx[local_top_idx]] = 1.0
                client_outcome += lo * w

            vote = (1 - gamma) * glb_outcome + gamma * client_outcome
            semi = vote.view(-1)[active_idx]
            self._ts_alpha[name].view(-1)[active_idx] += ratio * semi
            self._ts_beta[name].view(-1)[active_idx] += ratio * (1 - semi)

        # Phase 2: update Beta params for inactive weights (grow signal)
        for name in self._sp_mask_dict:
            mask = self._sp_mask_dict[name]
            K = int((mask == 1).sum().item())
            k = int(self.f_decay(t, self.adjust_alpha, self.T_end) * K)
            k = min(k, int((mask == 0).sum().item()))

            inactive_idx = (mask.view(-1) == 0).nonzero(as_tuple=False).view(-1)

            grad_outcome = torch.zeros(mask.numel())
            for pkg in packages.values():
                w = pkg["train_samples"] / total
                topk_rel = pkg["_sp_extra"].get(name + ".inactive_topk")
                if topk_rel is None:
                    continue
                lo = torch.zeros(mask.numel())
                lo[inactive_idx[topk_rel]] = 1.0
                grad_outcome += lo * w

            grad_vote = (1 - gamma) * 0.5 + gamma * grad_outcome
            semi = grad_vote.view(-1)[inactive_idx]
            self._ts_alpha[name].view(-1)[inactive_idx] += ratio * semi
            self._ts_beta[name].view(-1)[inactive_idx] += ratio * (1 - semi)

            samples = torch.distributions.Beta(
                self._ts_alpha[name].view(-1),
                self._ts_beta[name].view(-1),
            ).sample()
            _, top_idx = torch.topk(samples, K, largest=True)
            new_mask = torch.zeros(mask.numel())
            new_mask[top_idx] = 1.0
            self._sp_mask_dict[name] = new_mask.view(mask.shape)

        self.apply_mask(self.model, self._sp_mask_dict)


class FedRTS_Client(spFL_Client):
    """FedRTS client - votes on active/inactive weights via gradient+magnitude top-k."""

    def package(self) -> Dict[str, Any]:
        result = super().package()
        if not self._sp_is_adj:
            result["_sp_extra"] = {}
            return result

        grads = self._collect_gradients()
        extra: Dict[str, torch.Tensor] = {}

        t, T_end, alpha = self._sp_t, self._sp_T_end, self._sp_alpha
        for name, param in self.model.named_parameters():
            if name not in self._sp_mask_dict:
                continue
            mask = self._sp_mask_dict[name].view(-1)
            K = int((mask == 1).sum().item())
            k = int(self.f_decay(t, alpha, T_end) * K)
            kappa = K - k

            active_idx = (mask == 1).nonzero(as_tuple=False).view(-1)
            if kappa > 0 and len(active_idx) >= kappa:
                _, top_rel = torch.topk(
                    param.data.abs().cpu().view(-1)[active_idx], kappa, largest=True
                )
                extra[name + ".active_topk"] = top_rel.cpu()

            inactive_idx = (mask == 0).nonzero(as_tuple=False).view(-1)
            g = grads.get(name, torch.zeros(param.numel())).abs().view(-1)
            if k > 0 and len(inactive_idx) >= k:
                _, grow_rel = torch.topk(g[inactive_idx], k, largest=True)
                extra[name + ".inactive_topk"] = grow_rel.cpu()

        result["_sp_extra"] = extra
        return result
