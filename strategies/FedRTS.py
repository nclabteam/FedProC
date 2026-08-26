# -*- coding: utf-8 -*-
"""FedRTS combinatorial Thompson-sampling topology adjustment."""

from argparse import Namespace
from collections.abc import Mapping, Sequence
from typing import Any, Dict

import torch

from .spFL import spFL, spFL_Client


class FedRTSShared:
    """FedRTS posterior and voting math."""

    @staticmethod
    def topology_counts(
        mask_dict: Mapping[str, torch.Tensor],
        current_iter: int,
        adjust_alpha: float,
        T_end: int,
    ) -> Dict[str, int]:
        """Return K-kappa, the number of links reconsidered per layer."""
        fraction = spFL.f_decay(
            t=current_iter,
            alpha=adjust_alpha,
            T_end=T_end,
        )
        return {
            name: min(
                int(fraction * mask.sum().item()),
                int((~mask.bool()).sum().item()),
            )
            for name, mask in mask_dict.items()
        }

    @staticmethod
    def active_outcome(
        global_parameter: torch.Tensor,
        client_parameters: Sequence[torch.Tensor],
        active_indices: torch.Tensor,
        core_count: int,
        weights: Sequence[float],
        gamma: float,
    ) -> torch.Tensor:
        """Return active-link outcomes from global and client magnitudes."""
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        normalized = torch.as_tensor(weights, dtype=torch.float64)
        if normalized.numel() != len(client_parameters) or normalized.sum() <= 0:
            raise ValueError("one positive weight is required per client")
        normalized /= normalized.sum()
        global_vote = torch.zeros(active_indices.numel(), dtype=torch.float64)
        client_vote = torch.zeros_like(global_vote)
        if core_count > 0:
            global_core = torch.topk(
                input=global_parameter.detach().cpu().abs().flatten()[active_indices],
                k=core_count,
                largest=True,
                sorted=False,
            ).indices
            global_vote[global_core] = 1
            values = torch.stack(
                [
                    parameter.detach().cpu().abs().flatten()[active_indices]
                    for parameter in client_parameters
                ]
            )
            client_core = torch.topk(
                input=values,
                k=core_count,
                dim=1,
                largest=True,
                sorted=False,
            ).indices
            votes = torch.zeros_like(values, dtype=torch.float64)
            votes.scatter_(dim=1, index=client_core, value=1)
            client_vote = torch.tensordot(
                normalized,
                votes,
                dims=([0], [0]),
            )
        # Official implementation: (1-gamma) global + gamma client evidence.
        return (1 - gamma) * global_vote + gamma * client_vote

    @staticmethod
    def inactive_outcome(
        numel: int,
        inactive_indices: torch.Tensor,
        client_indices: Sequence[torch.Tensor],
        weights: Sequence[float],
        gamma: float,
    ) -> torch.Tensor:
        """Return inactive-link outcomes with the paper's 0.5 global prior."""
        normalized = torch.as_tensor(weights, dtype=torch.float64)
        if normalized.numel() != len(client_indices) or normalized.sum() <= 0:
            raise ValueError("one positive weight is required per client")
        normalized /= normalized.sum()
        votes = torch.zeros((len(client_indices), numel), dtype=torch.float64)
        for row, indices in enumerate(client_indices):
            votes[row, indices.detach().cpu().long()] = 1
        client_vote = torch.tensordot(
            normalized,
            votes,
            dims=([0], [0]),
        )
        return ((1 - gamma) * 0.5 + gamma * client_vote)[inactive_indices]

    @staticmethod
    def update_posterior(
        alpha: torch.Tensor,
        beta: torch.Tensor,
        indices: torch.Tensor,
        outcomes: torch.Tensor,
        evidence_scale: float,
    ) -> None:
        """Apply the Beta-Bernoulli semi-bandit update in place."""
        if evidence_scale <= 0:
            raise ValueError("evidence_scale must be positive")
        alpha.flatten()[indices] += evidence_scale * outcomes.to(alpha.dtype)
        beta.flatten()[indices] += evidence_scale * (1 - outcomes).to(beta.dtype)

    @staticmethod
    def sample_topology(
        alpha: Mapping[str, torch.Tensor],
        beta: Mapping[str, torch.Tensor],
        keep_counts: Mapping[str, int],
        seed: int,
    ) -> Dict[str, torch.Tensor]:
        """Sample each link's posterior and retain each layer's top-K arms."""
        result: Dict[str, torch.Tensor] = {}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for name, keep in keep_counts.items():
                samples = torch.distributions.Beta(
                    alpha[name].flatten(),
                    beta[name].flatten(),
                ).sample()
                chosen = torch.topk(
                    input=samples,
                    k=keep,
                    largest=True,
                    sorted=False,
                ).indices
                mask = torch.zeros_like(samples, dtype=torch.bool)
                mask[chosen] = True
                result[name] = mask.view_as(alpha[name])
        return result

    @staticmethod
    def growth_indices(
        gradients: Mapping[str, torch.Tensor],
        mask_dict: Mapping[str, torch.Tensor],
        counts: Mapping[str, int],
    ) -> Dict[str, torch.Tensor]:
        """Return absolute indices of each client's top inactive gradients."""
        result: Dict[str, torch.Tensor] = {}
        for name, count in counts.items():
            if count <= 0:
                continue
            inactive = (
                (~mask_dict[name].flatten().cpu().bool())
                .nonzero(as_tuple=False)
                .flatten()
            )
            gradient = gradients[name].detach().cpu().abs().flatten()
            result[name] = inactive[
                torch.topk(
                    input=gradient[inactive],
                    k=min(count, inactive.numel()),
                    largest=True,
                    sorted=False,
                ).indices
            ]
        return result


class FedRTS(FedRTSShared, spFL):
    """FedRTS server."""

    optional = {
        "delta_T": 10,
        "T_end": 300,
        "adjust_alpha": 0.4,
        "aggregated_gamma": 0.5,
        "evidence_scale": 10.0,
        "pruning_strategy": "ERK_random",
    }

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self._ts_alpha: Dict[str, torch.Tensor] = {}
        self._ts_beta: Dict[str, torch.Tensor] = {}

    def _sp_init_mask(self) -> None:
        super()._sp_init_mask()
        self._ts_alpha = {
            name: torch.ones_like(mask, dtype=torch.float32)
            for name, mask in self._sp_mask_dict.items()
        }
        self._ts_beta = {
            name: torch.ones_like(mask, dtype=torch.float32)
            for name, mask in self._sp_mask_dict.items()
        }

    def aggregate_client_updates(
        self,
        packages: Mapping[int, Dict[str, Any]],
    ) -> None:
        client_models = []
        client_extras = []
        weights = []
        for package in packages.values():
            client_models.append(package["regular_model_params"])
            client_extras.append(package.get("_sp_extra", {}))
            weights.append(package["score"])
        self._commit_global(
            new_params=self.mean_models(
                models=client_models,
                weights=weights,
            )
        )
        counts = self.topology_counts(
            mask_dict=self._sp_mask_dict,
            current_iter=self.current_iter,
            adjust_alpha=self.adjust_alpha,
            T_end=self.T_end,
        )
        for name, mask in self._sp_mask_dict.items():
            active = mask.flatten().bool().nonzero(as_tuple=False).flatten()
            core_count = active.numel() - counts[name]
            outcomes = self.active_outcome(
                global_parameter=self.public_model_params[name],
                client_parameters=[model[name] for model in client_models],
                active_indices=active,
                core_count=core_count,
                weights=weights,
                gamma=self.aggregated_gamma,
            )
            # Paper Eq. (8): (alpha_i,beta_i) += lambda(X_i,1-X_i).
            self.update_posterior(
                alpha=self._ts_alpha[name],
                beta=self._ts_beta[name],
                indices=active,
                outcomes=outcomes,
                evidence_scale=self.evidence_scale,
            )

        if self._sp_is_adj():
            for name, mask in self._sp_mask_dict.items():
                inactive = (~mask.flatten().bool()).nonzero(as_tuple=False).flatten()
                outcomes = self.inactive_outcome(
                    numel=mask.numel(),
                    inactive_indices=inactive,
                    client_indices=[
                        extra.get(name, torch.empty(0, dtype=torch.long))
                        for extra in client_extras
                    ],
                    weights=weights,
                    gamma=self.aggregated_gamma,
                )
                self.update_posterior(
                    alpha=self._ts_alpha[name],
                    beta=self._ts_beta[name],
                    indices=inactive,
                    outcomes=outcomes,
                    evidence_scale=self.evidence_scale,
                )
            # Paper Eq. (4): sample xi_i~Beta and choose the top-K links.
            self._sp_mask_dict = self.sample_topology(
                alpha=self._ts_alpha,
                beta=self._ts_beta,
                keep_counts={
                    name: int(mask.sum().item())
                    for name, mask in self._sp_mask_dict.items()
                },
                seed=self.current_iter,
            )
        self._sp_commit_mask()


class FedRTS_Client(FedRTSShared, spFL_Client):
    """FedRTS worker."""

    def package(self) -> Dict[str, Any]:
        if not self._sp_is_adj:
            return self._package_sparse_extra(extra={})
        counts = self.topology_counts(
            mask_dict=self._sp_mask_dict,
            current_iter=self.current_iter,
            adjust_alpha=self.adjust_alpha,
            T_end=self.T_end,
        )
        gradients = self._collect_gradients(names=set(self._sp_mask_dict))
        return self._package_sparse_extra(
            extra=self.growth_indices(
                gradients=gradients,
                mask_dict=self._sp_mask_dict,
                counts=counts,
            )
        )
