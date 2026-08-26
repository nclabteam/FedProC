# -*- coding: utf-8 -*-
"""FedTiny progressive block-wise topology adjustment."""

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Set

import torch

from .spFL import spFL, spFL_Client


class FedTinyShared:
    """Progressive-pruning math shared with FedMef."""

    @staticmethod
    def selected_block(
        mask_dict: Mapping[str, torch.Tensor],
        current_iter: int,
        delta_T: int,
        num_blocks: int,
    ) -> Set[str]:
        """Select one contiguous parameter block in output-to-input order."""
        names = list(mask_dict)
        block_count = min(max(num_blocks, 1), len(names))
        blocks = [
            names[
                index
                * len(names)
                // block_count : (index + 1)
                * len(names)
                // block_count
            ]
            for index in range(block_count)
        ]
        adjustment = max(current_iter // delta_T - 1, 0)
        return set(blocks[-1 - adjustment % len(blocks)])

    @staticmethod
    def adjustment_counts(
        mask_dict: Mapping[str, torch.Tensor],
        names: Set[str],
        current_iter: int,
        adjust_alpha: float,
        T_end: int,
    ) -> Dict[str, int]:
        """Return the paper's per-layer grow/prune counts."""
        fraction = spFL.f_decay(
            t=current_iter,
            alpha=adjust_alpha,
            T_end=T_end,
        )
        return {
            name: min(
                int(fraction * mask_dict[name].sum().item()),
                int((~mask_dict[name].bool()).sum().item()),
            )
            for name in names
        }

    @staticmethod
    def topk_inactive_gradients(
        gradients: Mapping[str, torch.Tensor],
        mask_dict: Mapping[str, torch.Tensor],
        counts: Mapping[str, int],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Keep only signed top-gradient values and their absolute indices."""
        payload: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, count in counts.items():
            if count <= 0 or name not in gradients:
                continue
            inactive = (
                (~mask_dict[name].flatten().cpu().bool())
                .nonzero(as_tuple=False)
                .flatten()
            )
            gradient = gradients[name].detach().cpu().flatten()
            relative = torch.topk(
                input=gradient[inactive].abs(),
                k=min(count, inactive.numel()),
                largest=True,
                sorted=False,
            ).indices
            indices = inactive[relative]
            payload[name] = {
                "indices": indices,
                "values": gradient[indices],
            }
        return payload

    @staticmethod
    def mean_sparse_gradients(
        extras: Sequence[Mapping[str, Dict[str, torch.Tensor]]],
        weights: Sequence[float],
        mask_dict: Mapping[str, torch.Tensor],
        names: Set[str],
    ) -> Dict[str, torch.Tensor]:
        """Average sparse client top-k gradients with omitted values as zero."""
        client_weights = torch.as_tensor(weights, dtype=torch.float64)
        if client_weights.numel() != len(extras) or client_weights.sum() <= 0:
            raise ValueError("one positive weight is required per client payload")
        client_weights /= client_weights.sum()
        averaged: Dict[str, torch.Tensor] = {}
        for name in names:
            flat = torch.zeros(mask_dict[name].numel(), dtype=torch.float64)
            for weight, extra in zip(client_weights, extras):
                if name not in extra:
                    continue
                flat.index_add_(
                    dim=0,
                    index=extra[name]["indices"].long(),
                    source=extra[name]["values"].double() * weight,
                )
            averaged[name] = flat.view_as(mask_dict[name]).float()
        return averaged

    @staticmethod
    def lowest_active_indices(
        parameters: Mapping[str, torch.Tensor],
        mask_dict: Mapping[str, torch.Tensor],
        counts: Mapping[str, int],
    ) -> Dict[str, torch.Tensor]:
        """Return active coordinates earmarked for magnitude pruning."""
        result: Dict[str, torch.Tensor] = {}
        for name, count in counts.items():
            active = (
                mask_dict[name].flatten().cpu().bool().nonzero(as_tuple=False).flatten()
            )
            if count > 0:
                weights = parameters[name].detach().cpu().abs().flatten()
                result[name] = active[
                    torch.topk(
                        input=weights[active],
                        k=min(count, active.numel()),
                        largest=False,
                        sorted=False,
                    ).indices
                ]
        return result

    @staticmethod
    def swap_topology(
        parameters: Mapping[str, torch.Tensor],
        gradients: Mapping[str, torch.Tensor],
        mask_dict: Mapping[str, torch.Tensor],
        counts: Mapping[str, int],
        prune_indices: Mapping[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Grow top-gradient inactive links and prune old active links."""
        updated = spFL.clone_mask(mask_dict=mask_dict)
        for name, requested in counts.items():
            original = mask_dict[name].flatten().cpu().bool()
            active = original.nonzero(as_tuple=False).flatten()
            inactive = (~original).nonzero(as_tuple=False).flatten()
            count = min(requested, active.numel(), inactive.numel())
            if count <= 0:
                continue
            if prune_indices is not None and name in prune_indices:
                prune = prune_indices[name].long()[:count]
            else:
                weights = parameters[name].detach().cpu().abs().flatten()
                prune = active[
                    torch.topk(
                        input=weights[active],
                        k=count,
                        largest=False,
                        sorted=False,
                    ).indices
                ]
            gradient = gradients[name].detach().cpu().abs().flatten()
            grow = inactive[
                torch.topk(
                    input=gradient[inactive],
                    k=count,
                    largest=True,
                    sorted=False,
                ).indices
            ]
            flat = original.clone()
            flat[grow] = True
            flat[prune] = False
            updated[name] = flat.view_as(mask_dict[name])
        return updated


class FedTiny(FedTinyShared, spFL):
    """FedTiny server."""

    optional = {
        "delta_T": 10,
        "T_end": 100,
        "adjust_alpha": 0.3,
        "num_blocks": 5,
    }

    def _sp_update_mask(self, packages: Mapping[int, Dict[str, Any]]) -> None:
        names = self.selected_block(
            mask_dict=self._sp_mask_dict,
            current_iter=self.current_iter,
            delta_T=self.delta_T,
            num_blocks=self.num_blocks,
        )
        counts = self.adjustment_counts(
            mask_dict=self._sp_mask_dict,
            names=names,
            current_iter=self.current_iter,
            adjust_alpha=self.adjust_alpha,
            T_end=self.T_end,
        )
        extras = []
        scores = []
        for package in packages.values():
            extras.append(package.get("_sp_extra", {}))
            scores.append(package["score"])
        # Paper Eq. (8): g_tilde = sum_k |D_k| / |D| * g_tilde_k.
        gradients = self.mean_sparse_gradients(
            extras=extras,
            weights=scores,
            mask_dict=self._sp_mask_dict,
            names=names,
        )
        self._sp_mask_dict = self.swap_topology(
            parameters=self.public_model_params,
            gradients=gradients,
            mask_dict=self._sp_mask_dict,
            counts=counts,
        )


class FedTiny_Client(FedTinyShared, spFL_Client):
    """FedTiny worker."""

    def package(self) -> Dict[str, Any]:
        if not self._sp_is_adj:
            return self._package_sparse_extra(extra={})
        names = self.selected_block(
            mask_dict=self._sp_mask_dict,
            current_iter=self.current_iter,
            delta_T=self.delta_T,
            num_blocks=self.num_blocks,
        )
        counts = self.adjustment_counts(
            mask_dict=self._sp_mask_dict,
            names=names,
            current_iter=self.current_iter,
            adjust_alpha=self.adjust_alpha,
            T_end=self.T_end,
        )
        gradients = self._collect_gradients(names=names)
        return self._package_sparse_extra(
            extra=self.topk_inactive_gradients(
                gradients=gradients,
                mask_dict=self._sp_mask_dict,
                counts=counts,
            )
        )
