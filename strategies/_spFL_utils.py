# -*- coding: utf-8 -*-
"""Shared utilities for sparse/pruning FL strategies.

Ported from FedPruning reference (api/pruning/init_scheme.py, model_pruning.py).
"""
import re
from typing import Dict, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Density schedule
# ---------------------------------------------------------------------------

def f_decay(t: int, alpha: float, T_end: int) -> float:
    """Cosine decay: alpha/2 * (1 + cos(t*pi/T_end))."""
    return alpha / 2 * (1 + np.cos(t * np.pi / T_end))


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

_DEFAULT_IGNORE = [r".*\.bias$", r".*bn.*", r".*ln.*"]
_DEFAULT_IGNORE_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)


def get_sparse_layers(
    model: nn.Module,
    ignore_patterns: Optional[list] = None,
) -> Set[str]:
    if ignore_patterns is None:
        ignore_patterns = _DEFAULT_IGNORE
    sparse = {n for n, _ in model.named_parameters()}
    for pat in ignore_patterns:
        sparse = {n for n in sparse if re.match(pat, n) is None}
    for name, module in model.named_modules():
        if isinstance(module, _DEFAULT_IGNORE_TYPES):
            sparse = {n for n in sparse if not n.startswith(name)}
    return sparse


# ---------------------------------------------------------------------------
# ERK / uniform density allocation
# ---------------------------------------------------------------------------

def _erk_densities(
    shape_dict: Dict[str, tuple],
    sparse_set: Set[str],
    target_density: float,
) -> Dict[str, float]:
    """Erdős–Rényi-Kernel layer-wise density allocation."""
    total = sum(np.prod(s) for s in shape_dict.values())
    dense_total = sum(np.prod(shape_dict[n]) for n in shape_dict if n not in sparse_set)
    remain = int(target_density * total) - dense_total
    assert remain > 0, "target_density too small for the number of dense layers"

    eps, dense_layers = 1.0, set()
    valid = False
    while not valid:
        divisor, rhs, raw = 0.0, 0.0, {}
        for name in sparse_set:
            if name in dense_layers:
                rhs -= int(np.prod(shape_dict[name]) * (1 - target_density))
                continue
            n = np.prod(shape_dict[name])
            rhs += int(n * target_density)
            p = np.sum(shape_dict[name]) / n
            raw[name] = p
            divisor += p * n
        eps = rhs / divisor
        max_p = max(raw.values())
        if eps * max_p > 1:
            for n, p in raw.items():
                if p == max_p:
                    dense_layers.add(n)
        else:
            valid = True

    return {
        n: 1.0 if n in dense_layers else min(eps * raw[n], 1.0)
        for n in sparse_set
    }


def generate_layer_density_dict(
    model: nn.Module,
    target_density: float,
    strategy: str = "ERK_magnitude",
) -> Dict[str, float]:
    dist, _ = strategy.split("_")
    shape_dict = {n: tuple(p.shape) for n, p in model.named_parameters()}
    sparse_set = get_sparse_layers(model)
    total = sum(np.prod(s) for s in shape_dict.values())
    dense_total = sum(np.prod(shape_dict[n]) for n in shape_dict if n not in sparse_set)

    if dist == "uniform":
        sparse_total = total - dense_total
        d = (int(target_density * total) - dense_total) / sparse_total
        return {n: float(d) for n in sparse_set}
    elif dist in ("ER", "ERK"):
        return _erk_densities(shape_dict, sparse_set, target_density)
    raise ValueError(f"Unknown density strategy: {dist}")


# ---------------------------------------------------------------------------
# Mask initialisation
# ---------------------------------------------------------------------------

def init_mask(
    model: nn.Module,
    target_density: float,
    strategy: str = "ERK_magnitude",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    dist, prune_strat = strategy.split("_")
    layer_density = generate_layer_density_dict(model, target_density, strategy)
    mask_dict: Dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if name not in layer_density:
            continue
        density = layer_density[name]
        n = param.numel()
        num_keep = max(1, int(n * density))

        if prune_strat in ("magnitude", "mag"):
            _, idx = torch.sort(param.data.abs().view(-1), descending=True)
            mask = torch.zeros(n, dtype=param.dtype)
            mask[idx[:num_keep]] = 1.0
            mask_dict[name] = mask.view(param.shape)
        elif prune_strat == "random":
            idx = torch.randperm(n)
            mask = torch.zeros(n, dtype=param.dtype)
            mask[idx[:num_keep]] = 1.0
            mask_dict[name] = mask.view(param.shape)
        else:
            raise ValueError(f"Unknown pruning strategy: {prune_strat}")

    return mask_dict, layer_density


# ---------------------------------------------------------------------------
# Mask application
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_mask(model: nn.Module, mask_dict: Dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if name in mask_dict:
            param.data.mul_(mask_dict[name].to(param.device))


# ---------------------------------------------------------------------------
# Prune + grow (magnitude prune, gradient grow)
# ---------------------------------------------------------------------------

def sparse_update_step(
    model: nn.Module,
    gradients: Dict[str, torch.Tensor],
    mask_dict: Dict[str, torch.Tensor],
    t: int,
    T_end: int,
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Prune k smallest active by magnitude; grow k largest inactive by gradient."""
    for name, param in model.named_parameters():
        if name not in mask_dict:
            continue
        mask = mask_dict[name]
        active_num = int((mask == 1).sum().item())
        k = int(f_decay(t, alpha, T_end) * active_num)
        if k <= 0:
            continue

        flat_w = param.data.abs().view(-1)
        flat_m = mask.view(-1)

        active_idx = (flat_m == 1).nonzero(as_tuple=False).view(-1)
        _, prune_rel = torch.topk(flat_w[active_idx], k, largest=False)
        flat_m[active_idx[prune_rel]] = 0.0

        inactive_idx = (flat_m == 0).nonzero(as_tuple=False).view(-1)
        flat_g = gradients[name].abs().view(-1).to(flat_w.device)
        _, grow_rel = torch.topk(flat_g[inactive_idx], min(k, len(inactive_idx)), largest=True)
        flat_m[inactive_idx[grow_rel]] = 1.0

        mask_dict[name] = flat_m.view(mask.shape)

    return mask_dict


def sparse_pruning_step(
    model: nn.Module,
    mask_dict: Dict[str, torch.Tensor],
    t: int,
    T_end: int,
    alpha: float,
) -> Dict[str, torch.Tensor]:
    for name, param in model.named_parameters():
        if name not in mask_dict:
            continue
        mask = mask_dict[name]
        active_num = int((mask == 1).sum().item())
        k = int(f_decay(t, alpha, T_end) * active_num)
        if k <= 0:
            continue
        flat_w = param.data.abs().view(-1)
        flat_m = mask.view(-1)
        active_idx = (flat_m == 1).nonzero(as_tuple=False).view(-1)
        _, prune_rel = torch.topk(flat_w[active_idx], k, largest=False)
        flat_m[active_idx[prune_rel]] = 0.0
        mask_dict[name] = flat_m.view(mask.shape)
    return mask_dict


def sparse_growing_step(
    model: nn.Module,
    gradients: Dict[str, torch.Tensor],
    mask_dict: Dict[str, torch.Tensor],
    layer_density_dict: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    for name, param in model.named_parameters():
        if name not in layer_density_dict or name not in mask_dict:
            continue
        mask = mask_dict[name]
        target_k = max(1, int(param.numel() * layer_density_dict[name]))
        current_k = int((mask == 1).sum().item())
        k = max(0, target_k - current_k)
        if k <= 0:
            continue
        flat_m = mask.view(-1)
        flat_g = gradients[name].abs().view(-1).to(flat_m.device)
        inactive_idx = (flat_m == 0).nonzero(as_tuple=False).view(-1)
        _, grow_rel = torch.topk(flat_g[inactive_idx], min(k, len(inactive_idx)), largest=True)
        flat_m[inactive_idx[grow_rel]] = 1.0
        mask_dict[name] = flat_m.view(mask.shape)
    return mask_dict


def union_masks(masks_list: list) -> Dict[str, torch.Tensor]:
    """Logical OR across a list of mask dicts."""
    result = {}
    for masks in masks_list:
        for name, m in masks.items():
            if name not in result:
                result[name] = m.clone().bool()
            else:
                result[name] = result[name] | m.bool()
    return {n: m.float() for n, m in result.items()}


def magnitude_reprune(
    model: nn.Module,
    union_mask: Dict[str, torch.Tensor],
    layer_density_dict: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """Magnitude-prune the unioned mask back to target density per layer."""
    new_mask = {}
    for name, param in model.named_parameters():
        if name not in layer_density_dict:
            continue
        density = layer_density_dict[name]
        num_keep = max(1, int(param.numel() * density))
        masked_w = param.data.abs() * union_mask[name].to(param.device)
        _, idx = torch.sort(masked_w.view(-1), descending=True)
        mask = torch.zeros_like(param.data.view(-1))
        mask[idx[:num_keep]] = 1.0
        new_mask[name] = mask.view(param.shape)
    return new_mask
