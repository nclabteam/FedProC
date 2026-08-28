"""FLuID: invariant-dropout submodels sized by runtime straggler detection."""

from __future__ import annotations

import math
import random
import time
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from .ptFL import ptFL, ptFLShared, ptFL_Client


class FLuIDShared(ptFLShared):
    """Invariant-neuron statistics shared by the FLuID server and client."""

    @staticmethod
    def _fluid_cells(model: nn.Module) -> tuple[tuple[int, nn.Module], ...]:
        cells = model.cells
        if isinstance(cells, nn.ModuleDict):
            return tuple((int(key), cells[key]) for key in cells)
        return tuple(enumerate(cells))

    @classmethod
    def _fluid_unit_axes(
        cls, model: nn.Module
    ) -> "OrderedDict[str, tuple[int, tuple[tuple[str, int], ...]]]":
        """Map every layer to the tensor axes its hidden units index.

        The official implementation scores a neuron by the largest relative
        change over *every* weight the neuron touches, which for stacked-gate
        LSTM weights is its row in each gate block, its bias entry, its column
        in the recurrent matrix, and its column in whatever consumes the layer
        (the next cell's input weights, or the forecast head for the last
        layer). That is exactly the set of coordinates the physical extraction
        removes when the neuron is dropped, so the score and the cut agree.
        """

        cells = cls._fluid_cells(model=model)
        axes: "OrderedDict[str, tuple[int, tuple[tuple[str, int], ...]]]" = (
            OrderedDict()
        )
        for position, (layer, cell) in enumerate(cells):
            hidden_size = int(cell.hidden_size)
            group_name = f"cells.{layer}"
            tagged: list[tuple[str, int]] = []
            for local_name, _ in cell.named_parameters(recurse=False):
                name = f"{group_name}.{local_name}"
                tagged.append((name, 0))
                if local_name.startswith(cls._PT_RECURRENT_WEIGHT_PREFIXES):
                    tagged.append((name, 1))
            if position + 1 < len(cells):
                next_layer, next_cell = cells[position + 1]
                for local_name, _ in next_cell.named_parameters(recurse=False):
                    if local_name.startswith(cls._PT_INPUT_WEIGHT_PREFIXES):
                        tagged.append((f"cells.{next_layer}.{local_name}", 1))
            else:
                tagged.append(("fc_pred.weight", 1))
            axes[group_name] = (hidden_size, tuple(tagged))
        return axes

    @staticmethod
    def _fluid_reduce_to_units(
        values: torch.Tensor, axis: int, hidden_size: int, reduce: str
    ) -> torch.Tensor:
        """Collapse one tagged axis of ``values`` onto its hidden units."""

        moved = values.movedim(axis, 0).reshape(values.shape[axis], -1)
        length = int(moved.shape[0])
        if length % hidden_size != 0:
            raise ValueError(
                f"axis {axis} of length {length} is not a whole number of "
                f"hidden blocks of size {hidden_size}"
            )
        units = torch.arange(length, dtype=torch.long) % hidden_size
        if reduce == "amax":
            per_index = moved.amax(dim=1)
            out = torch.full((hidden_size,), -math.inf, dtype=per_index.dtype)
            out.scatter_reduce_(dim=0, index=units, src=per_index, reduce="amax")
            return out
        per_index = moved.all(dim=1).to(dtype=torch.float32)
        out = torch.ones(hidden_size, dtype=torch.float32)
        out.scatter_reduce_(dim=0, index=units, src=per_index, reduce="amin")
        return out > 0.5

    @classmethod
    def _fluid_relative_changes(
        cls,
        model: nn.Module,
        previous: Mapping[str, torch.Tensor],
        current: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return ``|w(t) - w(t-1)| / |w(t-1)|`` for every scored tensor."""

        relative: dict[str, torch.Tensor] = {}
        wanted = {
            name
            for _hidden, tagged in cls._fluid_unit_axes(model=model).values()
            for name, _axis in tagged
        }
        for name in sorted(wanted):
            before = previous[name].detach().to(dtype=torch.float64)
            after = current[name].detach().to(dtype=torch.float64)
            if before.shape != after.shape:
                raise ValueError(
                    f"{name}: local shape {tuple(after.shape)} does not match "
                    f"the broadcast shape {tuple(before.shape)}"
                )
            change = (after - before).abs()
            # ``|w(t) - w(t-1)| <= th * |w(t-1)|`` is the invariance test, so a
            # zero broadcast weight admits no threshold at all unless the weight
            # did not move.
            relative[name] = torch.where(
                before == 0,
                torch.where(
                    change == 0,
                    torch.zeros_like(change),
                    torch.full_like(change, math.inf),
                ),
                change / before.abs(),
            )
        return relative

    @classmethod
    def _fluid_neuron_scores(
        cls,
        model: nn.Module,
        previous: Mapping[str, torch.Tensor],
        current: Mapping[str, torch.Tensor],
    ) -> "OrderedDict[str, torch.Tensor]":
        """Return the per-hidden-unit percent difference ``g`` for every cell.

        The paper defines ``g`` for neuron ``i`` of layer ``j`` on client ``c``
        as the minimum value satisfying
        ``g >= (w_ijc(t) - w_ij(t-1)) / w_ij(t-1)`` across the neuron's
        weights, that is the tightest bound on its relative weight change.
        """

        relative = cls._fluid_relative_changes(
            model=model, previous=previous, current=current
        )
        scores: OrderedDict[str, torch.Tensor] = OrderedDict()
        for group_name, (hidden_size, tagged) in cls._fluid_unit_axes(
            model=model
        ).items():
            unit_scores = torch.full((hidden_size,), -math.inf, dtype=torch.float64)
            for name, axis in tagged:
                unit_scores = torch.maximum(
                    unit_scores,
                    cls._fluid_reduce_to_units(
                        values=relative[name],
                        axis=axis,
                        hidden_size=hidden_size,
                        reduce="amax",
                    ),
                )
            scores[group_name] = torch.where(
                unit_scores == -math.inf,
                torch.zeros_like(unit_scores),
                unit_scores,
            )
        return scores


class FLuID(FLuIDShared, ptFL):
    """Server-side invariant dropout driven by measured client durations."""

    optional = {
        "fluid_initial_threshold": 30.0,
        "fluid_threshold_step": 0.1,
        "fluid_majority": 0.5,
    }

    # The paper raises the threshold "until the number of neurons below the
    # threshold is greater than or equal to the number of neurons to be left
    # out of the sub-model", but a neuron whose broadcast weight moved off zero
    # scores an infinite relative change and never qualifies, so the search is
    # bounded rather than unbounded.
    _FLUID_MAX_RAISES = 1000

    # The official ``aggregate_drop`` divides each coordinate by the number of
    # examples that actually trained it, which is the sample-weighted mean this
    # flag selects.
    _pt_send_score = True

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        if float(self.fluid_threshold_step) <= 0.0:
            raise ValueError("fluid_threshold_step must be positive")
        if not 0.0 < float(self.fluid_majority) < 1.0:
            raise ValueError("fluid_majority must lie in (0, 1)")

        # Straggler bookkeeping: ``cid -> the duration that made it a straggler``.
        self._fluid_straggler: dict[int, float] = {}
        self._fluid_just_updated = False
        # Sub-model size; the official strategy defaults to 0.95 and replaces it
        # from the first duration comparison.
        self._fluid_p_val = 0.95

        # "FLuID can have a different drop threshold for each layer."
        self._fluid_thresholds: dict[str, float] = {}

        self._fluid_unchanged: dict[str, list[int]] = {}
        self._fluid_def_drop: dict[str, list[int]] = {}
        self._fluid_prev_drop: dict[str, list[int]] = {}
        self._fluid_rng = random.Random(int(self.seed) + int(self.times))

    @property
    def _fluid_round(self) -> int:
        """The official strategy numbers rounds from one, ``current_iter`` from zero."""

        return int(self.current_iter) + 1

    def _pt_capacity_for_client(self, client_id: int) -> float:
        """Only a detected straggler receives a submodel, and only after round two."""

        if client_id in self._fluid_straggler and self._fluid_round > 2:
            return self._fluid_p_val
        return 1.0

    def _pt_select_indices(
        self,
        group_name: str,
        full_width: int,
        retained: int,
        client_id: int,
    ) -> torch.Tensor:
        del client_id
        if retained >= full_width:
            return torch.arange(full_width, dtype=torch.long)
        dropped = self._fluid_choose_dropped(
            group_name=group_name,
            full_width=full_width,
            num_drop=full_width - retained,
        )
        self._fluid_prev_drop[group_name] = sorted(dropped)
        keep = sorted(set(range(full_width)) - set(dropped))
        return torch.tensor(keep, dtype=torch.long)

    def _fluid_choose_dropped(
        self, group_name: str, full_width: int, num_drop: int
    ) -> list[int]:
        """Reproduce ``drop_dynamic``'s three-tier priority and random sampling."""

        definite = list(self._fluid_def_drop.get(group_name, ()))
        unchanged = list(self._fluid_unchanged.get(group_name, ()))

        if len(definite) >= num_drop:
            # Neurons that were invariant at the last cut and are invariant
            # again: the paper's "consistently fall below the threshold over
            # multiple epochs" tier.
            return sorted(self._fluid_rng.sample(definite, num_drop))
        if len(unchanged) >= num_drop:
            pool = [unit for unit in unchanged if unit not in set(definite)]
            chosen = self._fluid_rng.sample(pool, num_drop - len(definite))
            chosen.extend(definite)
            return sorted(chosen)
        pool = [unit for unit in range(full_width) if unit not in set(unchanged)]
        chosen = self._fluid_rng.sample(pool, num_drop - len(unchanged))
        chosen.extend(unchanged)
        return sorted(chosen)

    def _fluid_stable_units(
        self,
        sources: Sequence[dict[str, torch.Tensor]],
        hidden_size: int,
        tagged: Sequence[tuple[str, int]],
        threshold: float,
    ) -> list[int]:
        """Units a majority of non-stragglers left below ``threshold``.

        The paper drops neurons whose updates stay within the threshold "for the
        majority of non-stragglers", so the fraction is strict: an even split is
        not a majority. The reference implementation instead requires 75% with a
        non-strict comparison, which ``fluid_majority`` can still express.

        The count is taken per weight coordinate first and reduced over the
        unit afterwards, matching the official reduction order.
        """

        required = float(self.fluid_majority) * len(sources)
        invariant = torch.ones(hidden_size, dtype=torch.bool)
        for name, axis in tagged:
            below = torch.stack([source[name] <= threshold for source in sources]).sum(
                dim=0
            )
            invariant &= self._fluid_reduce_to_units(
                values=below > required,
                axis=axis,
                hidden_size=hidden_size,
                reduce="all",
            )
        return [unit for unit, flag in enumerate(invariant.tolist()) if flag]

    def _fluid_find_stable(self, sources: Sequence[dict[str, torch.Tensor]]) -> None:
        """Raise each layer's threshold until enough units fall below it.

        Per the paper the threshold climbs every epoch until at least as many
        neurons are invariant as the sub-model has to leave out.
        """

        step = 1.0 + float(self.fluid_threshold_step)
        # The threshold is compared against the score tensors, so it has to stay
        # finite in *their* dtype: overflowing to inf there would admit even the
        # units whose relative change is itself infinite.
        ceiling = float(torch.finfo(next(iter(sources[0].values())).dtype).max)
        for group_name, (hidden_size, tagged) in self._fluid_unit_axes(
            model=self.model
        ).items():
            threshold = self._fluid_thresholds.get(
                group_name, float(self.fluid_initial_threshold)
            )
            num_drop = hidden_size - self._pt_retained_width(
                full_width=hidden_size, capacity=self._fluid_p_val
            )
            units = self._fluid_stable_units(
                sources=sources,
                hidden_size=hidden_size,
                tagged=tagged,
                threshold=threshold,
            )
            for _ in range(self._FLUID_MAX_RAISES):
                if len(units) >= num_drop:
                    break
                # A seeded threshold of zero cannot grow by scaling.
                raised = (
                    threshold * step
                    if threshold > 0.0
                    else float(self.fluid_threshold_step)
                )
                if not math.isfinite(raised) or raised > ceiling:
                    break
                threshold = raised
                units = self._fluid_stable_units(
                    sources=sources,
                    hidden_size=hidden_size,
                    tagged=tagged,
                    threshold=threshold,
                )
            self._fluid_thresholds[group_name] = threshold
            self._fluid_unchanged[group_name] = units
            previous = set(self._fluid_prev_drop.get(group_name, ()))
            self._fluid_def_drop[group_name] = [
                unit for unit in units if unit in previous
            ]

    def _fluid_find_min(
        self, scores: Sequence["OrderedDict[str, torch.Tensor]"]
    ) -> None:
        """Seed each layer's threshold from its smallest percent update.

        The paper averages the minimum percent update over the initial few
        epochs, which are rounds two and three here.
        """

        if self._fluid_round not in (2, 3):
            return
        for group_name in scores[0]:
            minimum = float(
                torch.stack([score[group_name] for score in scores])
                .amax(dim=0)
                .amin()
                .item()
            )
            if not math.isfinite(minimum):
                continue
            seeded = self._fluid_thresholds.get(group_name)
            self._fluid_thresholds[group_name] = (
                minimum
                if self._fluid_round == 2 or seeded is None
                else (seeded + minimum) / 2.0
            )

    def _fluid_set_p_val(self, percent_difference: float) -> None:
        """The official five-bucket ladder from the straggler speedup ratio."""

        if percent_difference >= 0.90:
            self._fluid_p_val = 0.95
        elif percent_difference >= 0.80:
            self._fluid_p_val = 0.85
        elif percent_difference >= 0.70:
            self._fluid_p_val = 0.75
        elif percent_difference >= 0.60:
            self._fluid_p_val = 0.65
        else:
            self._fluid_p_val = 0.5

    def _fluid_update_straggler(
        self, packages: "OrderedDict[int, dict[str, Any]]"
    ) -> None:
        """Detect, and later revise, the straggler set from measured durations."""

        missing = [cid for cid in packages if "duration" not in packages[cid]]
        if missing:
            raise KeyError(
                f"FLuID needs a measured duration from every client; missing {missing}"
            )
        duration: Callable[[int], float] = lambda cid: float(packages[cid]["duration"])
        order = sorted(packages, key=duration)
        if len(order) < 2:
            raise ValueError("FLuID needs at least two clients to rank durations")

        if not self._fluid_straggler and self._fluid_round > 1:
            slowest = order[-1]
            self._fluid_straggler[slowest] = duration(slowest)
            self._fluid_set_p_val(duration(order[-2]) / duration(slowest))
        elif (
            self._fluid_straggler
            and self._fluid_round > 1
            and not self._fluid_just_updated
        ):
            slowest = order[-1]
            if slowest not in self._fluid_straggler:
                # Recover what the current straggler would have cost at full
                # width, so the two devices are compared on the same footing.
                for client_id in packages:
                    if client_id in self._fluid_straggler:
                        self._fluid_straggler[client_id] = (
                            duration(client_id) / self._fluid_p_val
                        )
                ranked = list(self._fluid_straggler.items())
                if duration(slowest) > ranked[0][1]:
                    self._fluid_straggler[slowest] = duration(slowest)
                    self._fluid_set_p_val(ranked[0][1] / duration(slowest))
                    self._fluid_just_updated = True
                    self._fluid_straggler.pop(ranked[0][0])
        else:
            self._fluid_just_updated = False

    def aggregate_client_updates(
        self, packages: "OrderedDict[int, dict[str, Any]]"
    ) -> None:
        full_width = [
            package
            for client_id, package in packages.items()
            if client_id not in self._fluid_straggler
        ]
        if full_width:
            relative = [
                self._fluid_relative_changes(
                    model=self.model,
                    previous=self.public_model_params,
                    current=package["regular_model_params"],
                )
                for package in full_width
            ]
            scores = [
                self._fluid_neuron_scores(
                    model=self.model,
                    previous=self.public_model_params,
                    current=package["regular_model_params"],
                )
                for package in full_width
            ]
            # Seeding comes first: the stability search climbs from it.
            self._fluid_find_min(scores=scores)
            self._fluid_find_stable(sources=relative)

        super().aggregate_client_updates(packages=packages)
        self._fluid_update_straggler(packages=packages)


class FLuID_Client(FLuIDShared, ptFL_Client):
    """FLuID client; it reports the training duration the server ranks on."""

    _pt_send_score = True

    def train(self, package: dict[str, Any]) -> dict[str, Any]:
        self.set_parameters(package=package)
        started = time.perf_counter()
        self.fit()
        elapsed = time.perf_counter() - started
        out = self.package()
        out["duration"] = elapsed
        out["__wire__"] = tuple(out["__wire__"]) + ("duration",)
        return out
