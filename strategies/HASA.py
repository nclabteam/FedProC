"""Heterogeneity-Aware Subnet Allocation."""

from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any, Mapping

import numpy as np
import ray
import torch

from .ptFL import ParameterState, ptFL, ptFL_Client
from .tFL import Trainer


class HASAShared:
    """HASA rules shared by the server and client."""

    @staticmethod
    def _hasa_validate_counts(counts: torch.Tensor) -> torch.Tensor:
        counts = torch.as_tensor(counts, dtype=torch.float64).flatten()
        if counts.numel() == 0:
            raise ValueError("HASA token-count vectors cannot be empty")
        if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
            raise ValueError("HASA token counts must be finite and nonnegative")
        if not bool(torch.equal(counts, counts.round())):
            raise ValueError("HASA token counts must be integer-valued")
        return counts

    @classmethod
    def _hasa_jsd_scores(
        cls, counts: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        if not alpha > 0.0:
            raise ValueError("hasa_alpha must be positive")
        if counts.ndim != 2 or counts.shape[0] < 2:
            raise ValueError("HASA requires at least two equal-length count vectors")
        counts = torch.stack(
            [cls._hasa_validate_counts(counts=row) for row in counts]
        )
        global_counts = counts.sum(dim=0)
        local = (counts + alpha) / (counts.sum(dim=1, keepdim=True) + alpha * counts.shape[1])
        global_distribution = (global_counts + alpha) / (
            global_counts.sum() + alpha * counts.shape[1]
        )
        mixture = 0.5 * (local + global_distribution)
        # Paper Eq. (11): JSD(p_i || p_g).
        return 0.5 * (
            (local * (local / mixture).log()).sum(dim=1)
            + (
                global_distribution
                * (global_distribution / mixture).log()
            ).sum(dim=1)
        )

    @staticmethod
    def _hasa_rank_normalize(scores: torch.Tensor) -> torch.Tensor:
        scores = torch.as_tensor(scores, dtype=torch.float64).flatten()
        if scores.numel() < 2 or not bool(torch.isfinite(scores).all()):
            raise ValueError("HASA requires at least two finite scores")
        order = scores.argsort(stable=True)
        sorted_scores = scores[order]
        _, inverse, counts = torch.unique_consecutive(
            sorted_scores,
            return_inverse=True,
            return_counts=True,
        )
        starts = counts.cumsum(dim=0) - counts
        average_ranks = starts.to(torch.float64) + (counts + 1) / 2
        ranks = torch.empty_like(scores)
        ranks[order] = average_ranks[inverse]
        # Paper Eq. (13): average-tie ranks mapped to [0, 1].
        return (ranks - 1) / (scores.numel() - 1)

    @classmethod
    def _hasa_allocate(
        cls,
        scores: torch.Tensor,
        sample_sizes: torch.Tensor,
        caps: torch.Tensor,
        minimum: float,
        maximum: float,
        budget: float,
    ) -> torch.Tensor:
        sample_sizes = torch.as_tensor(sample_sizes, dtype=torch.float64).flatten()
        raw_caps = torch.as_tensor(caps).flatten()
        tolerance = (
            4 * torch.finfo(raw_caps.dtype).eps
            if raw_caps.is_floating_point()
            else 0.0
        )
        caps = raw_caps.to(dtype=torch.float64)
        if sample_sizes.shape != caps.shape or sample_sizes.numel() != scores.numel():
            raise ValueError("HASA requires one sample size and cap per client")
        if not 0.0 < minimum <= maximum <= 1.0:
            raise ValueError("HASA width bounds must satisfy 0 < min <= max <= 1")
        if not bool(torch.isfinite(sample_sizes).all()) or bool(
            (sample_sizes <= 0).any()
        ):
            raise ValueError("HASA sample sizes must be positive and finite")
        if not bool(torch.isfinite(caps).all()) or bool(
            ((caps < minimum - tolerance) | (caps > maximum + tolerance)).any()
        ):
            raise ValueError("HASA client caps must lie within the width bounds")
        caps.clamp_(min=minimum, max=maximum)

        weights = sample_sizes / sample_sizes.sum()
        lower_budget = minimum
        upper_budget = torch.dot(weights, caps).item()
        if not lower_budget <= budget <= upper_budget:
            raise ValueError(
                f"hasa_budget must lie in [{lower_budget}, {upper_budget}]"
            )

        normalized = cls._hasa_rank_normalize(scores=scores)
        # Paper Eq. (14): map rank-normalized heterogeneity to preliminary width.
        widths = minimum + (maximum - minimum) * normalized
        # Paper Eqs. (16)-(17): exactly two scale-and-project passes.
        for _ in range(2):
            scale = budget / torch.dot(weights, widths)
            widths = (scale * widths).clamp(min=minimum)
            widths = torch.minimum(widths, caps)
        return widths


class HASA_Trainer(Trainer):
    """Collect HASA's one-time client statistics on reusable workers."""

    def collect_profiles(
        self, client_ids: list[int]
    ) -> OrderedDict[int, dict[str, Any]]:
        if not self.parallel:
            return OrderedDict(
                (
                    client_id,
                    self._receive(
                        cid=client_id,
                        out=self.worker.hasa_profile(client_id=client_id),
                    ),
                )
                for client_id in client_ids
            )

        futures = [
            self.workers[index % self.num_workers].hasa_profile.remote(
                client_id=client_id
            )
            for index, client_id in enumerate(client_ids)
        ]
        outputs = ray.get(futures)
        return OrderedDict(
            (
                client_id,
                self._receive(cid=client_id, out=output),
            )
            for client_id, output in zip(client_ids, outputs)
        )


class HASA(HASAShared, ptFL):
    """Allocate fixed prefix submodels from train-only token-distribution JSD."""

    optional = {
        "capacity": "0.8",
        "hasa_alpha": 1.0,
        "hasa_budget": 0.5,
        "hasa_min_ratio": 0.2,
        "hasa_max_ratio": 0.8,
        "hasa_count_bins": "n_neg,n_zero,n_pos",
        "hasa_aggregation": "full",
    }
    compulsory = {
        "join_ratio": 1.0,
        "random_join_ratio": False,
        "exclude_ratio": 0.0,
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--capacity",
            type=str,
            default=None,
            help="One HASA width cap or one comma-separated cap per client",
        )
        parser.add_argument("--hasa_alpha", type=float, default=None)
        parser.add_argument("--hasa_budget", type=float, default=None)
        parser.add_argument("--hasa_min_ratio", type=float, default=None)
        parser.add_argument("--hasa_max_ratio", type=float, default=None)
        parser.add_argument(
            "--hasa_count_bins",
            type=str,
            default=None,
            help="Comma-separated integer count statistics forming the histogram",
        )
        parser.add_argument(
            "--hasa_aggregation",
            type=str,
            choices=("full", "selective"),
            default=None,
            help="Published original-FedAvg or matched-baseline protocol",
        )

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        if len(self._pt_capacities) not in (1, self.num_clients):
            raise ValueError("capacity must contain one HASA cap or one per client")
        caps = torch.tensor(
            [
                self._pt_capacities[
                    0 if len(self._pt_capacities) == 1 else client_id
                ]
                for client_id in range(self.num_clients)
            ],
            dtype=torch.float64,
        )
        profiles = self.trainer.collect_profiles(
            client_ids=list(range(self.num_clients))
        )
        count_vectors = [profile["value_counts"] for profile in profiles.values()]
        if len({counts.numel() for counts in count_vectors}) != 1:
            raise ValueError("HASA clients must use one shared vocabulary")
        counts = torch.stack(count_vectors)
        sample_sizes = torch.tensor(
            [profile["sample_size"] for profile in profiles.values()],
            dtype=torch.float64,
        )
        scores = self._hasa_jsd_scores(counts=counts, alpha=self.hasa_alpha)
        widths = self._hasa_allocate(
            scores=scores,
            sample_sizes=sample_sizes,
            caps=caps,
            minimum=self.hasa_min_ratio,
            maximum=self.hasa_max_ratio,
            budget=self.hasa_budget,
        )
        self._hasa_sample_sizes = {
            client_id: float(sample_sizes[client_id])
            for client_id in range(self.num_clients)
        }
        self._hasa_capacities = {
            client_id: float(widths[client_id])
            for client_id in range(self.num_clients)
        }
        # The one-time statistics upload is real communication; charge it to
        # each client's uplink in the first round that reports a cost.
        self._hasa_profile_upload_mb = dict(self._uplink_sizes)

    def _make_trainer(self) -> HASA_Trainer:
        return HASA_Trainer(
            server=self,
            client_cls=self._client_cls(),
            configs=self.configs,
            times=self.times,
        )

    def _pt_capacity_for_client(self, client_id: int) -> float:
        return self._hasa_capacities[client_id]

    def _pt_select_indices(
        self,
        group_name: str,
        full_width: int,
        retained: int,
        client_id: int,
    ) -> torch.Tensor:
        del group_name, client_id
        return torch.arange(retained, dtype=torch.long)

    def _compute_send_mb(
        self, packages: Mapping[int, dict[str, Any]]
    ) -> tuple[dict[int, float], float]:
        uplink, downlink = super()._compute_send_mb(packages=packages)
        if self._hasa_profile_upload_mb:
            for client_id in uplink:
                uplink[client_id] += self._hasa_profile_upload_mb.pop(client_id, 0.0)
            self._hasa_profile_upload_mb = {}
        return uplink, downlink

    def _pt_aggregation_weight(
        self, client_id: int, package: Mapping[str, Any]
    ) -> float:
        del package
        return self._hasa_sample_sizes[client_id]

    def aggregate_client_updates(
        self, packages: OrderedDict[int, dict[str, Any]]
    ) -> None:
        if self.hasa_aggregation == "selective":
            super().aggregate_client_updates(packages=packages)
            return

        accum, counts, total_weight = self._pt_accumulate_client_updates(
            packages=packages
        )
        if total_weight <= 0.0:
            raise ValueError("HASA requires at least one client update")
        updated: ParameterState = OrderedDict()
        for name, original in self.public_model_params.items():
            # Paper original-FedAvg protocol: fill inactive coordinates with the
            # broadcast value before the sample-weighted full-tensor average.
            updated[name] = (
                accum[name]
                + original * (total_weight - counts[name]).to(dtype=original.dtype)
            ) / total_weight
        self._commit_global(new_params=updated)


class HASA_Client(HASAShared, ptFL_Client):
    """HASA client with one-time token counts and a fixed physical subnet."""

    def hasa_profile(self, client_id: int) -> dict[str, Any]:
        self.id = client_id
        self._load_private(client_id=client_id)
        bins = tuple(name.strip() for name in self.hasa_count_bins.split(","))
        # The published histogram is over a vocabulary. A time series has no
        # vocabulary, so the shared support is the data factory's own integer
        # count statistics, which partition each column's values into bins that
        # are identical in number and meaning for every client.
        values: list[float] = []
        for column in sorted(self.stats):
            for name in bins:
                if name not in self.stats[column]:
                    raise KeyError(
                        f"client {client_id} column {column!r} lacks HASA count "
                        f"statistic {name!r}"
                    )
                values.append(float(self.stats[column][name]))
        counts = self._hasa_validate_counts(counts=torch.tensor(values))
        with np.load(self.train_file) as data:
            sample_size = int(data["x"].shape[0])
        if counts.max().item() > torch.iinfo(torch.int32).max:
            raise OverflowError("HASA counts exceed the published int32 wire")
        return {
            "__wire__": ("value_counts", "sample_size"),
            "client_id": client_id,
            "value_counts": counts.to(dtype=torch.int32),
            "sample_size": sample_size,
        }
