# -*- coding: utf-8 -*-
import time
from collections import OrderedDict, deque
from contextlib import suppress
from typing import Any

import ray

from .tFL import tFL


class aFL(tFL):
    """Buffered asynchronous FL with reusable Ray workers."""

    optional = {"buffer_size": 10}

    def _dispatch_idle(
        self,
        idle: deque[int],
        available: deque[int],
        pending: dict[Any, tuple[int, int]],
    ) -> None:
        while idle and available:
            worker_id = idle.popleft()
            client_id = available.popleft()
            future = self.trainer.dispatch_one(cid=client_id, wid=worker_id)
            pending[future] = (client_id, worker_id)

    def train(self) -> None:
        if not self.parallel:
            raise RuntimeError(
                "aFL requires parallel Ray workers (set CUDA device IDs and "
                "--num_workers > 0)"
            )

        active = [i for i in range(self.num_clients) if not self.is_new[i]]
        if not active:
            raise ValueError("at least one incumbent client is required")
        if int(self.buffer_size) <= 0:
            raise ValueError("buffer_size must be positive")

        target_size = min(int(self.buffer_size), len(active))
        available = deque(active)
        idle = deque(range(self.trainer.num_workers))
        pending: dict[Any, tuple[int, int]] = {}
        buffer: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._dispatch_idle(idle=idle, available=available, pending=pending)

        for agg_idx in range(self.iterations):
            round_start = time.time()
            self.current_iter = agg_idx
            self.logger.info("")
            self.logger.info(
                f"--- Aggregation {str(agg_idx).zfill(4)} (K={target_size}) ---"
            )

            while len(buffer) < target_size:
                [done], _ = ray.wait(list(pending), num_returns=1)
                client_id, worker_id = pending.pop(done)
                output = self.trainer._receive(
                    cid=client_id,
                    out=ray.get(done),
                )
                self.trainer._write_back(cid=client_id, out=output)
                buffer[client_id] = output
                idle.append(worker_id)
                self._dispatch_idle(
                    idle=idle,
                    available=available,
                    pending=pending,
                )

            self.selected_clients = list(buffer)
            self.current_num_join_clients = len(buffer)

            if agg_idx % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    self._pre_eval_hook(dataset_type=dataset_type)

            self.aggregate_client_updates(packages=buffer)
            uplink, downlink = self._compute_send_mb(packages=buffer)
            self.metrics["downlink_mb"].append(downlink)
            for client_id, megabytes in uplink.items():
                self._round_client_data.setdefault(client_id, {})[
                    "uplink_mb"
                ] = megabytes
            completed = list(buffer)
            buffer.clear()
            available.extend(completed)
            self.current_iter = agg_idx + 1
            self._dispatch_idle(
                idle=idle,
                available=available,
                pending=pending,
            )
            self.current_iter = agg_idx

            if agg_idx % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    if not self.exclude_server_model_processes:
                        self.evaluate_generalization(dataset_type=dataset_type)
                self._save_best_hook()
            iter_time = time.time() - round_start
            self.metrics["time_per_iter"].append(iter_time)
            self.logger.info(f"{iter_time:.2f}s")
            self._flush_round()
            if self.early_stopping():
                break

        for future in pending:
            with suppress(Exception):
                ray.cancel(future, force=True)
        self._finish_training()
