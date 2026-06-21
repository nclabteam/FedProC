# -*- coding: utf-8 -*-
import itertools
import time
from collections import OrderedDict, deque

import ray

from .tFL import tFL


class aFL(tFL):
    """Buffered-async FL (FedBuff style).

    Workers run continuously. The server aggregates every time `buffer_size`
    client results have accumulated. Each aggregation counts as one iteration.
    Requires parallel=True (Ray actor pool).
    """

    optional = {"buffer_size": 10}

    def train(self) -> None:
        if not self.parallel:
            raise RuntimeError("aFL requires parallel Ray workers (set --num_workers > 0)")

        all_active = [i for i in range(self.num_clients) if not self.is_new[i]]
        K = min(self.buffer_size, len(all_active))
        global_version = 0

        pending = {}  # fut → (cid, wid, version_dispatched)
        idle = deque(range(self.trainer.num_workers))
        client_gen = itertools.cycle(all_active)

        def _dispatch():
            while idle:
                wid = idle.popleft()
                cid = next(client_gen)
                fut = self.trainer.dispatch_one(cid, wid)
                pending[fut] = (cid, wid, global_version)

        _dispatch()

        buffer: "OrderedDict[int, dict]" = OrderedDict()

        for agg_idx in range(self.iterations):
            round_start = time.time()
            self.current_iter = agg_idx
            self.logger.info("")
            self.logger.info(f"--- Aggregation {str(agg_idx).zfill(4)} (K={K}) ---")

            while len(buffer) < K:
                [done], _ = ray.wait(list(pending), num_returns=1)
                cid, wid, v_sent = pending.pop(done)
                out = ray.get(done)
                self.trainer._write_back(cid, out)
                buffer[cid] = out
                idle.append(wid)
                _dispatch()

            if agg_idx % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    self._pre_eval_hook(dataset_type)

            self.aggregate_client_updates(buffer)
            global_version += 1
            buffer.clear()

            if agg_idx % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    if not self.exclude_server_model_processes:
                        self.evaluate_generalization(dataset_type)

            self.metrics["send_mb"].append(self._send_mb_per_round)
            iter_time = time.time() - round_start
            self.metrics["time_per_iter"].append(iter_time)
            self.logger.info(f"Aggregation {str(agg_idx).zfill(4)} time: {iter_time:.2f}s")
            self.fix_results(default=self.default_value)
            if self.early_stopping():
                break

        for fut in pending:
            try:
                ray.cancel(fut, force=True)
            except Exception:
                pass

        self.save_results()
        try:
            self.close_logger()
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception:
            pass
