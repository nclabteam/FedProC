"""TidalLR: gated per-client LR retuning + gradient-norm rescaler for sparse hydrology DFL.

Thesis: on sparse non-IID hydrology DFL, per-client LR calibration dominates
aggregation-weight engineering. The aggregator broadcasts actual coverage so
w_ij != score_j for off-diagonal terms (unlike DFedHPO's pure-score aggregator).
Self-weight is adaptive: clients with sparse data lean more on neighbors.

Two LR mechanisms:
  1. Plateau-triggered, validation-gated retune: re-runs bracketed HPO only when
     training plateaus (median-of-deltas test), and only accepts the new LR if it
     strictly improves the incumbent loss re-evaluated under the current model state.
     A mandatory first probe fires at round (window + 5) regardless of plateau state.
  2. Per-step gradient-norm LR rescaler: between retune events, keeps
     lr * ||g||_ema near a per-client target so the optimizer adapts to shifts in
     gradient scale without extra communication. EMA is cold-reset at every round
     start with a k-step warmup so aggregation jumps don't contaminate the anchor.

Inherits from DFedHPO / DFedHPO_Client.
"""

import copy
import math
import time
from collections import deque

import numpy as np
import torch

from .DFedHPO import DFedHPO, DFedHPO_Client, _NoOpScheduler

# ============================================================== server side


class TidalLR(DFedHPO):
    """DFedHPO with plateau-triggered, validation-gated per-client LR retuning."""

    optional = {
        "topology": "FullyConnected",
        # DFedHPO cold-start HPO knobs
        "trials": 10,
        "eval_epochs": 3,
        "aggregator": "FA",
        "top_k": 3,
        "lr_min": 1e-5,
        "lr_max": 1e-1,
        # Aggregator knobs
        "tidal_alpha_beta": 4.0,  # sigmoid steepness: alpha = sigmoid(beta*(cov - 0.5))
        "tidal_alpha_min": 0.05,  # floor on adaptive self-weight
        "tidal_coverage_floor": 0.05,
        "tidal_score_floor": 1.0,
        "tidal_weight_floor": 1e-8,
        # Plateau-triggered retune knobs
        "lr_retune_window": 8,
        "lr_retune_tol": 1e-4,  # MAD threshold for plateau
        "lr_retune_sigma": 2.0,  # search half-range: [lr/sigma, lr*sigma]
        "lr_retune_trials": 5,
        "lr_retune_eval_epochs": 2,
        "lr_retune_min_round": 10,
        "lr_retune_cooldown": 8,
        # Gradient-norm LR rescaler knobs
        "lr_trim_enable": 1,
        "lr_trim_ema": 0.9,
        "lr_trim_clip": 4.0,
        "lr_trim_warmup": 5,  # steps after round start before rescaler drives LR
    }

    compulsory = {
        "exclude_server_model_processes": True,
    }

    @classmethod
    def args_update(cls, parser):
        super().args_update(parser)
        parser.add_argument("--tidal_alpha_beta", type=float, default=None)
        parser.add_argument("--tidal_alpha_min", type=float, default=None)
        parser.add_argument("--tidal_coverage_floor", type=float, default=None)
        parser.add_argument("--tidal_score_floor", type=float, default=None)
        parser.add_argument("--tidal_weight_floor", type=float, default=None)
        parser.add_argument("--lr_retune_window", type=int, default=None)
        parser.add_argument("--lr_retune_tol", type=float, default=None)
        parser.add_argument("--lr_retune_sigma", type=float, default=None)
        parser.add_argument("--lr_retune_trials", type=int, default=None)
        parser.add_argument("--lr_retune_eval_epochs", type=int, default=None)
        parser.add_argument("--lr_retune_min_round", type=int, default=None)
        parser.add_argument("--lr_retune_cooldown", type=int, default=None)
        parser.add_argument("--lr_trim_enable", type=int, default=None)
        parser.add_argument("--lr_trim_ema", type=float, default=None)
        parser.add_argument("--lr_trim_clip", type=float, default=None)
        parser.add_argument("--lr_trim_warmup", type=int, default=None)

    def train(self):
        self.run_hpo()
        for client in self.clients:
            client._tidal_init_state()

        for i in range(self.iterations):
            round_start_time = time.time()
            self.current_iter = i

            self.logger.info("")
            self.logger.info(
                f"-------------Round number: {str(self.current_iter).zfill(4)}-------------"
            )

            # Reset grad-norm EMA before each round so aggregation jumps don't corrupt anchor.
            for client in self.clients:
                client.on_round_start()

            if i >= int(self.clients[0].lr_retune_min_round):
                for client in self.clients:
                    client.maybe_retune_lr(round_idx=i)

            self.select_clients()
            self.send_to_clients()

            if self.current_iter % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    self._pre_eval_hook(dataset_type)

            self.pre_train_clients()
            self.train_clients()
            self.receive_from_clients()
            self.calculate_aggregation_weights()
            self.aggregate_models()

            for client in self.clients:
                client._tidal_record_round_loss()

            if self.current_iter % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    if not self.exclude_server_model_processes:
                        self.evaluate_generalization_loss(dataset_type)

            self.metrics["time_per_iter"].append(time.time() - round_start_time)
            _call_if_exists(super(), "log_round_metrics")

        _call_if_exists(super(), "finalize")


# ============================================================== client side


class TidalLR_Client(DFedHPO_Client):

    # ---------------------------------------------------------------- setup

    def _tidal_init_state(self):
        """Called once after the cold-start HPO phase."""
        win = int(self.lr_retune_window)
        self._loss_history = deque(maxlen=win * 2)
        self._last_retune_round = -(10**9)
        self._first_retune_done = False
        self._grad_norm_ema = None
        self._trim_target = None
        self._g_warmup_remaining = 0
        self._hpo_lr = float(self.optimal_config["lr"])
        # Running median anchor: seeded with HPO result.
        self._accepted_lrs = deque(maxlen=10)
        self._accepted_lrs.append(self._hpo_lr)
        # Received neighbor coverages (set by receive_from_server).
        self._neighbor_coverages = None

    def on_round_start(self):
        """Called by TidalLR.train() at the top of every round."""
        self._grad_norm_ema = None
        self._trim_target = None
        self._g_warmup_remaining = int(self.lr_trim_warmup)

    # ------------------------------------------- communication

    def variables_to_be_sent(self):
        payload = super().variables_to_be_sent()
        payload["tidal_coverage"] = float(self._local_coverage())
        return payload

    def receive_from_server(self, data):
        super().receive_from_server(data)
        own_cov = self._local_coverage()
        received = data.get("tidal_coverage")
        n = len(self.scores)
        if received is None or not isinstance(received, (list, tuple)):
            self._neighbor_coverages = [own_cov] + [1.0] * max(n - 1, 0)
        else:
            covs = list(received)
            if len(covs) < n:
                covs = covs + [1.0] * (n - len(covs))
            self._neighbor_coverages = covs[:n]

    # -------------------------------------------------------- aggregator

    def calculate_aggregation_weights(self):
        device = self.device
        n = len(self.scores)

        score_floor = max(float(self.tidal_score_floor), 0.0)
        scores = torch.tensor(self.scores, dtype=torch.float32, device=device)
        scores = torch.clamp(scores, min=score_floor)

        # Use broadcast neighbor coverages if available, else fall back to [own, 1.0, ...].
        own_cov = self._local_coverage()
        if self._neighbor_coverages is not None and len(self._neighbor_coverages) == n:
            cov_vals = [
                max(float(c), float(self.tidal_coverage_floor))
                for c in self._neighbor_coverages
            ]
        else:
            cov_vals = [own_cov] + [1.0] * max(n - 1, 0)
        coverage = torch.tensor(cov_vals, dtype=torch.float32, device=device)

        raw = scores * coverage

        floor = max(float(self.tidal_weight_floor), 0.0)
        if floor > 0.0:
            raw = torch.clamp(raw, min=floor)

        total = raw.sum()
        if float(total.detach().cpu()) <= 0.0:
            self.weights = torch.ones_like(raw) / float(n)
        else:
            self.weights = raw / total

        # Adaptive self-weight: clients with sparse data lean more on neighbors.
        # alpha = sigmoid(beta * (own_coverage - 0.5))  clamped to [alpha_min, 1)
        beta = float(self.tidal_alpha_beta)
        alpha_min = max(float(self.tidal_alpha_min), 0.0)
        alpha = 1.0 / (1.0 + math.exp(-beta * (own_cov - 0.5)))
        alpha = max(alpha, alpha_min)
        alpha = min(alpha, 0.9)

        if n > 1 and float(self.weights[0].detach().cpu()) < alpha:
            neighbor = self.weights[1:]
            denom = neighbor.sum()
            if float(denom.detach().cpu()) > 0.0:
                self.weights[1:] = neighbor / denom * (1.0 - alpha)
            else:
                self.weights[1:] = 0.0
            self.weights[0] = alpha

    def _local_coverage(self):
        stats = getattr(self, "stats", {}).get("average", {})
        count = max(float(stats.get("count", 0.0)), 0.0)
        n_null = max(float(stats.get("n_null", 0.0)), 0.0)
        total = max(count + n_null, 1.0)
        floor = max(float(self.tidal_coverage_floor), 1e-6)
        return max(count / total, floor)

    # -------------------------------------------- plateau detector & retune

    def _tidal_record_round_loss(self):
        loss = self._latest_personal_test_loss()
        if loss is not None:
            self._loss_history.append(float(loss))

    def _latest_personal_test_loss(self):
        m = getattr(self, "metrics", {}) or {}
        for key in ("personal_test_loss", "test_loss", "personal_avg_test_loss"):
            seq = m.get(key)
            if seq and len(seq) > 0:
                return float(seq[-1])
        return None

    def _is_plateau(self):
        win = int(self.lr_retune_window)
        if len(self._loss_history) < win + 1:
            return False
        hist = np.array(list(self._loss_history))
        # Median-of-deltas + MAD test: plateau if median descent and spread are both tiny.
        deltas = np.diff(hist[-(win + 1) :])  # win delta values
        med = float(np.median(deltas))
        mad = float(np.median(np.abs(deltas - med)))
        tol = float(self.lr_retune_tol)
        # Plateau: median improvement is small AND spread (MAD) is small.
        return med > -tol and mad < tol

    def maybe_retune_lr(self, round_idx):
        cooldown = int(self.lr_retune_cooldown)
        win = int(self.lr_retune_window)

        # Mandatory first probe fires once at round (window + 5).
        if not self._first_retune_done and round_idx >= win + 5:
            self._do_retune(round_idx)
            return

        if round_idx - self._last_retune_round < cooldown:
            return
        if not self._is_plateau():
            return

        self._do_retune(round_idx)

    def _do_retune(self, round_idx):
        self._first_retune_done = True

        lr_curr = float(self.optimal_config.get("lr", self.configs.learning_rate))
        sigma = max(float(self.lr_retune_sigma), 1.0 + 1e-6)
        lr_min = max(lr_curr / sigma, 1e-12)
        lr_max = max(lr_curr * sigma, lr_min * (1.0 + 1e-6))

        initial_state = copy.deepcopy(self.model.state_dict())

        saved_eval_epochs = int(self.eval_epochs)
        self.eval_epochs = max(int(self.lr_retune_eval_epochs), 1)

        trials = max(int(self.lr_retune_trials), 1)
        log_lo, log_hi = math.log(lr_min), math.log(lr_max)

        try:
            # Evaluate incumbent from clean snapshot.
            self.model.load_state_dict(initial_state)
            incumbent_loss = self._evaluate_config({"lr": lr_curr}, initial_state)

            best_loss = incumbent_loss
            best_lr = lr_curr

            for _ in range(trials):
                # Each candidate starts from the same clean snapshot.
                self.model.load_state_dict(initial_state)
                lr = float(np.exp(np.random.uniform(log_lo, log_hi)))
                loss = self._evaluate_config({"lr": lr}, initial_state)
                if loss < best_loss:
                    best_loss = loss
                    best_lr = lr
        finally:
            self.eval_epochs = saved_eval_epochs
            self.model.load_state_dict(initial_state)

        self._last_retune_round = round_idx
        self._loss_history.clear()

        if best_lr == lr_curr:
            self.logger.info(
                f"LR retune @ r{round_idx}: incumbent {lr_curr:.3e} "
                f"(loss {incumbent_loss:.4f}) wins, no change."
            )
            return

        self.optimal_config = {"lr": best_lr}
        for pg in self.optimizer.param_groups:
            pg["lr"] = best_lr
        self._hpo_lr = best_lr
        self._accepted_lrs.append(best_lr)
        # Reset rescaler so it re-anchors to the new LR regime.
        self._trim_target = None
        self._grad_norm_ema = None
        self.logger.info(
            f"LR retune @ r{round_idx}: {lr_curr:.3e} -> {best_lr:.3e} "
            f"(loss {incumbent_loss:.4f} -> {best_loss:.4f})"
        )

    # ------------------------------------------ gradient-norm LR rescaler

    def train_one_epoch(
        self,
        model,
        dataloader,
        optimizer,
        criterion,
        scheduler,
        device,
        offload_after=True,
    ):
        # HPO probe path uses _NoOpScheduler; bypass rescaler so probe losses are
        # apples-to-apples with DFedHPO.
        if not int(self.lr_trim_enable) or isinstance(scheduler, _NoOpScheduler):
            return super().train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                device=device,
                offload_after=offload_after,
            )

        model.to(device)
        self._move_optimizer_state_to_param_devices(optimizer)
        model.train()

        ema_alpha = float(self.lr_trim_ema)
        clip = max(float(self.lr_trim_clip), 1.0)
        # Anchor from running median of accepted LRs (not the potentially stale HPO value).
        lr_anchor = (
            float(np.median(list(self._accepted_lrs)))
            if self._accepted_lrs
            else float(self._hpo_lr)
        )
        lo = lr_anchor / clip
        hi = lr_anchor * clip

        warmup_remaining = int(getattr(self, "_g_warmup_remaining", 0))

        for batch in dataloader:
            x, y, x_mark, y_mark = _unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x, x_mark=x_mark, y_mark=y_mark)
            loss = criterion(out, y)
            loss.backward()

            gnorm = _param_grad_norm(model)
            if gnorm > 0.0 and math.isfinite(gnorm):
                if self._grad_norm_ema is None:
                    self._grad_norm_ema = gnorm
                else:
                    self._grad_norm_ema = (
                        ema_alpha * self._grad_norm_ema + (1.0 - ema_alpha) * gnorm
                    )

                if warmup_remaining > 0:
                    # During warmup: update target but don't drive LR yet.
                    self._trim_target = lr_anchor * self._grad_norm_ema
                    warmup_remaining -= 1
                else:
                    if self._trim_target is None:
                        self._trim_target = lr_anchor * self._grad_norm_ema
                    live_lr = self._trim_target / max(self._grad_norm_ema, 1e-12)
                    live_lr = float(min(max(live_lr, lo), hi))
                    for pg in optimizer.param_groups:
                        pg["lr"] = live_lr

            optimizer.step()
            scheduler.step()

        self._g_warmup_remaining = warmup_remaining

        if offload_after:
            model.to("cpu")


# ============================================================= module helpers


def _call_if_exists(obj, method_name):
    fn = getattr(obj, method_name, None)
    if callable(fn):
        fn()


def _param_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += float(torch.sum(p.grad.detach() ** 2).item())
    return math.sqrt(total)


def _unpack_batch(batch, device):
    if isinstance(batch, (list, tuple)) and len(batch) >= 4:
        x, y, x_mark, y_mark = batch[0], batch[1], batch[2], batch[3]
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
        x_mark, y_mark = None, None
    elif isinstance(batch, dict):
        x, y = batch["x"], batch["y"]
        x_mark, y_mark = batch.get("x_mark"), batch.get("y_mark")
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")
    if isinstance(x, torch.Tensor):
        x = x.to(device, non_blocking=True)
    if isinstance(y, torch.Tensor):
        y = y.to(device, non_blocking=True)
    if isinstance(x_mark, torch.Tensor):
        x_mark = x_mark.to(device, non_blocking=True)
    if isinstance(y_mark, torch.Tensor):
        y_mark = y_mark.to(device, non_blocking=True)
    return x, y, x_mark, y_mark
