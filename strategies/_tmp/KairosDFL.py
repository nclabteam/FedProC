"""KairosDFL: Directed lagged-correlation dynamic topology for sparse hydrology DFL.

Motivated by tidal intrusion wave propagation in river deltas: coastal stations
change first (A), then mid-branch (B), then inland (C). Station A's current
observations are predictive of C's near-future observations — a directional,
time-lagged relationship that symmetric aggregation kernels miss entirely.

Two mechanisms layered on top of DFedHPO (pre-trained LR baseline):
  1. Static directed topology prior from lagged cross-correlation estimation.
     Each client broadcasts its recent z-scored time series as a fingerprint.
     Receiver i estimates the directed Pearson correlation between j's series
     lagged by s steps and i's unlagged series (j leads i), for s in [0, lag_max].
     Only the positive-lag direction is counted (j can lead i, not assumed symmetric).
  2. Dynamic regime-state weighting updated every round: each client broadcasts
     a short window of recent z-scored observations; the aggregation weight uses
     cosine similarity between receiver and neighbor windows to measure tidal
     phase alignment.

Aggregation weight for neighbor j at receiver i:
  w_ij ∝ score_j * coverage_j * lag_corr_ij * cos_regime(v_i, v_j)

Communication overhead:
  Discovery phase: fp_len floats per client (one-time; fp_len ~ lag_max + 30).
  Per round: regime_window floats per client (window vector, not scalar).
  No raw data shared outside the fingerprint window.
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .DFedHPO import DFedHPO, DFedHPO_Client

# ============================================================== server side


class KairosDFL(DFedHPO):
    """DFedHPO with directed lagged-correlation topology and dynamic regime weighting."""

    optional = {
        "topology": "FullyConnected",
        # DFedHPO cold-start HPO knobs
        "trials": 10,
        "eval_epochs": 3,
        "aggregator": "FA",
        "top_k": 3,
        "lr_min": 1e-5,
        "lr_max": 1e-1,
        # Discovery phase knobs
        "kairos_lag_max": 7,
        "kairos_min_lag_obs": 10,
        "kairos_corr_threshold": 0.3,
        "kairos_corr_floor": 1e-4,
        "kairos_fp_len": 0,  # 0 = auto: lag_max + min_lag_obs + 10
        "kairos_lag_refresh": 0,  # 0 = one-time only; N > 0 = refresh every N rounds
        # Aggregation knobs
        "kairos_gamma": 2.0,
        "kairos_coverage_floor": 0.05,
        "kairos_self_weight": 0.1,
        "kairos_score_floor": 1.0,
        "kairos_weight_floor": 1e-8,
        # Regime state knob
        "kairos_regime_window": 7,
    }

    compulsory = {
        "exclude_server_model_processes": True,
    }

    @classmethod
    def args_update(cls, parser):
        super().args_update(parser)
        parser.add_argument("--kairos_lag_max", type=int, default=None)
        parser.add_argument("--kairos_min_lag_obs", type=int, default=None)
        parser.add_argument("--kairos_corr_threshold", type=float, default=None)
        parser.add_argument("--kairos_corr_floor", type=float, default=None)
        parser.add_argument("--kairos_fp_len", type=int, default=None)
        parser.add_argument("--kairos_lag_refresh", type=int, default=None)
        parser.add_argument("--kairos_gamma", type=float, default=None)
        parser.add_argument("--kairos_coverage_floor", type=float, default=None)
        parser.add_argument("--kairos_self_weight", type=float, default=None)
        parser.add_argument("--kairos_score_floor", type=float, default=None)
        parser.add_argument("--kairos_weight_floor", type=float, default=None)
        parser.add_argument("--kairos_regime_window", type=int, default=None)

    def train(self):
        # Step 1: Cold-start HPO (identical to DFedHPO).
        self.run_hpo()

        # Step 2: Lag discovery phase (one-time, before round 0).
        self.logger.info("--- KairosDFL: lag discovery phase ---")
        self._run_lag_discovery()
        self.logger.info("--- KairosDFL: lag discovery complete ---")

        lag_refresh = (
            int(getattr(self.clients[0], "kairos_lag_refresh", 0))
            if self.clients
            else 0
        )

        # Step 3: DFL training loop (replicated from DFL.train to support lag refresh).
        for i in range(self.iterations):
            round_start_time = time.time()
            self.current_iter = i

            self.logger.info("")
            self.logger.info(
                f"-------------Round number: {str(self.current_iter).zfill(4)}-------------"
            )

            if lag_refresh > 0 and i > 0 and i % lag_refresh == 0:
                self.logger.info(f"--- KairosDFL: lag refresh at round {i} ---")
                self._run_lag_discovery()

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

            if self.current_iter % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    if not self.exclude_server_model_processes:
                        self.evaluate_generalization_loss(dataset_type)

            self.metrics["time_per_iter"].append(time.time() - round_start_time)
            _call_if_exists(super(), "log_round_metrics")

        _call_if_exists(super(), "finalize")

    def _run_lag_discovery(self):
        # Step 1: each client computes its z-scored fingerprint series.
        for client in self.clients:
            client.kairos_fingerprint = client._compute_fingerprint()

        # Step 2: server relays fingerprints to each client's neighbors.
        all_fps = {c.id: c.kairos_fingerprint for c in self.clients}
        for client in self.clients:
            neighbor_fps = {
                nid: all_fps[nid]
                for nid in client._kairos_neighbor_ids
                if nid in all_fps
            }
            client._receive_fingerprints(neighbor_fps)

        # Step 3: each client estimates directed lag correlations.
        for client in self.clients:
            client._estimate_lag_correlations()
            for nid, corr in client.kairos_lag_corr.items():
                self.logger.info(
                    f"  Client {client.id} <- Client {nid}: "
                    f"lag_corr={corr:.3f}, best_lag={client.kairos_best_lag.get(nid, -1)}"
                )


# ============================================================== client side


class KairosDFL_Client(DFedHPO_Client):

    # -------------------------------------------------------- communication

    def variables_to_be_sent(self):
        payload = super().variables_to_be_sent()
        # Broadcast window vector for regime-phase matching.
        payload["kairos_state"] = self._compute_regime_state()
        return payload

    def receive_from_server(self, data):
        super().receive_from_server(data)
        own_state = self._compute_regime_state()
        received = data.get("kairos_state")
        n = len(self.scores)

        if received is None or not isinstance(received, (list, tuple)):
            self.kairos_states = [own_state] * n
            return

        parsed = []
        for s in received:
            if isinstance(s, (list, tuple)):
                parsed.append(list(s))
            elif isinstance(s, (int, float)):
                # Scalar fallback: expand into a constant vector.
                win = int(self.kairos_regime_window)
                parsed.append([float(s)] * win)
            else:
                parsed.append(own_state)

        while len(parsed) < n:
            parsed.append(own_state)
        self.kairos_states = parsed[:n]

    # --------------------------------------------------------- aggregation

    def calculate_aggregation_weights(self):
        device = self.device
        n = len(self.scores)

        score_floor = max(float(self.kairos_score_floor), 0.0)
        scores = torch.tensor(self.scores, dtype=torch.float32, device=device)
        scores = torch.clamp(scores, min=score_floor)

        # Coverage: broadcast own; neighbors' values come from received payload.
        own_cov = self._local_coverage()
        neighbor_covs = list(getattr(self, "_kairos_neighbor_covs", []))
        cov_vals = [own_cov]
        for k in range(1, n):
            if k - 1 < len(neighbor_covs):
                cov_vals.append(
                    max(float(neighbor_covs[k - 1]), float(self.kairos_coverage_floor))
                )
            else:
                cov_vals.append(1.0)
        coverage = torch.tensor(cov_vals, dtype=torch.float32, device=device)

        lag_w = self._lag_weight_vector(n, device)
        regime_w = self._regime_weight_vector(n, device)

        raw = scores * coverage * lag_w * regime_w

        floor = max(float(self.kairos_weight_floor), 0.0)
        if floor > 0.0:
            raw = torch.clamp(raw, min=floor)

        total = raw.sum()
        if float(total.detach().cpu()) <= 0.0:
            self.weights = torch.ones_like(raw) / float(n)
        else:
            self.weights = raw / total

        # Fixed self-weight floor.
        alpha = min(max(float(self.kairos_self_weight), 0.0), 1.0)
        if n > 1 and alpha > 0.0 and float(self.weights[0].detach().cpu()) < alpha:
            neighbor = self.weights[1:]
            denom = neighbor.sum()
            if float(denom.detach().cpu()) > 0.0:
                self.weights[1:] = neighbor / denom * (1.0 - alpha)
            else:
                self.weights[1:] = 0.0
            self.weights[0] = alpha

    def _lag_weight_vector(self, n, device):
        corr_threshold = float(self.kairos_corr_threshold)
        corr_floor = max(float(self.kairos_corr_floor), 1e-12)

        values = [1.0]  # index 0 = self; self is perfectly predictive of self
        lag_corr_map = getattr(self, "kairos_lag_corr", {})

        for nid in self._kairos_neighbor_ids:
            corr = float(lag_corr_map.get(nid, 0.0))
            values.append(
                max(corr, corr_floor) if corr >= corr_threshold else corr_floor
            )

        while len(values) < n:
            values.append(corr_floor)
        values = values[:n]

        return torch.tensor(values, dtype=torch.float32, device=device)

    def _regime_weight_vector(self, n, device):
        """Cosine similarity between own and neighbor regime windows."""
        gamma = max(float(self.kairos_gamma), 0.0)
        if gamma == 0.0:
            return torch.ones(n, dtype=torch.float32, device=device)

        win = int(self.kairos_regime_window)
        own_state = self._compute_regime_state()
        states = list(getattr(self, "kairos_states", [own_state] * n))
        while len(states) < n:
            states.append(own_state)
        states = states[:n]

        own_vec = torch.tensor(own_state, dtype=torch.float32, device=device)
        if len(own_vec) < win:
            own_vec = F.pad(own_vec, (0, win - len(own_vec)))
        else:
            own_vec = own_vec[:win]
        own_norm = own_vec.norm()

        weights = []
        for state in states:
            if isinstance(state, (list, tuple)):
                vec = torch.tensor(list(state), dtype=torch.float32, device=device)
            else:
                vec = own_vec.clone()
            if len(vec) < win:
                vec = F.pad(vec, (0, win - len(vec)))
            else:
                vec = vec[:win]

            v_norm = vec.norm()
            if own_norm < 1e-10 or v_norm < 1e-10:
                cos_sim = 1.0
            else:
                cos_sim = float((own_vec * vec).sum() / (own_norm * v_norm))
                cos_sim = max(min(cos_sim, 1.0), -1.0)

            # Map cosine distance [0, 2] → weight via exponential decay.
            dist = 1.0 - cos_sim
            weights.append(math.exp(-gamma * dist))

        return torch.tensor(weights, dtype=torch.float32, device=device)

    # ------------------------------------------------- fingerprint / discovery

    @property
    def _kairos_neighbor_ids(self):
        """Neighbor IDs (excluding self), matching indices 1..n-1 in self.scores."""
        neighbors = list(getattr(self, "neighbors", []))
        return [nid for nid in neighbors if nid != self.id]

    def _compute_fingerprint(self):
        """Recent z-scored time series for directed cross-correlation estimation."""
        lag_max = int(self.kairos_lag_max)
        min_obs = int(self.kairos_min_lag_obs)
        fp_len = int(self.kairos_fp_len)
        if fp_len <= 0:
            fp_len = lag_max + min_obs + 10

        series = self._local_series()
        if len(series) == 0:
            return [float("nan")] * fp_len

        non_null = series[~np.isnan(series)]
        if len(non_null) < 2:
            return [float("nan")] * min(len(series), fp_len)

        mean = float(np.mean(non_null))
        std = float(np.std(non_null))
        if std < 1e-10:
            return [0.0] * min(len(series), fp_len)

        z = (series - mean) / std
        if len(z) > fp_len:
            z = z[-fp_len:]

        return z.tolist()

    def _receive_fingerprints(self, neighbor_fps):
        self.kairos_neighbor_fps = dict(neighbor_fps)

    def _estimate_lag_correlations(self):
        """Estimate peak directed Pearson correlation for each neighbor (j leads i)."""
        lag_max = int(self.kairos_lag_max)
        min_obs = int(self.kairos_min_lag_obs)

        own_fp = np.array(getattr(self, "kairos_fingerprint", []), dtype=np.float64)
        self.kairos_lag_corr = {}
        self.kairos_best_lag = {}

        for nid, neighbor_fp in getattr(self, "kairos_neighbor_fps", {}).items():
            nbr_fp = np.array(neighbor_fp, dtype=np.float64)
            n = min(len(own_fp), len(nbr_fp))
            if n == 0:
                self.kairos_lag_corr[nid] = 0.0
                self.kairos_best_lag[nid] = 0
                continue

            z_i = own_fp[:n]
            z_j = nbr_fp[:n]

            best_corr = 0.0
            best_lag = 0

            for s in range(lag_max + 1):
                # j leads i by s steps: align j[0:n-s] with i[s:n]
                if s == 0:
                    a, b = z_i, z_j
                else:
                    if n <= s:
                        continue
                    a = z_i[s:]  # receiver i: from offset s onward
                    b = z_j[: n - s]  # neighbor j: up to offset n-s

                min_len = min(len(a), len(b))
                if min_len == 0:
                    continue
                a, b = a[:min_len], b[:min_len]
                mask = ~(np.isnan(a) | np.isnan(b))
                if mask.sum() < min_obs:
                    continue

                xv, yv = a[mask], b[mask]
                xstd = float(np.std(xv))
                ystd = float(np.std(yv))
                if xstd < 1e-10 or ystd < 1e-10:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(xv, yv)[0, 1])
                    if not math.isfinite(corr):
                        corr = 0.0

                if corr > best_corr:
                    best_corr = corr
                    best_lag = s

            self.kairos_lag_corr[nid] = max(best_corr, 0.0)
            self.kairos_best_lag[nid] = best_lag

    # -------------------------------------------------- regime state

    def _compute_regime_state(self):
        """Return a short window of recent z-scored observations for phase matching."""
        series = self._local_series()
        win = int(self.kairos_regime_window)

        non_null = series[~np.isnan(series)]
        if len(non_null) == 0:
            return [0.0] * win

        mean = float(np.mean(non_null))
        std = float(np.std(non_null)) if len(non_null) > 1 else 1.0
        if std < 1e-10:
            return [0.0] * win

        recent = non_null[-win:] if len(non_null) >= win else non_null
        z_recent = ((recent - mean) / std).tolist()

        # Pad front with zeros (neutral/mean) if fewer than win observations.
        while len(z_recent) < win:
            z_recent = [0.0] + z_recent

        return z_recent

    def _local_series(self):
        """Extract local training time series as a chronological 1-D numpy array."""
        # Try to access raw series from dataset directly (avoids shuffled DataLoader).
        try:
            loader = self.load_train_data()
            dataset = loader.dataset
            for attr in ("raw_series", "y_raw", "y", "targets", "data"):
                if hasattr(dataset, attr):
                    raw = getattr(dataset, attr)
                    if isinstance(raw, (np.ndarray, list)):
                        arr = np.array(raw, dtype=np.float32).reshape(-1)
                        return np.where(np.abs(arr) > 1e10, np.nan, arr)
                    elif isinstance(raw, torch.Tensor):
                        arr = raw.detach().cpu().numpy().reshape(-1).astype(np.float32)
                        return np.where(np.abs(arr) > 1e10, np.nan, arr)
        except Exception:
            pass

        # Last resort: reconstruct from per-batch targets (not chronological, but usable
        # for computing marginal statistics like mean and std).
        try:
            loader = self.load_train_data()
            values = []
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    y = batch[1]
                elif isinstance(batch, dict):
                    y = batch["y"]
                else:
                    continue
                if isinstance(y, torch.Tensor):
                    vals = y.detach().cpu().numpy().reshape(-1)
                    vals = np.where(np.abs(vals) > 1e10, np.nan, vals)
                    values.extend(vals.tolist())
            if values:
                return np.array(values, dtype=np.float32)
        except Exception:
            pass

        stats = getattr(self, "stats", {}).get("average", {})
        mean_val = float(stats.get("mean", 0.0))
        count = int(stats.get("count", 1))
        return np.full(count, mean_val, dtype=np.float32)

    def _local_coverage(self):
        stats = getattr(self, "stats", {}).get("average", {})
        count = max(float(stats.get("count", 0.0)), 0.0)
        n_null = max(float(stats.get("n_null", 0.0)), 0.0)
        total = max(count + n_null, 1.0)
        floor = max(float(self.kairos_coverage_floor), 1e-6)
        return max(count / total, floor)


# ============================================================= module helpers


def _call_if_exists(obj, method_name):
    fn = getattr(obj, method_name, None)
    if callable(fn):
        fn()
