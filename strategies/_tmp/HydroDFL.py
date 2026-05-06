import csv
import math
import os

import torch

from .dFL import dFL, dFL_Client


class HydroDFL(dFL):
    """Hydrological-regime-aware DFL for sparse salinity forecasting."""

    optional = {
        "topology": "FullyConnected",
        "hydro_quality_gamma": 1.0,
        "hydro_profile_lambda": 2.0,
        "hydro_coverage_gamma": 0.25,
        "hydro_geo_lambda": 0.0,
        "hydro_geo_scale_km": 100.0,
        "hydro_river_bonus": 1.0,
        "hydro_self_weight": 0.0,
        "hydro_weight_floor": 1e-8,
    }

    compulsory = {
        "exclude_server_model_processes": True,
    }

    @classmethod
    def args_update(cls, parser):
        super().args_update(parser)
        parser.add_argument("--hydro_quality_gamma", type=float, default=None)
        parser.add_argument("--hydro_profile_lambda", type=float, default=None)
        parser.add_argument("--hydro_coverage_gamma", type=float, default=None)
        parser.add_argument("--hydro_geo_lambda", type=float, default=None)
        parser.add_argument("--hydro_geo_scale_km", type=float, default=None)
        parser.add_argument("--hydro_river_bonus", type=float, default=None)
        parser.add_argument("--hydro_self_weight", type=float, default=None)
        parser.add_argument("--hydro_weight_floor", type=float, default=None)


class HydroDFL_Client(dFL_Client):
    profile_keys = (
        "mean",
        "std",
        "median",
        "q1",
        "q3",
        "iqr",
        "max",
        "cv",
        "zero_rate",
        "null_rate",
        "pos_rate",
    )
    metadata_cache = None

    def variables_to_be_sent(self):
        payload = super().variables_to_be_sent()
        payload["hydro_profile"] = self._local_profile()
        payload["hydro_meta"] = self._local_meta()
        return payload

    def receive_from_server(self, data):
        super().receive_from_server(data)
        self.hydro_profiles = data.get("hydro_profile", [self._local_profile()])
        self.hydro_metas = data.get("hydro_meta", [self._local_meta()])

    def _local_profile(self):
        stats = self.stats.get("average", {})
        count = max(float(stats.get("count", 0.0)), 0.0)
        n_null = max(float(stats.get("n_null", 0.0)), 0.0)
        n_zero = max(float(stats.get("n_zero", 0.0)), 0.0)
        n_pos = max(float(stats.get("n_pos", 0.0)), 0.0)
        observed = max(count, 1.0)
        total = max(count + n_null, 1.0)

        profile = {
            "mean": float(stats.get("mean", 0.0)),
            "std": float(stats.get("std", 0.0)),
            "median": float(stats.get("median", 0.0)),
            "q1": float(stats.get("q1", 0.0)),
            "q3": float(stats.get("q3", 0.0)),
            "iqr": float(stats.get("iqr", 0.0)),
            "max": float(stats.get("max", 0.0)),
            "cv": float(stats.get("cv", 0.0)),
            "zero_rate": n_zero / observed,
            "null_rate": n_null / total,
            "pos_rate": n_pos / observed,
            "coverage": count / total,
        }
        return profile

    def _local_meta(self):
        station_code = self._station_code()
        if not station_code:
            return {}
        return self._metadata().get(station_code, {})

    def _station_code(self):
        source = getattr(self, "private_data", {}).get("file", "")
        if not source:
            return ""
        stem = os.path.splitext(os.path.basename(source))[0]
        return stem.split("_")[0]

    @classmethod
    def _metadata(cls):
        if cls.metadata_cache is not None:
            return cls.metadata_cache

        path = os.path.join(
            "datasets",
            "MekongSalinity",
            "metadata",
            "hydrology_stations.csv",
        )
        records = {}
        try:
            with open(path, newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    code = str(row.get("hydrology_station_code", "")).strip()
                    if not code:
                        continue
                    records[code] = {
                        "longitude": _as_float(row.get("longtitude")),
                        "latitude": _as_float(row.get("latitude")),
                        "river": str(row.get("river_name", "")).strip(),
                    }
        except OSError:
            records = {}

        cls.metadata_cache = records
        return records

    def _profile_vector(self, profile):
        values = [float(profile.get(key, 0.0)) for key in self.profile_keys]
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _quality_weights(self):
        scores = torch.tensor(self.scores, dtype=torch.float32, device=self.device)
        scores = torch.clamp(scores, min=1.0)
        gamma = max(float(self.hydro_quality_gamma), 0.0)
        return torch.pow(scores, gamma)

    def _coverage_weights(self):
        gamma = max(float(self.hydro_coverage_gamma), 0.0)
        if gamma == 0 or not getattr(self, "hydro_profiles", None):
            return torch.ones(len(self.scores), dtype=torch.float32, device=self.device)

        values = [
            max(float(profile.get("coverage", 0.0)), 0.05)
            for profile in self.hydro_profiles
        ]
        coverage = torch.tensor(values, dtype=torch.float32, device=self.device)
        return torch.pow(coverage, gamma)

    def _profile_weights(self):
        profile_lambda = max(float(self.hydro_profile_lambda), 0.0)
        if profile_lambda == 0 or not getattr(self, "hydro_profiles", None):
            return torch.ones(len(self.scores), dtype=torch.float32, device=self.device)

        own = self._profile_vector(self.hydro_profiles[0])
        denom = torch.clamp(torch.abs(own), min=1.0)
        distances = []
        for profile in self.hydro_profiles:
            other = self._profile_vector(profile)
            distances.append(torch.mean(torch.abs(other - own) / denom))
        distances = torch.stack(distances)
        return torch.exp(-profile_lambda * distances)

    def _geo_weights(self):
        geo_lambda = max(float(self.hydro_geo_lambda), 0.0)
        river_bonus = max(float(self.hydro_river_bonus), 0.0)
        if geo_lambda == 0 and river_bonus == 1.0:
            return torch.ones(len(self.scores), dtype=torch.float32, device=self.device)
        if not getattr(self, "hydro_metas", None):
            return torch.ones(len(self.scores), dtype=torch.float32, device=self.device)

        own = self.hydro_metas[0]
        scale = max(float(self.hydro_geo_scale_km), 1.0)
        weights = []
        for meta in self.hydro_metas:
            weight = 1.0
            distance = _haversine_km(own, meta)
            if distance is not None and geo_lambda > 0:
                weight *= math.exp(-geo_lambda * distance / scale)
            if (
                river_bonus != 1.0
                and own.get("river")
                and own.get("river") == meta.get("river")
            ):
                weight *= river_bonus
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def calculate_aggregation_weights(self):
        raw_weights = (
            self._quality_weights()
            * self._profile_weights()
            * self._coverage_weights()
            * self._geo_weights()
        )

        floor = max(float(self.hydro_weight_floor), 0.0)
        if floor > 0:
            raw_weights = torch.clamp(raw_weights, min=floor)

        total = raw_weights.sum()
        if float(total.detach().cpu()) <= 0:
            self.weights = torch.ones_like(raw_weights) / len(raw_weights)
        else:
            self.weights = raw_weights / total

        self_weight = min(max(float(self.hydro_self_weight), 0.0), 1.0)
        if (
            len(self.weights) > 1
            and float(self.weights[0].detach().cpu()) < self_weight
        ):
            neighbor_weights = self.weights[1:]
            neighbor_total = neighbor_weights.sum()
            if float(neighbor_total.detach().cpu()) > 0:
                self.weights[1:] = (
                    neighbor_weights / neighbor_total * (1.0 - self_weight)
                )
            else:
                self.weights[1:] = 0.0
            self.weights[0] = self_weight


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _haversine_km(a, b):
    lat1 = a.get("latitude")
    lon1 = a.get("longitude")
    lat2 = b.get("latitude")
    lon2 = b.get("longitude")
    if None in (lat1, lon1, lat2, lon2):
        return None

    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    term = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.atan2(math.sqrt(term), math.sqrt(1.0 - term))
