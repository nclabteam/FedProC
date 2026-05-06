import torch

from .dFL import dFL, dFL_Client


class FedSPD(dFL):
    """Soft-clustering personalized DFL adapted for scalar forecasting clients."""

    optional = {
        "fedspd_clusters": 3,
        "fedspd_temperature": 2.0,
        "fedspd_quality_gamma": 1.0,
        "fedspd_profile_lambda": 1.0,
        "fedspd_self_weight": 0.0,
        "fedspd_weight_floor": 1e-8,
    }

    compulsory = {
        "exclude_server_model_processes": True,
    }

    @classmethod
    def args_update(cls, parser):
        super().args_update(parser)
        parser.add_argument("--fedspd_clusters", type=int, default=None)
        parser.add_argument("--fedspd_temperature", type=float, default=None)
        parser.add_argument("--fedspd_quality_gamma", type=float, default=None)
        parser.add_argument("--fedspd_profile_lambda", type=float, default=None)
        parser.add_argument("--fedspd_self_weight", type=float, default=None)
        parser.add_argument("--fedspd_weight_floor", type=float, default=None)


class FedSPD_Client(dFL_Client):
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

    def variables_to_be_sent(self):
        payload = super().variables_to_be_sent()
        payload["fedspd_profile"] = self._local_profile()
        return payload

    def receive_from_server(self, data):
        super().receive_from_server(data)
        self.fedspd_profiles = data.get("fedspd_profile", [self._local_profile()])

    def _local_profile(self):
        stats = self.stats.get("average", {})
        count = max(float(stats.get("count", 0.0)), 0.0)
        n_null = max(float(stats.get("n_null", 0.0)), 0.0)
        n_zero = max(float(stats.get("n_zero", 0.0)), 0.0)
        n_pos = max(float(stats.get("n_pos", 0.0)), 0.0)
        observed = max(count, 1.0)
        total = max(count + n_null, 1.0)

        return {
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
        }

    def _profile_matrix(self):
        rows = []
        for profile in self.fedspd_profiles:
            rows.append([float(profile.get(key, 0.0)) for key in self.profile_keys])
        if not rows:
            rows = [[0.0 for _ in self.profile_keys]]
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _normalized_profiles(self):
        matrix = self._profile_matrix()
        center = matrix.median(dim=0).values
        scale = torch.clamp(torch.std(matrix, dim=0, unbiased=False), min=1.0)
        return (matrix - center) / scale

    def _quality_weights(self):
        scores = torch.tensor(self.scores, dtype=torch.float32, device=self.device)
        scores = torch.clamp(scores, min=1.0)
        gamma = max(float(self.fedspd_quality_gamma), 0.0)
        return torch.pow(scores, gamma)

    def _cluster_centers(self, profiles, num_clusters):
        centers = [profiles[0]]
        while len(centers) < num_clusters:
            stacked = torch.stack(centers)
            distances = _squared_distances(profiles, stacked)
            nearest = torch.min(distances, dim=1).values
            centers.append(profiles[torch.argmax(nearest)])

        centers = torch.stack(centers)
        for _ in range(3):
            distances = _squared_distances(profiles, centers)
            assignments = torch.argmin(distances, dim=1)
            updated = []
            for index in range(num_clusters):
                mask = assignments == index
                if bool(mask.any()):
                    updated.append(profiles[mask].mean(dim=0))
                else:
                    updated.append(centers[index])
            centers = torch.stack(updated)
        return centers

    def _cluster_weights(self):
        profiles = self._normalized_profiles()
        num_clients = profiles.shape[0]
        if num_clients <= 1:
            return torch.ones(num_clients, dtype=torch.float32, device=self.device)

        num_clusters = min(max(int(self.fedspd_clusters), 1), num_clients)
        centers = self._cluster_centers(profiles, num_clusters)
        distances = torch.sqrt(
            torch.clamp(_squared_distances(profiles, centers), min=0.0)
        )
        temperature = max(float(self.fedspd_temperature), 1e-6)
        memberships = torch.softmax(-distances / temperature, dim=1)
        return memberships.matmul(memberships[0])

    def _profile_weights(self):
        profile_lambda = max(float(self.fedspd_profile_lambda), 0.0)
        if profile_lambda == 0:
            return torch.ones(len(self.scores), dtype=torch.float32, device=self.device)

        profiles = self._normalized_profiles()
        distances = torch.mean(torch.abs(profiles - profiles[0]), dim=1)
        return torch.exp(-profile_lambda * distances)

    def calculate_aggregation_weights(self):
        raw_weights = (
            self._quality_weights() * self._cluster_weights() * self._profile_weights()
        )

        floor = max(float(self.fedspd_weight_floor), 0.0)
        if floor > 0:
            raw_weights = torch.clamp(raw_weights, min=floor)

        total = raw_weights.sum()
        if float(total.detach().cpu()) <= 0:
            self.weights = torch.ones_like(raw_weights) / len(raw_weights)
        else:
            self.weights = raw_weights / total

        self_weight = min(max(float(self.fedspd_self_weight), 0.0), 1.0)
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


def _squared_distances(values, centers):
    return torch.sum((values.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
