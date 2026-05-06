import torch
import torch.nn.functional as F

from .dFL import dFL, dFL_Client


class Nahar(dFL):
    """Salinity-profile-gated decentralized aggregation for sparse TSF clients."""

    optional = {
        "topology": "FullyConnected",
        "nahar_tau": 0.5,
        "nahar_quality_gamma": 1.0,
        "nahar_self_weight": 0.0,
        "nahar_similarity_floor": 0.0,
        "nahar_profile_lambda": 1.0,
        "nahar_model_lambda": 0.0,
    }

    compulsory = {
        "exclude_server_model_processes": True,
    }

    @classmethod
    def args_update(cls, parser):
        super().args_update(parser)
        parser.add_argument("--nahar_tau", type=float, default=None)
        parser.add_argument("--nahar_quality_gamma", type=float, default=None)
        parser.add_argument("--nahar_self_weight", type=float, default=None)
        parser.add_argument("--nahar_similarity_floor", type=float, default=None)
        parser.add_argument("--nahar_profile_lambda", type=float, default=None)
        parser.add_argument("--nahar_model_lambda", type=float, default=None)


class Nahar_Client(dFL_Client):
    profile_keys = ("mean", "std", "median", "max", "iqr", "cv", "n_zero")

    def variables_to_be_sent(self):
        payload = super().variables_to_be_sent()
        payload["profile"] = self._local_profile()
        return payload

    def receive_from_server(self, data):
        super().receive_from_server(data)
        self.profiles = data.get("profile", [self._local_profile()])

    def _local_profile(self):
        stats = self.stats.get("average", {})
        count = max(float(stats.get("count", 1.0)), 1.0)
        profile = {key: float(stats.get(key, 0.0)) for key in self.profile_keys}
        profile["n_zero"] = profile["n_zero"] / count
        return profile

    def _profile_vector(self, profile):
        values = [float(profile.get(key, 0.0)) for key in self.profile_keys]
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _flatten_model(self, model):
        tensors = [
            parameter.detach().float().reshape(-1).to(self.device)
            for parameter in model.parameters()
        ]
        return torch.cat(tensors) if tensors else torch.empty(0, device=self.device)

    def _quality_weights(self):
        scores = torch.tensor(self.scores, dtype=torch.float32, device=self.device)
        scores = torch.clamp(scores, min=1.0)
        gamma = max(float(self.nahar_quality_gamma), 0.0)
        return torch.pow(scores, gamma)

    def _profile_weights(self):
        profile_lambda = max(float(self.nahar_profile_lambda), 0.0)
        if profile_lambda == 0 or not getattr(self, "profiles", None):
            return torch.ones(len(self.scores), dtype=torch.float32, device=self.device)

        own = self._profile_vector(self.profiles[0])
        denom = torch.clamp(torch.abs(own), min=1.0)
        distances = []
        for profile in self.profiles:
            other = self._profile_vector(profile)
            distances.append(torch.mean(torch.abs(other - own) / denom))
        distances = torch.stack(distances)
        return torch.exp(-profile_lambda * distances)

    def _model_weights(self):
        model_lambda = max(float(self.nahar_model_lambda), 0.0)
        if model_lambda == 0:
            return torch.ones(len(self.scores), dtype=torch.float32, device=self.device)

        own_vector = self._flatten_model(self.models[0])
        similarities = []
        for model in self.models:
            other_vector = self._flatten_model(model)
            if own_vector.numel() == 0 or other_vector.numel() == 0:
                similarities.append(torch.tensor(1.0, device=self.device))
                continue
            similarity = F.cosine_similarity(
                own_vector.unsqueeze(0),
                other_vector.unsqueeze(0),
            ).squeeze(0)
            similarity = torch.clamp(similarity, min=self.nahar_similarity_floor)
            similarities.append(similarity)
        similarities = torch.stack(similarities)
        tau = max(float(self.nahar_tau), 1e-6)
        return torch.softmax(model_lambda * similarities / tau, dim=0)

    def calculate_aggregation_weights(self):
        raw_weights = (
            self._quality_weights() * self._profile_weights() * self._model_weights()
        )
        total = raw_weights.sum()
        if float(total.detach().cpu()) <= 0:
            self.weights = torch.ones_like(raw_weights) / len(raw_weights)
        else:
            self.weights = raw_weights / total

        self_weight = min(max(float(self.nahar_self_weight), 0.0), 1.0)
        if (
            len(self.weights) > 1
            and float(self.weights[0].detach().cpu()) < self_weight
        ):
            neighbor_weights = self.weights[1:]
            neighbor_total = neighbor_weights.sum()
            if float(neighbor_total.detach().cpu()) > 0:
                self.weights[1:] = neighbor_weights / neighbor_total * (1 - self_weight)
            else:
                self.weights[1:] = 0
            self.weights[0] = self_weight
