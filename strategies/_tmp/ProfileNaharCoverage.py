import torch

from .Nahar import Nahar, Nahar_Client


class ProfileNaharCoverage(Nahar):
    """ProfileNahar plus a mild local coverage-confidence term."""

    optional = {
        **Nahar.optional,
        "nahar_coverage_gamma": 0.25,
    }

    @classmethod
    def args_update(cls, parser):
        super().args_update(parser)
        parser.add_argument("--nahar_coverage_gamma", type=float, default=None)


class ProfileNaharCoverage_Client(Nahar_Client):
    def variables_to_be_sent(self):
        payload = super().variables_to_be_sent()
        payload["coverage"] = self._coverage()
        return payload

    def receive_from_server(self, data):
        super().receive_from_server(data)
        self.coverages = data.get("coverage", [self._coverage()])

    def _coverage(self):
        stats = self.stats.get("average", {})
        count = max(float(stats.get("count", 0.0)), 0.0)
        n_null = max(float(stats.get("n_null", 0.0)), 0.0)
        return count / max(count + n_null, 1.0)

    def _coverage_weights(self):
        gamma = max(float(self.nahar_coverage_gamma), 0.0)
        if gamma == 0 or not getattr(self, "coverages", None):
            return torch.ones(len(self.scores), dtype=torch.float32, device=self.device)

        values = [max(float(coverage), 0.05) for coverage in self.coverages]
        coverage = torch.tensor(values, dtype=torch.float32, device=self.device)
        return torch.pow(coverage, gamma)

    def calculate_aggregation_weights(self):
        raw_weights = (
            self._quality_weights()
            * self._profile_weights()
            * self._model_weights()
            * self._coverage_weights()
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
                self.weights[1:] = (
                    neighbor_weights / neighbor_total * (1.0 - self_weight)
                )
            else:
                self.weights[1:] = 0.0
            self.weights[0] = self_weight
