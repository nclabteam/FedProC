from collections import OrderedDict

import torch

from ._core import StatelessClient, StatelessServer


class Krum(StatelessServer):

    optional = {
        "num_malicious_clients": 0,
        "num_clients_to_keep": 0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "--num_malicious_clients",
            type=int,
            default=None,
            help="Number of malicious clients in the system",
        )
        parser.add_argument(
            "--num_clients_to_keep",
            type=int,
            default=None,
            help="Number of clients to keep before averaging (MultiKrum). Defaults to 0, in that case classical Krum is applied.",
        )

    def aggregate_client_updates(self, packages):
        client_weights = [p["regular_model_params"] for p in packages.values()]
        distance_matrix = self.compute_distances(client_weights)

        num_clients = len(client_weights)
        num_closest = max(1, num_clients - self.num_malicious_clients - 2)
        closest_indices = [
            torch.argsort(distance)[1 : num_closest + 1].tolist()
            for distance in distance_matrix
        ]
        scores = torch.tensor(
            [
                torch.sum(distance_matrix[i, closest_indices[i]])
                for i in range(len(distance_matrix))
            ],
            device=distance_matrix.device,
        )

        if self.num_clients_to_keep > 0:
            best_indices = torch.argsort(scores, descending=True)[
                -self.num_clients_to_keep :
            ]
            best_clients = [client_weights[i] for i in best_indices]
            new_params = OrderedDict()
            for name in self.public_model_params:
                layers = torch.stack([client[name] for client in best_clients])
                new_params[name] = torch.mean(layers, dim=0).clone()
            self._commit_global(new_params)
        else:
            best_index = int(torch.argmin(scores))
            self._commit_global(client_weights[best_index])

    def compute_distances(self, weights: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute the matrix of squared L2 distances between client weight vectors."""
        flat_w = torch.stack(
            [
                torch.cat([w.flatten() for w in model_weights.values()])
                for model_weights in weights
            ]
        )
        num_models = len(flat_w)
        distance_matrix = torch.zeros((num_models, num_models), device=flat_w.device)
        for i, flat_w_i in enumerate(flat_w):
            for j, flat_w_j in enumerate(flat_w):
                distance_matrix[i, j] = torch.norm(flat_w_i - flat_w_j, p=2) ** 2
        return distance_matrix


class Krum_Client(StatelessClient):
    pass
