from collections import OrderedDict

import torch

from .sFL import sFL, sFL_Client


class Krum(sFL):
    """Krum and Multi-Krum Byzantine-robust aggregation (Blanchard et al., NeurIPS 2017).

    For each client i, compute the sum of squared L2 distances to its
    (n - f - 2) nearest neighbors. Select the client with the lowest score
    (Krum), or average the top-k lowest-score clients (Multi-Krum).

    Set ``num_malicious_clients`` to the known (or upper-bound) number of
    Byzantine workers f. Set ``num_clients_to_keep > 0`` to use Multi-Krum.
    """

    optional = {
        "num_malicious_clients": 0,
        "num_clients_to_keep": 0,
    }

    @classmethod
    def args_update(cls, parser):
        super().args_update(parser)
        parser.add_argument(
            "--num_malicious_clients",
            type=int,
            default=None,
            help="f in Krum: assumed Byzantine count. 0 = derive from --malicious_frac.",
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
        f = self.num_malicious_clients or len(self.malicious_ids)
        num_closest = max(1, num_clients - f - 2)
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
            best_indices = torch.argsort(scores)[: self.num_clients_to_keep]
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


class Krum_Client(sFL_Client):
    pass
