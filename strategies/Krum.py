import torch

from .base import Client, Server

optional = {
    "num_malicious_clients": 0,
    "num_clients_to_keep": 0,
}


def args_update(parser):
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
        help="Number of clients to keep before averaging (MultiKrum). Defaults to 0, inthat case classical Krum is applied.",
    )


class Krum(Server):
    """
    Paper: https://arxiv.org/abs/1703.02757
    Source: https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/krum.py
    """

    def calculate_aggregation_weights(self):
        pass

    def aggregate_models(self):
        self.model = self.reset_model(self.model)

        # Create a list of weights
        client_weights = [client["model"].state_dict() for client in self.client_data]

        # Compute distances between vectors
        distance_matrix = self.compute_distances(client_weights)

        # For each client, take the n-f-2 closest parameters vectors
        num_clients = len(client_weights)
        num_closest = max(1, num_clients - self.num_malicious_clients - 2)

        # Sort distances and take the indices of the closest clients (excluding self)
        closest_indices = [
            torch.argsort(distance)[1 : num_closest + 1].tolist()
            for distance in distance_matrix
        ]

        # Compute the score for each client, that is the sum of the distances of the n-f-2 closest parameters vectors
        scores = torch.tensor(
            [
                torch.sum(distance_matrix[i, closest_indices[i]])
                for i in range(len(distance_matrix))
            ],
            device=distance_matrix.device,
        )

        if self.num_clients_to_keep > 0:
            # Choose to_keep clients and return their average (MultiKrum)
            best_indices = torch.argsort(scores, descending=True)[
                -self.num_clients_to_keep :
            ]
            best_clients = [self.client_data[i]["model"] for i in best_indices]

            for name, param in self.model.named_parameters():
                layers = torch.stack(
                    [client.state_dict()[name] for client in best_clients]
                )
                param.data = torch.mean(layers, dim=0).clone()
        else:
            # Index with lowest score
            best_index = torch.argmin(scores)
            for global_param, param in zip(
                self.model.parameters(), client_weights[best_index].values()
            ):
                global_param.data = param.clone()

    def compute_distances(self, weights: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute distances between model weight vectors.

        Input: weights - list of model state dicts (each representing a client's model)
        Output: distance_matrix - matrix of squared distances between the models
        """
        # Flatten and concatenate all layers for each model into a single vector
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
                delta = flat_w_i - flat_w_j
                norm = torch.norm(delta, p=2)
                distance_matrix[i, j] = norm**2  # Squared Euclidean distance

        return distance_matrix


class Krum_Client(Client):
    def variables_to_be_sent(self):
        return {"model": self.model}
