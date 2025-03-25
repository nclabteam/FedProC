import torch

from .base import Client, Server

optional = {
    "beta": 0.2,
}


def args_update(parser):
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Fraction to cut off of both tails of the distribution",
    )


class FedTrimmedAvg(Server):
    """
    Paper: https://arxiv.org/abs/1803.01498
    Source: https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedtrimmedavg.py
    """

    def calculate_aggregation_weights(self):
        pass

    def aggregate_models(self):
        self.model = self.reset_model(self.model)

        lowercut = int(self.beta * len(self.client_data))
        uppercut = len(self.client_data) - lowercut
        if lowercut > uppercut:
            raise ValueError(
                "The fraction to cut off of both tails of the distribution is too large"
            )

        for name, param in self.model.named_parameters():
            layers = torch.stack(
                [client["model"].state_dict()[name] for client in self.client_data]
            )
            # Sort along client axis (axis=0)
            sorted_layers, _ = torch.sort(layers, dim=0)

            # Compute mean excluding trimmed values
            param.data = torch.mean(sorted_layers[lowercut:uppercut], dim=0)


class FedTrimmedAvg_Client(Client):
    def variables_to_be_sent(self):
        return {"model": self.model}
