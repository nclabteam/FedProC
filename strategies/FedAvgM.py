import copy

from .tFL import tFL as Server


class FedAvgM(Server):

    optional = {"server_momentum": 0.9, "server_learning_rate": 0.01}

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--server_momentum", type=float, default=None)
        parser.add_argument("--server_learning_rate", type=float, default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum_vector = None

    def aggregate_models(self):
        prev_model = copy.deepcopy(self.model)

        # Call the base class method (FedAvg)
        super().aggregate_models()  # This will perform the FedAvg aggregation

        # Now apply momentum if applicable
        if self.server_momentum > 0.0:
            # Calculate the pseudo-gradient as the difference between previous and new models
            pseudo_gradient = []
            for curr_param, prev_param in zip(
                self.model.parameters(), prev_model.parameters()
            ):
                pseudo_gradient.append(prev_param.data - curr_param.data)

            # Apply momentum
            if self.momentum_vector is None:
                self.momentum_vector = pseudo_gradient
            else:
                self.momentum_vector = [
                    self.server_momentum * momentum + gradient
                    for momentum, gradient in zip(self.momentum_vector, pseudo_gradient)
                ]

            pseudo_gradient = self.momentum_vector

            # Update the global model with the momentum-adjusted pseudo-gradient
            for global_param, prev_param, grad in zip(
                self.model.parameters(), prev_model.parameters(), pseudo_gradient
            ):
                global_param.data = prev_param.data - self.server_learning_rate * grad
