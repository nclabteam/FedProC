from argparse import Namespace

from .pFL import pFL, pFL_Client


class FedBN(pFL):
    """
    FedBN: Federated Learning on Non-IID Features via Local Batch Normalization.

    Identical to FedAvg except that Batch Normalization parameters (matched
    by 'bn' in the parameter name) are excluded from aggregation and kept
    local.  For models without BN layers this degenerates to standard FedAvg.

    Reference: Li et al., "FedBN: Federated Learning on Non-IID Features via
    Local Batch Normalization", ICLR 2021. arXiv 2102.07623.
    """

    def aggregate_models(self) -> None:
        self.model = self.reset_model(self.model)
        for client, weight in zip(self.client_data, self.weights):
            for (name, global_param), local_param in zip(
                self.model.named_parameters(),
                client["model"].parameters(),
            ):
                if "bn" in name.lower():
                    continue
                global_param.data.add_(
                    local_param.data.to(global_param.device), alpha=weight
                )


class FedBN_Client(pFL_Client):
    def receive_from_server(self, data: dict) -> None:
        for (name, old_param), new_param in zip(
            self.model.named_parameters(),
            data["model"].parameters(),
        ):
            if "bn" not in name.lower():
                old_param.data.copy_(new_param.data)
