from .dFL import dFL, dFL_Client
from .tFL import tFL, tFL_Client


class FedAvg(tFL):
    """Federated Averaging (McMahan et al., AISTATS 2017).

    Each round: sample a fraction C of clients, run E local SGD epochs with
    mini-batch size B, then aggregate by weighted average w ∝ n_k (local
    dataset size). Aggregation and training loop are handled by tFL.

    Reference: arXiv:1602.05629.
    """


class FedAvg_Client(tFL_Client):
    pass


class DFedAvg(dFL):
    """Decentralized variant of FedAvg using topology-based gossip averaging.

    Clients train locally (same as FedAvg), then perform one gossip round
    with their topology neighbors (uniform weight). No central server aggregation.
    """


class DFedAvg_Client(dFL_Client):
    pass
