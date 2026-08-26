from .dFL import dFL, dFL_Client
from .tFL import tFL, tFL_Client


class FedAvg(tFL):
    """Federated Averaging (McMahan et al., AISTATS 2017)."""


class FedAvg_Client(tFL_Client):
    """Use the standard stateless worker."""


class DFedAvg(dFL):
    """Decentralized variant of FedAvg using topology-based gossip averaging."""


class DFedAvg_Client(dFL_Client):
    """Use the decentralized stateless worker."""
