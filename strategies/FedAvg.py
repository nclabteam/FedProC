from .dFL import dFL, dFL_Client
from .tFL import tFL, tFL_Client


class FedAvg(tFL):
    pass


class FedAvg_Client(tFL_Client):
    pass


class DFedAvg(dFL):
    """Simplest decentralized FedAvg — topology-based neighbor averaging."""


class DFedAvg_Client(dFL_Client):
    pass
