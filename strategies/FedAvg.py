from ._core import StatelessClient, StatelessServer
from .dFL import dFL, dFL_Client


class FedAvg(StatelessServer):
    pass


class FedAvg_Client(StatelessClient):
    pass


class DFedAvg(dFL):
    """Simplest decentralized FedAvg — topology-based neighbor averaging."""


class DFedAvg_Client(dFL_Client):
    pass
