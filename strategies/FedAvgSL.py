"""FedAvg on the stateless v1.12 core — parity prototype.

Proves the stateless-client / server-owned-state core reproduces legacy
``FedAvg`` numerics. Compare its parity golden against ``FedAvg``'s.
"""

from ._core import StatelessClient, StatelessServer


class FedAvgSL(StatelessServer):
    pass


class FedAvgSL_Client(StatelessClient):
    pass
