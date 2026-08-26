from .nFL import nFL, nFL_Client


class LocalOnly(nFL):
    """Local-only training: each client trains independently without any aggregation."""


class LocalOnly_Client(nFL_Client):
    """Use the independent stateless worker."""
