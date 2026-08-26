from .peftFL import peftFL, peftFL_Client


class FedIT(peftFL):
    """FedIT: sample-weighted FedAvg of both LoRA factors (Zhang et al., 2023)."""


class FedIT_Client(peftFL_Client):
    """FedIT worker; both LoRA factors are trainable and shared."""
