from typing import Any

from .mFL import mFL, mFL_Client


class PerAvg(mFL):
    """Per-FedAvg shared meta-initialization with one-step personalization."""

    optional = {"beta": 1e-3, "hf": False, "delta": 1e-3}

    @classmethod
    def args_update(cls, parser: Any) -> Any:
        parser.add_argument("--beta", type=float, default=None)
        parser.add_argument("--hf", action="store_true", default=None)
        parser.add_argument("--delta", type=float, default=None)
        return parser


class PerAvg_Client(mFL_Client):
    """Use the shared meta-learning worker."""
