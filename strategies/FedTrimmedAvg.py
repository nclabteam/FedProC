from typing import Any

from .sFL import sFL, sFL_Client


class FedTrimmedAvg(sFL):
    """Coordinate-wise trimmed-mean aggregation (Byzantine-robust)."""

    optional = {
        "beta": 0.2,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        super().args_update(parser=parser)
        parser.add_argument(
            "--beta",
            type=float,
            default=None,
            help="Fraction to cut off of both tails of the distribution",
        )

    def aggregate_client_updates(self, packages: Any) -> None:
        self._commit_global(
            new_params=self.coordinate_trimmed_mean(
                models=[p["regular_model_params"] for p in packages.values()],
                beta=self.beta,
            )
        )


class FedTrimmedAvg_Client(sFL_Client):
    """Use the security-aware stateless worker."""
