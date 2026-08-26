from typing import Any

from .sFL import sFL, sFL_Client


class FedMedian(sFL):
    """Coordinate-wise median aggregation (Byzantine-robust)."""

    def aggregate_client_updates(self, packages: Any) -> None:
        self._commit_global(
            new_params=self.coordinate_median(
                models=[p["regular_model_params"] for p in packages.values()]
            )
        )


class FedMedian_Client(sFL_Client):
    """Use the security-aware stateless worker."""
