import random

from .base import Topology


class KConnected(Topology):
    """Select a fixed number of distinct neighbors per node."""

    def __init__(self, num_nodes: int, k: int = 2) -> None:
        if k >= num_nodes:
            raise ValueError(f"k ({k}) must be less than num_nodes ({num_nodes})")
        if k < 1:
            raise ValueError(f"k ({k}) must be at least 1")

        self.k = k
        super().__init__(num_nodes=num_nodes)

    def _gen(self) -> dict[int, list[int]]:
        return {
            node: random.sample(
                population=[peer for peer in range(self.num_nodes) if peer != node],
                k=self.k,
            )
            for node in range(self.num_nodes)
        }

    def get_connectivity_info(self) -> dict[str, str | int | float]:
        data = super().get_connectivity_info()
        data["k"] = self.k
        return data
