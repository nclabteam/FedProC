from .base import Topology


class Ring(Topology):
    """Connect each node to its previous and next node."""

    def _gen(self) -> dict[int, list[int]]:
        return {
            node: [
                (node - 1) % self.num_nodes,
                (node + 1) % self.num_nodes,
            ]
            for node in range(self.num_nodes)
        }
