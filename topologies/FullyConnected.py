from .base import Topology


class FullyConnected(Topology):
    """Connect every node to every other node."""

    def _gen(self) -> dict[int, list[int]]:
        return {
            node: [peer for peer in range(self.num_nodes) if peer != node]
            for node in range(self.num_nodes)
        }
