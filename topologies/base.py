class Topology:
    """Base class for generated client-neighbor maps."""

    def __init__(self, num_nodes: int) -> None:
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        self.num_nodes = num_nodes
        self.neighbors = self._gen()

    def get_neighbors(self, node_id: int) -> list[int]:
        return self.neighbors[node_id]

    def get_connectivity_info(self) -> dict[str, str | int | float]:
        """Summarize the generated connectivity."""
        total_connections = sum(len(neighbors) for neighbors in self.neighbors.values())
        return {
            "topology_type": type(self).__name__,
            "num_nodes": self.num_nodes,
            "total_connections": total_connections,
            "avg_connections_per_node": total_connections / self.num_nodes,
        }

    def _gen(self) -> dict[int, list[int]]:
        raise NotImplementedError
