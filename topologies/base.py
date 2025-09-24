class Topology:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.neighbors = self._gen()

    def get_neighbors(self, id):
        return self.neighbors[id]

    def get_connectivity_info(self):
        """Return information about the topology connectivity."""
        return {
            "topology_type": "k-connected",
            "num_nodes": self.num_nodes,
            "total_connections": sum(
                len(neighbors) for neighbors in self.neighbors.values()
            ),
            "avg_connections_per_node": sum(
                len(neighbors) for neighbors in self.neighbors.values()
            )
            / self.num_nodes,
        }

    def _gen(self):
        raise NotImplementedError

    def _plot(self):
        raise NotImplementedError
