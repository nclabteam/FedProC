import random

from .base import Topology


class KConnected(Topology):
    def __init__(self, num_nodes: int, k: int = 2):
        """
        Initialize K-Connected topology.

        Args:
            num_nodes (int): Number of nodes in the network
            k (int): Number of connections each node should have
            seed (int): Random seed for reproducible topology generation
        """
        if k >= num_nodes:
            raise ValueError(f"k ({k}) must be less than num_nodes ({num_nodes})")
        if k < 1:
            raise ValueError(f"k ({k}) must be at least 1")

        self.k = k

        super().__init__(num_nodes)

    def _gen(self):
        """Generate k-connected topology where each node has exactly k neighbors."""
        neighbors = {node: [] for node in range(self.num_nodes)}

        # For each node, randomly select k neighbors
        for node in range(self.num_nodes):
            # Get all possible neighbors (all nodes except current node)
            possible_neighbors = list(range(self.num_nodes))
            possible_neighbors.remove(node)

            # Randomly select k neighbors
            selected_neighbors = random.sample(possible_neighbors, self.k)
            neighbors[node] = selected_neighbors

        return neighbors

    def get_connectivity_info(self):
        data = super().get_connectivity_info()
        data["k"] = self.k
        return data
