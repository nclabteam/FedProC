from .base import Topology


class Ring(Topology):
    def _gen(self):
        """Generate ring topology where each node connects to its immediate neighbors."""
        neighbors = {}
        for node in range(self.num_nodes):
            neighbors[node] = []
            # Add previous neighbor (wrap around for node 0)
            prev_node = (node - 1) % self.num_nodes
            neighbors[node].append(prev_node)
            # Add next neighbor (wrap around for last node)
            next_node = (node + 1) % self.num_nodes
            neighbors[node].append(next_node)
        return neighbors
