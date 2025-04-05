from .base import Topology


class FullyConnected(Topology):
    def _gen(self):
        neighbors = {}
        for node in range(self.num_nodes):
            neighbors[node] = list(range(self.num_nodes))
            neighbors[node].remove(node)
        return neighbors

    def get_neighbors(self, id):
        return self.neighbors[id]
