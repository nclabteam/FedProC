class Topology:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.neighbors = self._gen()

    def _gen(self):
        raise NotImplementedError

    def _plot(self):
        raise NotImplementedError
