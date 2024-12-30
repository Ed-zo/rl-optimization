from typing import List
from torch_geometric.data import Data
import numpy as np

class GraphGenerator():

    def __init__(self, graphs: List[Data] = None):
        self.graphs = graphs

        self.random = False
        if graphs is None:
            self.random = True

    def to(self, device):
        self.graphs = [graph.to(device) for graph in self.graphs]

    def _generate_random(self):
        pass

    def generate(self):
        if self.random:
            return self._generate_random()

        return self.graphs[np.random.randint(0, len(self.graphs))]