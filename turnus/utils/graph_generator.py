
import numpy as np
from utils.graph_utils import load_problem
import torch
from torch_geometric.data import Data

class GraphGenerator:
    def __init__(self, path='data', problems = [10], size = 100):
        self.size = size

        self.problems = []
        for problem in problems:
            graph, opt = load_problem(f'{path}/{problem}')
            self.problems.append(graph)

    def generate(self) -> Data:
        graph: Data = self.problems[np.random.choice(len(self.problems))]

        subgraph_nodes = torch.randperm(graph.num_nodes - 2) # -2 because we want to exclude the depot nodes
        subgraph_nodes = subgraph_nodes + 1 # Add 1 to exclude the starting depo
        subgraph_nodes = torch.cat((subgraph_nodes[:self.size - 2], torch.tensor([0, graph.num_nodes - 1]))) # Add the depot nodes
        subgraph_nodes = subgraph_nodes.unique()    # This mainly sorts the nodes

        sub_graph = graph.subgraph(subgraph_nodes)
        
        assert sub_graph.has_isolated_nodes() == False, "Cant have isolated nodes" # Make sure there are no isolated nodes

        return sub_graph

        