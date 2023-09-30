import torch
from torch_geometric.data import Data

MAX_VEHICLES = 50

class Env:
    def __init__(self, graph: Data, device = 'cpu'):
        self.graph = graph
        self.device = device
        self.vehicleID = 1

        self.reset()

    def reset(self):
        self.graph.x[:, 6] = 0
        self.vehicleID = 1

    def compute_objective_function(self):

    # Return next state, mask, reward, and terminal state
    def step(self, action) -> (torch.Tensor, torch.Tensor):
        self.graph.x[action, 6] = 1
        mask = self.graph.x[:, 6] == 0 

        reward = 0 
        terminal = ~torch.any(mask).item()

        return self.graph, mask, reward, terminal
        