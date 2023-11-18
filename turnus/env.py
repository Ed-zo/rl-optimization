import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import torch_geometric.transforms as T

MAX_VEHICLES = 50



class Env:
    def __init__(self, graph: Data, device = 'cpu'):
        self.graph = graph.clone()
        self.device = device
        self.vehicleID = 1
        self.last_visited_node = 0

        # Add env state to the graph
        add_visited_flag = T.Constant(0)
        self.graph = add_visited_flag(self.graph)

        self.visited_flag_index = self.graph.num_node_features - 1

        self.reset()

    def action_space(self) -> int:
        return self.graph.num_nodes

    def state_space(self) -> int:
        return self.graph.num_node_features

    def reset(self):
        self.graph.x[:, self.visited_flag_index] = 0
        self.vehicleID = 1
        self.last_visited_node = 0

        mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool)

        # Mask all nodes that are not connected to the current node
        _, edge_index, _, _ = k_hop_subgraph(0, 1, self.graph.edge_index, flow='target_to_source')
        mask[edge_index[1]] = 1

        return self.graph, mask


    # Return next state, mask, reward, and terminal state
    def step(self, action) -> (torch.Tensor, torch.Tensor):
        if action != self.graph.num_nodes - 1:
            self.graph.x[action, self.visited_flag_index] = self.vehicleID / MAX_VEHICLES
        
        self.last_visited_node = action

        reward = 0 
        terminal = False
        # The vehicle has reached the depot
        if action == self.graph.num_nodes - 1:
            self.vehicleID += 1
            self.last_visited_node = 0

            # Maybe reaching the terminal state
            terminal = True

        mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool)

        # Mask all nodes that are not connected to the current node
        _, edge_index, _, _ = k_hop_subgraph(self.last_visited_node, 1, self.graph.edge_index, flow='target_to_source')
        mask[edge_index[1]] = 1

        # Mask all nodes that are already visited
        visited_mask = self.graph.x[:, self.visited_flag_index] > 0
        mask[visited_mask] = 0

        # All nodes are visited (except the depots)
        terminal = terminal & (visited_mask.sum() == self.graph.num_nodes - 2).item()

        if terminal:
            reward = -(self.vehicleID - 1)


        return self.graph.clone(), mask, reward, terminal, None
        