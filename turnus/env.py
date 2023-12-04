import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import torch_geometric.transforms as T
from utils import obj_to_reward

class Env:
    def __init__(self, graph: Data, device = 'cpu'):
        self.graph = graph.clone().to(device)
        self.device = device
        self.vehicleID = 1
        self.last_visited_node = 0
        self.MAX_VEHICLES = graph.num_nodes

        # Add env state to the graph
        add_visited_flag = T.Constant(0)
        self.graph = add_visited_flag(self.graph)

        self.visited_flag_index = self.graph.num_node_features - 1

        self.reset()

    def action_space(self) -> int:
        return self.graph.num_nodes + 1

    def state_space(self) -> int:
        return self.graph.num_node_features

    def reset(self):
        self.graph.x[:, self.visited_flag_index] = 0
        self.vehicleID = 1
        self.last_visited_node = 0

        mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)

        # Mask all nodes that are not connected to the current node
        _, edge_index, _, _ = k_hop_subgraph(0, 1, self.graph.edge_index, flow='target_to_source')
        mask[edge_index[1]] = 1

        return self.graph.clone(), mask


    # Return next state, mask, reward, and terminal state
    def step(self, action) -> (torch.Tensor, torch.Tensor):
        if action != self.graph.num_nodes:
            self.graph.x[action, self.visited_flag_index] = self.vehicleID / self.MAX_VEHICLES
        
        self.last_visited_node = action

        reward = 0 
        terminal = False
        # The vehicle has reached the depot
        if action == self.graph.num_nodes:
            self.vehicleID += 1
            self.last_visited_node = 0

            # Maybe reaching the terminal state
            terminal = True

        mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)

        # Mask all nodes that are not connected to the current node
        _, edge_index, _, _ = k_hop_subgraph(self.last_visited_node, 1, self.graph.edge_index, flow='target_to_source')
        mask[edge_index[1]] = 1

        # Mask all nodes that are already visited
        visited_mask = self.graph.x[:, self.visited_flag_index] > 0
        mask[visited_mask] = 0

        # All nodes are visited (except the depots)
        terminal = terminal & (visited_mask.sum() == self.graph.num_nodes - 2).item()

        if self.vehicleID >= self.MAX_VEHICLES:
            terminal = True

        if terminal:
            reward = obj_to_reward(self.vehicleID, 4, self.MAX_VEHICLES)

        return self.graph.clone(), mask, reward, terminal, None
        