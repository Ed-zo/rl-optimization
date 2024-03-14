import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import torch_geometric.transforms as T
from utils.graph_generator import GraphGenerator
from utils.graph_utils import add_graph_feature
from utils.utils import obj_to_reward, reward_to_obj

class EnvLocalRewards:
    def __init__(self, graph_generator: GraphGenerator, device = 'cpu'):
        self.device = device
        self.vehicle_ID = 1
        self.vehicle_visited_node_count = 0
        self.last_visited_node = 0
        self.MAX_VEHICLES = graph_generator.size
        self.graph_generator = graph_generator

        self.reset()

    def action_space(self) -> int:
        return self.graph.num_nodes

    def state_space(self) -> int:
        return self.graph.num_node_features
    
    def reward_to_score_transformer(self):
        def transform(reward):
            return reward_to_obj(reward, 2, self.MAX_VEHICLES)
        return transform

    def reset(self):
        self.graph = self.graph_generator.generate()

        self.starting_depo = 0
        self.ending_depo = self.graph.num_nodes - 1

        # Add env state to the graph
        self.graph, self.flag_visited_index = add_graph_feature(self.graph)
        self.graph, self.flag_current_node_index = add_graph_feature(self.graph)

        self.graph.x[:, self.flag_visited_index] = 0

        self.graph.x[:, self.flag_current_node_index] = 0
        self.graph.x[self.starting_depo, self.flag_current_node_index] = 1

        self.vehicleID = 1
        self.last_visited_node = 0

        mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)

        # Mask all nodes that are not connected to the current node
        _, edge_index, _, _ = k_hop_subgraph(0, 1, self.graph.edge_index, flow='target_to_source')
        mask[edge_index[1]] = 1
        mask[self.starting_depo] = 0

        return self.graph.clone(), mask


    # Return next state, mask, reward, and terminal state
    def step(self, action) -> (torch.Tensor, torch.Tensor):
        # We dont need to pick starting depo
        if action == 0:
            raise 'Error'
        
        node_id = action
        reward = 0 
        terminal = False
        # The vehicle has reached the final depot
        if node_id == self.ending_depo:
            self.vehicle_ID += 1
            self.graph.x[self.starting_depo, self.flag_current_node_index] = 1
            self.graph.x[self.last_visited_node, self.flag_current_node_index] = 0
            self.last_visited_node = 0

            reward = -1 + (self.vehicle_visited_node_count / (self.graph.num_nodes - 2))
            # Maybe reaching the terminal state, now it depends if all nodes are visited
            terminal = True
            self.vehicle_visited_node_count = 0
        else:
            self.graph.x[node_id, self.flag_visited_index] = self.vehicle_ID / self.MAX_VEHICLES
            self.graph.x[node_id, self.flag_current_node_index] = 1
            self.graph.x[self.last_visited_node, self.flag_current_node_index] = 0
            self.last_visited_node = node_id
            self.vehicle_visited_node_count += 1

        action_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)

        # Mask all nodes that are not connected to the current node
        _, edge_index, _, _ = k_hop_subgraph(self.last_visited_node, 1, self.graph.edge_index, flow='target_to_source')
        action_mask[edge_index[1]] = 1

        # Mask all nodes that are already visited
        visited_mask = self.graph.x[:, self.flag_visited_index] > 0
        action_mask[visited_mask] = 0
        action_mask[self.starting_depo] = 0

        # Check if all nodes are visited (except the depots)
        if terminal:
            terminal = (visited_mask.sum() == self.graph.num_nodes - 2).item()

        if self.vehicle_ID >= self.MAX_VEHICLES:
            terminal = True

        # if terminal:
            # reward = obj_to_reward(self.vehicle_ID - 1, self.optimal_vehicles, self.MAX_VEHICLES)

        return self.graph.clone(), action_mask, reward, terminal, None
        