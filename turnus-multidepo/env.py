from typing import Tuple
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import torch_geometric.transforms as T
from utils.problem_loader import ProblemLoader
from utils.graph_generator import GraphGenerator
from utils.graph_utils import add_graph_feature
from utils.utils import obj_to_reward, reward_to_obj

class Env:
    def __init__(self, graph_generator: GraphGenerator = None, graph: Data = None, device = 'cpu'):

        if graph is None and graph_generator is None:
            raise ValueError('Either graph or graph_generator must be provided')

        self.device = device
        self.vehicleID = 1
        self.last_visited_node = 0

        self.objective_value = 0
        # self.MAX_VEHICLES = np.max(graph_generator.size) if graph_generator is not None else graph.num_nodes
        self.graph_generator = graph_generator
        
        # state
        self.graph: Data = graph
        self.choosing_morning_depot = True
        
        if graph is not None:
            self.graph = self.graph.to(self.device)
            self._extend_graph()

        self.reset()

    def action_space(self) -> int:
        return self.graph.num_nodes

    def state_space(self) -> int:
        return self.graph.num_node_features
    
    def reward_to_score_transformer(self):
        def transform(reward):
            return reward_to_obj(reward, 0, self.MAX_VEHICLES)
        return transform

    def reset(self):
        if self.graph_generator is not None:
            self.graph = self.graph_generator.generate().to(self.device)
            self._extend_graph()

        self.MAX_VEHICLES = int(torch.sum(self.graph.depot_vehicle_count).item())
        self.DEPOT_MAX_VEHICLES = torch.max(self.graph.depot_vehicle_count).item()
        self.EDGE_MAX_TIME = torch.max(self.graph.time_matrix).item()

        self.morning_depot_indices = torch.where(self.graph.x[:, ProblemLoader.FEATURES['morning_depot']] == 1)[0]
        self.morning_depot_set = set(self.morning_depot_indices.tolist())
        self.evening_depot_indices = torch.where((self.graph.x[:, ProblemLoader.FEATURES['morning_depot']] == 0) *
                                                  (self.graph.x[:, ProblemLoader.FEATURES['node_type']] == 1))[0]
        self.evening_depot_set = set(self.evening_depot_indices.tolist())
        self.link_indices = torch.where((self.graph.x[:, ProblemLoader.FEATURES['morning_depot']] == 0) *
                                                  (self.graph.x[:, ProblemLoader.FEATURES['node_type']] == 0))[0]

        # Starting from choosing the depot
        self.choosing_morning_depot = True
        self.graph.x[:, self.choosing_morning_depot_flag] = 1
        # No nodes are visited
        self.graph.x[:, self.visited_flag] = 0
        # No nodes are last visited
        self.graph.x[:, self.last_visited_node_flag] = 0
        # Setting the "dynamic" vehicle count in morning depots
        self.graph.x[self.morning_depot_indices, self.vehicle_depot_count_flag] = \
            self.graph.x[self.morning_depot_indices, ProblemLoader.FEATURES['depot_vehicle_count']]
        
        self.objective_value = 0
        self.last_chosen_depot = -1
        self.vehicleID = 1
        self.last_visited_node = -1

        mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)
        mask[self.morning_depot_indices] = 1

        self.graph.x[:, self.possible_nodes_flag] = mask

        self.last_mask = mask

        return self.graph.clone(), mask

    def _extend_graph(self):
        # Add env state to the graph
        self.graph, self.visited_flag = add_graph_feature(self.graph)
        self.graph, self.last_visited_node_flag = add_graph_feature(self.graph)
        self.graph, self.vehicle_depot_count_flag = add_graph_feature(self.graph)
        self.graph, self.possible_nodes_flag = add_graph_feature(self.graph)
        self.graph, self.choosing_morning_depot_flag = add_graph_feature(self.graph)

    # Return next state, mask, reward, and terminal state
    def step(self, action) -> Tuple[torch.Tensor, torch.Tensor, float, bool, None]:
        # We dont need to pick starting depo
        node_id = action
        reward = 0 
        terminal = False
        spent_all_vehicles = False

        if self.last_mask[node_id] == 0:
            raise ValueError('Action is not allowed')

        transport_time = 0

        # Choose one of the depots
        if self.choosing_morning_depot:
            if node_id not in self.morning_depot_set:
                raise ValueError('Invalid depot chosen')
            
            self.choosing_morning_depot = False
            self.graph.x[:, self.choosing_morning_depot_flag] = 0
            self.last_chosen_depot = node_id
            self.last_visited_node = node_id

            self.graph.x[node_id, self.vehicle_depot_count_flag] -= 1 / self.DEPOT_MAX_VEHICLES

        # Choose a node
        else:

            # get edge distance
            transport_time = self.graph.time_matrix[self.last_visited_node, node_id]
            # self.cumulative_distance += transport_time

            # The vehicle has reached the final depot
            if node_id in self.evening_depot_set:
                self.graph.x[node_id, self.vehicle_depot_count_flag] += 1 / self.DEPOT_MAX_VEHICLES
                self.choosing_morning_depot = True
                self.graph.x[:, self.choosing_morning_depot_flag] = 1
                self.vehicleID += 1

                # Maybe reaching the terminal state, now it depends if all nodes are visited
                terminal = True
            else:
                self.graph.x[node_id, self.visited_flag] = 1

        action_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)

        if self.choosing_morning_depot:
            # Get morning depots that have available vehicles
            indices = torch.where((self.graph.x[:, self.vehicle_depot_count_flag] > 0) * 
                                  self.graph.x[:, ProblemLoader.FEATURES['morning_depot']] == 1)[0]
            if len(indices) > 0:
                action_mask[indices] = 1
            else:
                spent_all_vehicles = True
                reward = -10
        else:
            # Mask all nodes that are not connected to the current node
            _, edge_index, _, _ = k_hop_subgraph(node_id, 1, self.graph.edge_index, flow='target_to_source')
            action_mask[edge_index[1]] = 1

            # Mask all nodes that are already visited
            visited_mask = self.graph.x[:, self.visited_flag] > 0
            action_mask[visited_mask] = 0
            action_mask[self.morning_depot_indices] = 0
            action_mask[self.evening_depot_indices] = 0
            action_mask[node_id] = 0
            # allow to return to the selected depot
            if node_id not in self.morning_depot_set:
                action_mask[self.link_indices[-1] + self.last_chosen_depot + 1] = 1


        self.graph.x[:, self.possible_nodes_flag] = action_mask

        self.graph.x[self.last_visited_node, self.last_visited_node_flag] = 0
        self.graph.x[node_id, self.last_visited_node_flag] = 1
        self.last_visited_node = node_id

        # Check if all nodes are visited (except the depots)
        if terminal:
            visited_mask = self.graph.x[self.link_indices, self.visited_flag] == 1
            terminal = visited_mask.all().item()

        if self.vehicleID > self.MAX_VEHICLES:
            terminal = True

        if terminal:
            reward = obj_to_reward(self.vehicleID - 1, 0, self.MAX_VEHICLES)
            self.choosing_morning_depot = False
            self.graph.x[:, self.choosing_morning_depot_flag] = 0
            action_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)
            self.graph.x[:, self.possible_nodes_flag] = action_mask

        self.last_mask = action_mask

        if spent_all_vehicles:
            terminal = True

        return self.graph.clone(), action_mask, reward, terminal, None
        