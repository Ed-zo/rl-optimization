from io import TextIOWrapper
import os
import numpy as np
import torch
from torch_geometric.data import Data

"""
Problem loader. The detailed description of .dat file is on:
https://andreas-ernst.github.io/Mathprog-ORlib/info/readmeMDVSP.html
"""
class ProblemLoader():

    FEATURES = {key: i for i, key in enumerate(['start_times', 'end_times', 'durations', 
                                                             'depot_vehicle_count', 'node_type', 'morning_depot'])}

    def __init__(self, path: str) -> None:
        self.path = path

    def load(self, name: str):
        file = os.path.join(self.path, f'{name}.dat')
        if not os.path.exists(file):
            raise FileNotFoundError(f'File {file} does not exist')

        with open(file, 'r') as f:
            # Load node features and time matrix
            start_locations, start_times, end_locations, end_times, \
                durations, depot_vehicle_count, node_type, morning_depot = self._load_features(f)

            node_features = np.stack([start_times, end_times, durations, depot_vehicle_count, node_type, morning_depot], axis=-1)
            # normalize node features using min-max normalization
            node_features = self._normalize_feature(node_features, strategy='min-max')
            node_features = torch.from_numpy(node_features).float()

            time_matrix = np.loadtxt(f, delimiter=' ')

            # build edge connections using time matrix and time tables
            edge_connections, edge_weights = self.__load_edge_connections(time_matrix, start_locations, end_locations, 
                                                                          start_times, end_times, node_type, morning_depot)
            
            edge_connections = torch.from_numpy(edge_connections).long()
            edge_weights = torch.from_numpy(edge_weights).float()
            normalized_edge_weights = edge_weights / torch.max(edge_weights)

            data = Data(x=node_features, edge_index=edge_connections, edge_attr=normalized_edge_weights)
            
            data.depot_vehicle_count = torch.from_numpy(depot_vehicle_count)

            time_matrix = torch.zeros((node_features.shape[0], node_features.shape[0]))
            time_matrix[edge_connections[0], edge_connections[1]] = edge_weights
            data.time_matrix = time_matrix

        optimal_file = os.path.join(self.path, f'{name}.opt')
        optimal = None
        if os.path.exists(optimal_file):
            with open(optimal_file, 'r') as f:
                optimal = int(f.readline())
            
        return data, optimal

    def __load_edge_connections(self, time_matrix: np.ndarray, start_locations: np.ndarray, end_locations: np.ndarray, 
                                start_times: np.ndarray, finish_times: np.ndarray, node_type: np.ndarray, morning_depot: np.ndarray):
        
        # get all possible connections between nodes        
        start_grid, end_grid = np.meshgrid(start_locations, end_locations)
        # transform the time matrix of locations to the time matrix of nodes
        time_matrix = time_matrix[end_grid, start_grid]
        # get transport (delta) time from ending location to every other location
        transport_times = time_matrix[end_locations]
        # calculcate the absolute time of finishing the link + transport time
        end_transport_times = transport_times + np.expand_dims(finish_times, axis=-1)

        possible_transports = end_transport_times <= start_times

        # get indiced of morning and evening depots
        morning_depot_indices = np.where(morning_depot == 1)[0]
        evening_depot_indices = np.where((morning_depot == 0) * (node_type == 1))[0]
        morning_depot_indices, evening_depot_indices = np.meshgrid(morning_depot_indices, evening_depot_indices)
        # remove connections from morning depots to evening depots
        possible_transports[morning_depot_indices, evening_depot_indices] = False

        edge_indices = np.array(np.where(possible_transports))

        edge_weights = time_matrix[edge_indices[0], edge_indices[1]].flatten()

        return edge_indices, edge_weights

    def _normalize_feature(self, features: np.array, strategy='min-max'):
        if strategy == 'z-score':
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            return (features - mean) / std
        elif strategy == 'min-max':
            min = np.min(features, axis=0)
            max = np.max(features, axis=0)
            return (features - min) / (max - min)


    def _load_features(self, file: TextIOWrapper):
        # load first three numbers on the first line
        depots, number_of_trips, number_of_locations = map(int, file.readline().split())
        vehicle_capacities = list(map(int, file.readline().split()))
        node_count = depots * 2 + number_of_trips
        start_locations = np.zeros(node_count, dtype=int)
        end_locations = np.zeros(node_count, dtype=int)
        start_times = np.zeros(node_count)
        end_times = np.zeros(node_count)
        durations = np.zeros(node_count)
        depot_vehicle_count = np.zeros(node_count)
        node_type = np.zeros(node_count) #0 - trip, 1 - depot
        morning_depot = np.zeros(node_count) #0 - evening depot, 1 - morning depot

        # Loading morning depots
        start_times[:depots] = 0
        end_times[:depots] = 0
        durations[:depots] = 0
        depot_vehicle_count[:depots] = vehicle_capacities
        start_locations[:depots] = np.arange(depots)
        end_locations[:depots] = np.arange(depots)
        node_type[:depots] = 1
        morning_depot[:depots] = 1

        # Loading main nodes - trips
        for i in range(depots, number_of_trips + depots):
            start_location, start_time, end_location, end_time = map(int, file.readline().split())
            start_locations[i] = start_location
            end_locations[i] = end_location
            start_times[i] = start_time
            end_times[i] = end_time
            durations[i] = end_time - start_time

        # Loading evening depots
        evening_depot_index = depots + number_of_trips
        start_locations[evening_depot_index:] = np.arange(depots)
        end_locations[evening_depot_index:] = np.arange(depots)
        start_times[evening_depot_index:] = 1600
        end_times[evening_depot_index:] = 1600
        durations[evening_depot_index:] = 0
        depot_vehicle_count[evening_depot_index:] = 0
        node_type[evening_depot_index:] = 1
        morning_depot[evening_depot_index:] = 0

        return start_locations, start_times, end_locations, end_times, durations, depot_vehicle_count, node_type, morning_depot
