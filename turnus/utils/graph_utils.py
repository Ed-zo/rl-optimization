import os
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch
import pandas as pd
import numpy as np

def load_node_csv(path: str, index_col: str, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
        df = df[encoders.keys()].copy()

    return x, mapping, df

def build_edge_connections(time_matrix_path: str, start_times: np.array, finish_times: np.array, delimiter=';'):
    matrix = np.loadtxt(time_matrix_path, delimiter=delimiter)
    # Add the transport times (matrix) to the finishing time of every task
    transport_times = matrix + np.expand_dims(finish_times, axis=-1)

    possible_connection = np.where(transport_times <= start_times)

    return torch.from_numpy(np.array(possible_connection)).to(torch.long)

class IdentityEncoder:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

# Normalize and convert dataframe column (Series) into a tensor
class NumberNormEncoder:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(((df - df.mean()) / df.std()).values).view(-1, 1).to(self.dtype)

def load_graph(path: str):
    nodes, mapping, df = load_node_csv(f'{path}/tasks.csv', 'Index', 
                               {name: NumberNormEncoder() for name in ['ZastavkaStart', 'ZastavkaFinish', 'CasStart', 'CasFinish', 'Vzdialenost', 'Trvanie']}, 
                               sep=';')

    edge_connections = build_edge_connections(f'{path}/Tij.csv', df['CasStart'].values, df['CasFinish'].values)

    # connections between starting and finishing depot are not allowed
    edge_connections = remove_depot_connections(edge_connections, [0, nodes.shape[0] - 1])
    edge_connections = torch.cat([edge_connections, torch.tensor([[0], [nodes.shape[0] - 1]])], dim=-1)

    data = Data(x=nodes.float(), edge_index=edge_connections)
    # data = Data(x=torch.ones((nodes.shape[0], 1)), edge_index=edge_connections)


    return data

def remove_depot_connections(edge_connections, depot_indices):
    depot_indices = torch.tensor(depot_indices)
    connections = edge_connections.t()
    # generate mask of connections between depots and remove them
    mask = torch.isin(connections[:, 0], depot_indices).logical_and(torch.isin(connections[:, 1], depot_indices))

    return connections[mask.logical_not()].t()

def add_node_degree(graph: Data):
    add_degree = T.OneHotDegree(max_degree=10)
    graph = add_degree(graph)

    return graph

def add_graph_feature(graph: Data, value = 0):
    add_constant_fn = T.Constant(value)
    graph = add_constant_fn(graph)
    flag_index = graph.num_node_features - 1

    return graph, flag_index

def load_problem(path: str):
    graph = load_graph(path)

    optimal_file = os.path.join(path, 'optimal.txt')
    optimal_result = None
    if os.path.exists(optimal_file):
        with open(optimal_file, 'r') as w:
            optimal_result = int(w.readline())

    return graph, optimal_result

def get_neighbor_indices(edge_index, node_index):
    return edge_index[1][edge_index[0] == node_index]