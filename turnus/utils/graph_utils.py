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

    return torch.tensor(np.array(possible_connection)).to(torch.long)

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

    data = Data(x=nodes.to(torch.float), edge_index=edge_connections)

    return data

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