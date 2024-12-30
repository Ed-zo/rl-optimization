import os
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch
import pandas as pd
import numpy as np

def add_node_degree(graph: Data):
    add_degree = T.OneHotDegree(max_degree=10)
    graph = add_degree(graph)

    return graph

def add_graph_feature(graph: Data, value = 0):
    add_constant_fn = T.Constant(value)
    graph = add_constant_fn(graph)
    flag_index = graph.num_node_features - 1

    return graph, flag_index


def get_neighbor_indices(edge_index, node_index):
    return edge_index[1][edge_index[0] == node_index]