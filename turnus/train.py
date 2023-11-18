import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv

import torch
import torch.nn as nn
import datetime
import torch.nn.functional as F
from ppo_parallel import Agent
import torch.nn.init as init
import numpy as np
import signal
from env import Env
import graph_utils


class GCN(torch.nn.Module):
    def __init__(self, state_size, node_count):
        super().__init__()
        self.state_size = state_size
        self.node_count = node_count

        self.conv1 = GCNConv(state_size, 16)
        self.conv2 = GCNConv(16, 16)

        self.fc_p1 = nn.Linear(node_count * self.conv2.out_channels, 32)
        self.fc_p2 = nn.Linear(32, node_count)

        self.fc_v1 = nn.Linear(node_count * self.conv2.out_channels, 16)
        self.fc_v2 = nn.Linear(16, 1)

        features_layers = []  # [self.conv1, self.conv2]
        for layer in features_layers:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

        output_layers = [self.fc_p1, self.fc_p2, self.fc_v1, self.fc_v2]
        for layer in output_layers:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = x.view(-1, self.fc_p1.in_features)

        X = F.relu(self.fc_p1(x))
        X = self.fc_p2(X)

        V = F.relu(self.fc_v1(x))
        V = self.fc_v2(V)

        return X, V

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    start_date = datetime.datetime.now()
    graph = graph_utils.load_graph('data/1')

    env = Env(graph, device)

    net = GCN(env.state_space(), env.action_space()).to(device)
    # net = torch.load('models/p_med_last.pt')

    agent = Agent(net, 'Adam', device=device, name='ppo', path='results')

    agent.train([graph, device], Env, graph.num_nodes)

    # torch.save(net, 'models/save.net')
