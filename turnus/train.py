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

        self.conv1 = GCNConv(state_size, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)

        self.fc_p1 = nn.Linear(node_count * self.conv3.out_channels, 256)
        self.fc_p2 = nn.Linear(256, node_count)

        self.fc_v1 = nn.Linear(node_count * self.conv3.out_channels, 256)
        self.fc_v2 = nn.Linear(256, 1)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
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
    print('Start time:', start_date)
    graph = graph_utils.load_graph('data/1')

    env = Env(graph, device)

    net = GCN(env.state_space(), env.action_space()).to(device)

    agent = Agent(net, 'Adam', device=device, name='ppo_1', path='results/', gamma=1, epsilon=0.1, lr=0.001)

    agent.train([graph, device], Env, graph.num_nodes, count_of_iterations=10000, count_of_processes=4, count_of_envs=16, count_of_steps=52*3, batch_size=768)

    end_date = datetime.datetime.now()
    print('End time:', end_date)
    print('Tooked', end_date - start_date)

    torch.save(net, 'save.net')
