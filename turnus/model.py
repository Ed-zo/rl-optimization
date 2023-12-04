import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, state_size, node_count):
        super().__init__()
        self.state_size = state_size
        self.node_count = node_count

        self.conv1 = GCNConv(state_size, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 32)
        self.conv5 = GCNConv(32, 32)

        self.fc_p1 = nn.Linear(node_count * self.conv5.out_channels, 256)
        self.fc_p2 = nn.Linear(256, node_count)

        self.fc_v1 = nn.Linear(node_count * self.conv5.out_channels, 256)
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
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.relu(x)

        x = x.view(-1, self.fc_p1.in_features)

        X = F.relu(self.fc_p1(x))
        X = self.fc_p2(X)

        V = F.relu(self.fc_v1(x))
        V = self.fc_v2(V)

        return X, V