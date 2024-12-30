import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
 

class HybridConv(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.state_size = in_features

        self.lin = nn.Linear(in_features, out_features)
        self.conv = GATConv(in_features * 2, out_features)

    def out_channels(self):
        return self.lin.out_features + self.conv.out_channels

    def forward(self, x_lin, x_conv, edge_index, edge_attr=None):

        x_combined = torch.cat([x_lin, x_conv], dim=-1)
        x_conv = self.conv(x_combined, edge_index, edge_attr=edge_attr)

        x_lin = self.lin(x_lin)

        return x_lin, x_conv


class HybridNetwork(torch.nn.Module):
    def __init__(self, state_size, node_count):
        super().__init__()

        self.state_size = state_size
        self.node_count = node_count

        self.h1 = HybridConv(state_size, 16)
        self.h2 = HybridConv(16, 16)
        self.h3 = HybridConv(16, 16)
        self.h4 = HybridConv(16, 16)

        self.l1 = nn.Linear(self.h4.out_channels(), 32)
        self.l2 = nn.Linear(32, 1)

        self.v_0 = nn.Linear(self.h4.out_channels(), 16)
        self.attention_fc = nn.Linear(16, 1)
        self.v_1 = nn.Linear(self.v_0.out_features, 16)
        self.v_2 = nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.adj_t if hasattr(data, 'adj_t') else data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        x_lin, x_conv = self.h1(x, x, edge_index, edge_attr)
        x_lin, x_conv = F.relu(x_lin), F.relu(x_conv)

        x_lin, x_conv = self.h2(x_lin, x_conv, edge_index, edge_attr)
        x_lin, x_conv = F.relu(x_lin), F.relu(x_conv)

        x_lin, x_conv = self.h3(x_lin, x_conv, edge_index, edge_attr)
        x_lin, x_conv = F.relu(x_lin), F.relu(x_conv)

        x_lin, x_conv = self.h4(x_lin, x_conv, edge_index, edge_attr)
        x_lin, x_conv = F.relu(x_lin), F.relu(x_conv)

        x_combined = torch.cat([x_lin, x_conv], dim=-1)
        x = F.relu(self.l1(x_combined))
        x = self.l2(x)
        X = x.view(-1, self.node_count)

        V = F.relu(self.v_0(x_combined))

        attention = self.attention_fc(V)
        attention = attention.view(-1, self.node_count)
        # Attention-Based Pooling for Critic
        attention_weights = F.softmax(attention, dim=1).unsqueeze(2)  # Learn attention weights

        # sum features over all nodes
        V = V.view(-1, self.node_count, self.v_0.out_features)
        V = V * attention_weights
        graph_embedding = torch.sum(V, dim=1)   # Weighted sum

        V = self.v_2(graph_embedding)

        return X, V, V.clone().detach()