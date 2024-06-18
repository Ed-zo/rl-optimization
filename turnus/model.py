import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import Sequential, GCNConv, GATConv

class GCNStaticPolicy(torch.nn.Module):
    def __init__(self, state_size, node_count):
        super().__init__()
        self.state_size = state_size
        self.node_count = node_count

        self.conv1 = GCNConv(state_size, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 8)

        self.fc_p1 = nn.Linear(node_count * self.conv3.out_channels, 256)
        self.fc_p2 = nn.Linear(256, 256)
        self.fc_p3 = nn.Linear(256, node_count)

        self.fc_v1 = nn.Linear(node_count * self.conv3.out_channels, 256)
        self.fc_int_v2 = nn.Linear(256, 1)
        self.fc_ext_v2 = nn.Linear(256, 1)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, data):
        # x, edge_index = data.x, data.edge_index

        x, edge_index = data.x, data.adj_t if hasattr(data, 'adj_t') else data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = x.view(-1, self.fc_p1.in_features)

        X = F.relu(self.fc_p1(x))
        X = F.relu(self.fc_p2(X))
        X = self.fc_p3(X)

        V = F.relu(self.fc_v1(x))
        int_V = self.fc_int_v2(V)
        ext_V = self.fc_ext_v2(V)

        return X, ext_V, int_V
    

class GCNPolicy(torch.nn.Module):
    def __init__(self, state_size, node_count):
        super().__init__()
        self.state_size = state_size
        self.node_count = node_count

        self.conv1 = GCNConv(state_size, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 64)
        self.conv4 = GCNConv(64, 64)
        self.conv5 = GCNConv(64, 64)

        self.conv_p1 = GCNConv(64, 32)
        self.conv_p2 = GCNConv(32, 16)
        self.conv_p3 = GCNConv(16, 1)

        # self.fc_p1 = nn.Linear(node_count * self.conv3.out_channels, 256)
        # self.fc_p2 = nn.Linear(256, 256)
        # self.fc_p3 = nn.Linear(256, node_count)

        self.fc_v1 = nn.Linear(self.conv5.out_channels, 256)
        self.fc_int_v2 = nn.Linear(256, 1)
        self.fc_ext_v2 = nn.Linear(256, 1)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, data):
        data = data

        x, edge_index = data.x, data.adj_t if hasattr(data, 'adj_t') else data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        a = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(a, edge_index))
        x = F.relu(self.conv5(x, edge_index))

        x = x + a
        X = F.relu(self.conv_p1(x, edge_index))
        X = F.relu(self.conv_p2(X, edge_index))
        X = self.conv_p3(X, edge_index)
        X = X.view(-1, self.node_count)

        x = x.unsqueeze(0)
        x = x.view(-1, self.node_count, self.conv5.out_channels)
        V = x.mean(dim=1)
        V = F.relu(self.fc_v1(V))
        int_V = self.fc_int_v2(V)
        ext_V = self.fc_ext_v2(V)

        return X, ext_V, int_V


class RNDModel(torch.nn.Module):
    def __init__(self, state_size, node_count):
        super().__init__()
        self.state_size = state_size
        self.node_count = node_count

        self.pred_conv = Sequential('x, edge_index', [
            (GCNConv(state_size, 16), 'x, edge_index -> x1'),
            nn.ELU(inplace=True),
            (GCNConv(16, 8), 'x1, edge_index -> x2'),
            nn.ELU(inplace=True),
            (GCNConv(8, 8), 'x2, edge_index -> x3'),
            nn.ELU(inplace=True)
        ])

        self.pred_fc_1 = nn.Linear(node_count * list(self.pred_conv.children())[-2].out_channels, 256)
        self.pred_fc_2 = nn.Linear(256, 256)

        self.targ_conv = Sequential('x, edge_index', [
            (GCNConv(state_size, 16), 'x, edge_index -> x1'),
            nn.ELU(inplace=True),
            (GCNConv(16, 8), 'x1, edge_index -> x2'),
            nn.ELU(inplace=True),
            (GCNConv(8, 8), 'x2, edge_index -> x3'),
            nn.ELU(inplace=True)
        ])

        self.targ_fc_1 = nn.Linear(node_count * list(self.targ_conv.children())[-2].out_channels, 256)

        self.apply(self._init_weights)

        self.targ_conv.train(False)
        self.targ_fc_1.train(False)
        for pr in self.targ_conv.parameters():
            pr.requires_grad = False
        self.targ_fc_1.requires_grad_ = False


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index

        prediction = self.pred_conv(x, edge_index)
        prediction = prediction.view(-1, self.pred_fc_1.in_features)
        prediction = F.elu( self.pred_fc_1(prediction) )
        prediction = self.pred_fc_2(prediction)
        
        with torch.no_grad():
            target = self.targ_conv(x, edge_index)
            target = target.view(-1, self.targ_fc_1.in_features)
            target = self.targ_fc_1(target)

        return prediction, target