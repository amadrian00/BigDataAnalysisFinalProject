
# models/gcn_autoencoder.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from torch_geometric.nn import global_mean_pool

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)

        self.hidden_channels = hidden_channels

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class GCNAnomalyDetector(nn.Module):
    def __init__(self, encoder):
        super(GCNAnomalyDetector, self).__init__()
        self.encoder = encoder
        self.lin = nn.Linear(encoder.hidden_channels, 1)

    def forward(self, x, edge_index):
        x_hat = self.encoder(x, edge_index)
        loss = nn.functional.mse_loss(x_hat, x)
        return x_hat, loss

    def ft(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.encoder.conv1(x, edge_index)
        return self.lin(global_mean_pool(x, batch))
