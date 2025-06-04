
# models/chebnet_autoencoder.py

import torch
import torch.nn as nn
from .projector import Projector
from torch_geometric.nn import ChebConv

class ChebEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, K=2):
        super(ChebEncoder, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        self.conv2 = ChebConv(hidden_channels, in_channels, K=K)

        self.projector = Projector(hidden_channels)

    def forward(self, x, edge_index):
        z = self.conv1(x, edge_index)
        x = torch.relu(z)
        return self.conv2(x, edge_index), z, self.projector(z)

class ChebAnomalyDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels, K=2):
        super(ChebAnomalyDetector, self).__init__()
        self.encoder = ChebEncoder(in_channels, hidden_channels, K=K)

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)
