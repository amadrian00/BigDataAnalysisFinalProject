
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

class ChebEncoderP(nn.Module):
    def __init__(self, in_channels, hidden_channels, k=2):
        super(ChebEncoderP, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=k)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=k)

        self.bn = nn.LayerNorm(hidden_channels)

        self.activation = nn.LeakyReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = self.activation(x)

        x = self.conv2(x, edge_index)
        return x

class ChebAnomalyDetectorP(nn.Module):
    def __init__(self, in_channels, hidden_channels, k=2):
        super(ChebAnomalyDetectorP, self).__init__()
        self.encoder = ChebEncoderP(in_channels, hidden_channels, k=k)
        self.decoder = torch.nn.Linear(hidden_channels, in_channels)
        self.projector = Projector(hidden_channels)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return torch.nn.functional.tanh(self.decoder(z)), z, self.projector(z)
