
# models/chebnet_autoencoder.py

import torch
import torch.nn as nn
from .projector import Classifier
from torch_geometric.nn import ChebConv

class ChebEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, k=2):
        super(ChebEncoder, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=k)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=k)

        self.bn = nn.LayerNorm(hidden_channels)

        self.activation = nn.LeakyReLU()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = self.activation(x)

        x = self.conv2(x, edge_index)
        return x

class ChebAnomalyDetector(nn.Module):
    def __init__(self, encoder):
        super(ChebAnomalyDetector, self).__init__()
        self.encoder = encoder
        self.decoder = torch.nn.Linear(self.encoder.hidden_channels, self.encoder.in_channels)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return torch.nn.functional.tanh(self.decoder(z)), z

class ChebEncoder2(nn.Module):
    def __init__(self, in_channels, hidden_channels, k=2):
        super(ChebEncoder2, self).__init__()
        self.encoder = ChebEncoder(in_channels, hidden_channels, k)

        self.lin = Classifier(hidden_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.encoder(x, edge_index)
        return self.lin(x, batch)
