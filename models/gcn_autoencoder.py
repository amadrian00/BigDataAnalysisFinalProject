
# models/gcn_autoencoder.py

import torch
import torch.nn as nn
from .projector import Projector
from torch_geometric.nn import GCNConv

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)

        self.projector = Projector(hidden_channels)

    def forward(self, x, edge_index):
        z = self.conv1(x, edge_index)
        x = torch.relu(z)
        return self.conv2(x, edge_index), z, self.projector(z)

class GCNAnomalyDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super(GCNAnomalyDetector, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

class GCNEncoderP(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNEncoderP, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

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

class GCNAnomalyDetectorP(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super(GCNAnomalyDetectorP, self).__init__()
        self.encoder = GCNEncoderP(in_channels, hidden_channels)
        self.decoder = torch.nn.Linear(hidden_channels, in_channels)
        self.projector = Projector(hidden_channels)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return torch.nn.functional.tanh(self.decoder(z)), z, self.projector(z)
