
# models/gat_autoencoder.py

import torch
import torch.nn as nn
from .projector import Projector
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, in_channels, heads=1)

        self.projector = Projector(hidden_channels)

    def forward(self, x, edge_index):
        z = self.conv1(x, edge_index)
        x = torch.relu(z)
        return self.conv2(x, edge_index), z, self.projector(z)

class GATAnomalyDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super(GATAnomalyDetector, self).__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, heads=heads)

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

class GATEncoderP(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super(GATEncoderP, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=1)

        self.bn = nn.LayerNorm(hidden_channels*heads)

        self.activation = nn.LeakyReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = self.activation(x)

        x = self.conv2(x, edge_index)
        return x

class GATAnomalyDetectorP(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=2):
        super(GATAnomalyDetectorP, self).__init__()
        self.encoder = GATEncoderP(in_channels, hidden_channels, heads=heads)
        self.decoder = torch.nn.Linear(hidden_channels, in_channels)
        self.projector = Projector(hidden_channels)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return torch.nn.functional.tanh(self.decoder(z)), z, self.projector(z)