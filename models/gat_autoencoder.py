
# models/gat_autoencoder.py

import torch
import torch.nn as nn
from .projector import Classifier
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=1)

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

class GATAnomalyDetector(nn.Module):
    def __init__(self, encoder):
        super(GATAnomalyDetector, self).__init__()
        self.encoder = encoder
        self.decoder = torch.nn.Linear(self.encoder.hidden_channels, self.encoder.in_channels)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return torch.nn.functional.tanh(self.decoder(z)), z

class GATEncoder2(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super(GATEncoder2, self).__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, heads=heads)

        self.lin = Classifier(hidden_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.encoder(x, edge_index)
        return self.lin(x, batch)
