
# models/gcn_autoencoder.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from torch_geometric.nn import global_mean_pool

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, int(hidden_channels / 2))
        self.conv3 = GCNConv(int(hidden_channels / 2), in_channels)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_channels)
        self.bn2 = nn.BatchNorm1d(num_features=int(hidden_channels / 2))

        self.activation = nn.LeakyReLU()

        self.hidden_channels = hidden_channels

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        z = self.activation(x)

        x = self.conv3(z, edge_index)
        return x

class GCNAnomalyDetector(nn.Module):
    def __init__(self, encoder):
        super(GCNAnomalyDetector, self).__init__()
        self.encoder = encoder

        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.4)
        self.lin = nn.Linear(int(encoder.hidden_channels/2), 1)

    def forward(self, x, edge_index):
        x_hat = self.encoder(x, edge_index)
        loss = nn.functional.mse_loss(x_hat, x)
        return x_hat, loss

    def ft(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.encoder.conv1(x, edge_index)
        x = self.encoder.bn1(x)
        x = self.encoder.activation(x)

        x = self.encoder.conv2(x, edge_index)
        x = self.encoder.bn2(x)
        x = self.encoder.activation(x)

        x = global_mean_pool(x, batch)

        x = self.dropout(x)
        return self.lin(x)

class GCNEncoder2(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNEncoder2, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.bn = nn.BatchNorm1d(num_features=hidden_channels)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.4)

        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = self.activation(x)

        x = self.conv2(x, edge_index)
        x= global_mean_pool(x, batch)
        x= self.dropout(x)
        return self.lin(x)