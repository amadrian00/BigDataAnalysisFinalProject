
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class Projector(nn.Module):
    def __init__(self, hidden_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, input_dim//2),
                                 nn.LayerNorm(input_dim//2),
                                 nn.LeakyReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(input_dim//2, 1))

    def forward(self, features, batch):
        pooled = global_mean_pool(features, batch)
        return self.net(pooled)