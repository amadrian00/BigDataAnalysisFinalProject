import torch.nn as nn

class Projector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, int(2*input_dim)),
                                 nn.BatchNorm1d(int(2*input_dim)),
                                 nn.LeakyReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(int(2*input_dim), input_dim))

    def forward(self, features):
        return self.net(features)