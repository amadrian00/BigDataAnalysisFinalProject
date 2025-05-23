import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

from models.gcn_autoencoder import GCNEncoder, GCNAnomalyDetector, GCNEncoder2
from models.gat_autoencoder import GATEncoder, GATAnomalyDetector, GATEncoder2
from models.chebnet_autoencoder import ChebEncoder, ChebAnomalyDetector, ChebEncoder2

from train_test import  pretrain, train, test, print_summary_table, compare_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # para multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Load the .npz file
data = np.load('data/830MDDvs771NC_cc200_Combat.npz', allow_pickle=True)
fc_matrices = data['fc']  # Shape: (n_samples, n_nodes, n_nodes)
labels = data['label']  # Shape: (n_samples,)
subjects = data['subject']  # Shape: (n_samples,)

# Function to convert an FC matrix into a graph with edge thresholding
def create_graph(adj_matrix, th):
    # Apply absolute thresholding and exclude the diagonal
    mask = np.triu(np.abs(adj_matrix) >= th, k=1)
    row, col = np.where(mask)

    edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long, device=device)
    edge_attr = torch.tensor(adj_matrix[row, col], dtype=torch.float, device=device)

    # Dummy node features (identity matrix)
    x = torch.eye(adj_matrix.shape[0], dtype=torch.float)
    x = torch.tensor(adj_matrix, dtype=torch.float, device=device)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Build list of graphs
graphs = [create_graph(fc_matrices[i], th=0.5) for i in range(len(fc_matrices))]
for i, graph in enumerate(graphs):
    graph.y = torch.tensor([(labels[i] + 1) // 2], dtype=torch.long, device=device)
    graph.subject = subjects[i]

# Generate stratified k-fold splits (5 folds, repeated 5 times)
k = 5
seeds = [21, 42, 63, 84, 105]

input_dim = graphs[0].x.shape[1]
hidden_dim = 64
epochs = 50

batch_size = 64
lr = 1e-3
wd = 1e-5

results = {'GCN': [], 'GAT': [], 'Chb': [], 'GCN2': [], 'GAT2': [], 'Chb2': []}

for repeat, s in enumerate(seeds):
    print(f"Repeat {repeat+1}/{len(seeds)}")

    set_seed(s)

    fold_splits = []

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=s)
    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]

        y = [data.y for data in train_graphs]
        n_pos = sum(label == 1 for label in y)
        n_neg = sum(label == 0 for label in y)

        weight = (n_neg / n_pos).to(device)

        fold_splits.append((train_graphs, test_graphs, weight))

    for i, (train_graphs, test_graphs, weight) in enumerate(fold_splits):
        print(f"    Fold {i + 1}/{len(fold_splits)}")

        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=len(test_graphs))

        for model_name, ModelClass, AEClass in zip(['GCN', 'GAT', 'Chb', 'GCN2', 'GAT2', 'Chb2'],
                                                             [GCNEncoder, GATEncoder, ChebEncoder, GCNEncoder2, GATEncoder2, ChebEncoder2],
                                                   [GCNAnomalyDetector, GATAnomalyDetector, ChebAnomalyDetector, None, None, None]):
            baseline = model_name[-1]  == '2'
            if baseline:
                model = ModelClass(in_channels=input_dim, hidden_channels=hidden_dim).to(device)
            else:
                encoder = ModelClass(in_channels=input_dim, hidden_channels=hidden_dim).to(device)
                model = AEClass(encoder=encoder).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            if not baseline:
                for epoch in range(epochs):
                    train_loss = pretrain(model, train_loader, optimizer, device=device)

                for epoch in range(10):
                    for param in model.encoder.parameters():
                        param.requires_grad = False

                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
                    train_loss = train(model, train_loader, optimizer, weight, device=device)

                for epoch in range(epochs-10):
                    for param in model.parameters():
                        param.requires_grad = True

                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                    train_loss = train(model, train_loader, optimizer, weight, device=device)

            else:
                for epoch in range(epochs):
                    train_loss = train(model, train_loader, optimizer, weight, device=device)

            res  = test(model, test_loader, device=device)
            results[model_name].append(res)
            print(f"        {model_name}: " + ", ".join([f"{k}={v:.4f}" for k, v in res.items()]))
    print()

np.save('resultsGCN.npy', results['GCN'])
np.save('resultsGAT.npy', results['GAT'])
np.save('resultsCheb.npy', results['Chb'])
print_summary_table(results)

compare_models(results['GCN'], results['GCN2'], label_a="GCN", label_b="GCN2")
compare_models(results['GAT'], results['GAT2'], label_a="GAT", label_b="GAT2")
compare_models(results['Chb'], results['Chb2'], label_a="Cheb", label_b="Cheb2")