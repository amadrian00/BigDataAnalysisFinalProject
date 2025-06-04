import os
import torch
import numpy as np
from sklearn.svm import SVC
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

from models.gcn_autoencoder import GCNAnomalyDetector
from models.gat_autoencoder import GATAnomalyDetector
from models.chebnet_autoencoder import ChebAnomalyDetector

from train_test import train, contrastive_train, test, print_summary_table, compare_models, extract_embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # para multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

data = np.load('data/830MDDvs771NC_cc200_Combat.npz', allow_pickle=True)
fc_matrices = data['fc']  # Shape: (n_samples, n_nodes, n_nodes)
labels = data['label']  # Shape: (n_samples,)
subjects = [s.split("-")[0] for s in data['subject']]  # Shape: (n_samples,)

# Function to convert an FC matrix into a graph with edge thresholding
def create_graph(adj_matrix, th):
    # Apply absolute thresholding and exclude the diagonal
    mask = np.triu(np.abs(adj_matrix) >= th, k=1)
    row, col = np.where(mask)

    edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long, device=device)
    edge_index = torch.cat((edge_index, torch.flip(edge_index, dims=[0])), dim=1)

    edge_attr = torch.tensor(adj_matrix[row, col], dtype=torch.float, device=device)

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
hidden_dim = 128
epochs = 25

batch_size = 64

lr = 1e-3
wd = 1e-3

results = {'GCN ': [], 'GAT ': [], 'Chb ': [], 'PGCN': [], 'PGAT': [], 'PChb': []}

for repeat, s in enumerate(seeds):
    print(f"Repeat {repeat+1}/{len(seeds)}")

    set_seed(s)

    fold_splits = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=s)
    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        pretrain_graphs = [graphs[i] for i in train_idx]
        train_graphs = [graphs[i] for i in test_idx]

        fold_splits.append((pretrain_graphs, train_graphs))

    for i, (train_graphs, test_graphs) in enumerate(fold_splits):
        print(f"    Fold {i + 1}/{len(fold_splits)}")

        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_graphs, batch_size=len(test_graphs), shuffle=True)

        for model_name, AEClass in zip(['GCN ', 'GAT ', 'Chb ', 'PGCN', 'PGAT', 'PChb'],
                                        [GCNAnomalyDetector, GATAnomalyDetector, ChebAnomalyDetector,
                                         GCNAnomalyDetector, GATAnomalyDetector, ChebAnomalyDetector]):

            model = AEClass(in_channels=input_dim, hidden_channels=hidden_dim).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            if model_name[0] == 'P':
                for epoch in range(epochs):
                    train_loss = contrastive_train(model, train_loader, optimizer, device=device)
            else:
                for epoch in range(epochs):
                    train_loss = train(model, train_loader, optimizer, device=device)

            X_tr, y_tr = extract_embeddings(model, train_loader, device='cuda')

            classifier = SVC()
            classifier.fit(X_tr, y_tr)

            res  = test(model, test_loader, classifier, device=device)
            results[model_name].append(res)
            print(f"        {model_name}: " + ", ".join([f"{k}={v:.4f}" for k, v in res.items()]))

    print()

np.save('resultsGCN.npy', results['GCN '])
np.save('resultsGAT.npy', results['GAT '])
np.save('resultsCheb.npy', results['Chb '])
print_summary_table(results)

compare_models(results['GCN '], results['PGCN'], label_a="GCN", label_b="P GCN")
compare_models(results['GAT '], results['PGAT'], label_a="GCN", label_b="P GAT")
compare_models(results['Chb '], results['PChb'], label_a="GCN", label_b="P Cheb")