import torch
import numpy as np
from scipy.stats import ttest_rel, tstd
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train(model, loader, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        loss = torch.nn.functional.mse_loss(out[0], data.x.type_as(out[0],).view_as(out[0],))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, classifier=None, device='cpu'):
    model.eval()
    preds = []
    truths = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data.x, data.edge_index)

            z = global_mean_pool(out[1], data.batch)

            pred = torch.tensor(classifier.predict(z.cpu()))

            preds.append(pred)
            truths.append(data.y.cpu())

    preds = torch.cat(preds)
    truths = torch.cat(truths)

    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds)

    tn, fp, fn, tp = confusion_matrix(truths, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
    }

def print_summary_table(all_metrics):
    """Print mean ± std for each model and each metric."""
    models = list(all_metrics.keys())
    metrics = list(all_metrics[models[0]][0].keys())

    print("\n--- Models Summary (Mean ± Std) ---")
    header = f"{'Model':<10}" + "".join([f"{m:<20}" for m in metrics])
    print(header)
    print("-" * len(header))

    for model in models:
        line = f"{model:<10}"
        for metric in metrics:
            values = [m[metric] for m in all_metrics[model]]
            mean = np.mean(values)
            std = tstd(values)
            line += f"{mean:.4f}±{std:.4f}       "
        print(line)

def compare_models(metrics_a, metrics_b, label_a="Model A", label_b="Model B"):
    """Print paired t-test and % improvement for two models."""
    print(f"\n--- Comparison: {label_a} vs {label_b} ---")
    print(f"{'Metric':<15}{'Mean ' + label_a:>12}{'Std ' + label_a:>12}"
          f"{'Mean ' + label_b:>15}{'Std ' + label_b:>12}{'% Inc.':>15}{'p-value':>12}")
    print("-" * 90)

    for key in metrics_a[0].keys():
        values_a = [m[key] for m in metrics_a]
        values_b = [m[key] for m in metrics_b]

        mean_a = np.mean(values_a)
        std_a = tstd(values_a)

        mean_b = np.mean(values_b)
        std_b = tstd(values_b)

        t_stat, p_val = ttest_rel(values_b, values_a)

        try:
            pct_inc = ((mean_b - mean_a) / abs(mean_a)) * 100
        except ZeroDivisionError:
            pct_inc = float('nan')

        print(f"{key:<15}{mean_a:>12.4f}{std_a:>12.4f}"
              f"{mean_b:>15.4f}{std_b:>12.4f}"
              f"{pct_inc:>15.2f}{p_val:>12.4f}")

def graph_augment_fmri(x, e, noise_std=0.25, drop_prob=0.25):
    x_aug = x.clone()

    noise = torch.randn_like(x_aug) * noise_std
    x_aug = x_aug + noise
    x_aug = torch.nn.functional.dropout(x_aug, drop_prob)

    return x_aug, e

def negative_cosine_similarity(p, z):
    p = torch.nn.functional.normalize(p, dim=-1)
    z = torch.nn.functional.normalize(z.detach(), dim=-1)
    return - (p * z).sum(dim=-1).mean()

def simsiam_loss(p1, z2, p2, z1):
    return (negative_cosine_similarity(p1, z2) + negative_cosine_similarity(p2, z1)) / 2

def contrastive_train(model, loader, optimizer, device='cpu'):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        x = data.x
        e = data.edge_index

        # Aumentaciones
        x1, e1 = graph_augment_fmri(x, e)
        x2, e2 = graph_augment_fmri(x, e)
        x1, x2, e1, e2 = x1.to(device), x2.to(device), e1.to(device), e2.to(device)

        # Forward
        _, h1, p1 = model(x1, e1)
        _, h2, p2 = model(x2, e2)

        # Para batch training, reordena en [batch, feature]
        B = h1.shape[0] // 200
        h1 = h1.view(B, -1)
        h2 = h2.view(B, -1)
        p1 = p1.view(B, -1)
        p2 = p2.view(B, -1)

        # SimSiam loss
        loss = simsiam_loss(p1, h2, p2, h1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def extract_embeddings(encoder, loader, device='cpu'):
    encoder.eval()
    all_embeddings = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            _, z, _ = encoder(data.x, data.edge_index)
        pooled = global_mean_pool(z, data.batch)
        all_embeddings.append(pooled.cpu())
        all_labels.append(data.y.cpu())
    X = torch.cat(all_embeddings).numpy()
    y = torch.cat(all_labels).numpy()
    return X, y