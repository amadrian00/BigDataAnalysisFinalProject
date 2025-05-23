import torch
import numpy as np
from scipy.stats import ttest_rel, tstd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def pretrain(model, loader, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out, loss = model(data.x, data.edge_index)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train(model, loader, optimizer, weight, device='cpu'):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        try:
            out = model.ft(data)  # logits shape: [batch_size]
        except:
            out = model(data)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out,
            data.y.type_as(out).view_as(out),
            pos_weight= weight
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, device='cpu'):
    model.eval()
    preds = []
    probs = []
    truths = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            try:
                out = model.ft(data)  # logits shape: [batch_size]
            except:
                out = model(data)

            prob = torch.sigmoid(out)  # No squeeze
            pred = (prob >= 0.5).long()

            preds.append(pred.cpu())
            probs.append(prob.cpu())
            truths.append(data.y.cpu())

    preds = torch.cat(preds)
    probs = torch.cat(probs)
    truths = torch.cat(truths)

    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds)
    auc = roc_auc_score(truths, probs)

    tn, fp, fn, tp = confusion_matrix(truths, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": auc
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


