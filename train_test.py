import torch
import numpy as np
from scipy.stats import ttest_rel, tstd
from torch_geometric.utils.dropout import dropout_edge
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train(model, loader, optimizer, device='cpu'):
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
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, classifier=None, device='cpu'):
    model.eval()
    preds = []
    probs = []
    truths = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            try:
                out = model(data)
            except:
                out = model(data.x, data.edge_index)
            if classifier:
                out = classifier(out[1], data.batch)

            prob = torch.sigmoid(out)  # No squeeze
            pred = (prob >= 0.5).long()

            preds.append(pred.cpu())
            probs.append(prob.cpu())
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

def update_target_encoder(student, teacher, alpha):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data

def augment_fn(x, a, soft_mask_range=0.1, noise_std=0.01, drop_edge_prob=0.1):
    u = torch.empty_like(x).uniform_(1 - soft_mask_range, 1 + soft_mask_range)
    x_aug = x * u

    if noise_std > 0:
        noise = torch.randn_like(x) * noise_std
        x_aug += noise

    a_aug, _ = dropout_edge(a, drop_edge_prob)

    return x_aug, a_aug

def pretrain_bgrl(student, teacher, projector, optimizer, unlabeled_loader, alpha=0.99, lambda_rec=0.1, device='cuda'):
    student.train()
    projector.train()
    total_loss = 0.0

    for data_u in unlabeled_loader:
        data_u = data_u.to(device)
        x_u, ei_u = data_u.x, data_u.edge_index

        x1, ei1 = augment_fn(x_u, ei_u)
        x2, ei2 = augment_fn(x_u, ei_u)
        x1, x2 = x1.to(device), x2.to(device)
        ei1, ei2 = ei1.to(device), ei2.to(device)

        x_hat1, z1 = student(x1, ei1)
        p1 = projector(z1)

        with torch.no_grad():
            _, z2 = teacher(x2, ei2)

        loss_contrastive = torch.nn.functional.mse_loss(p1, z2.detach())

        loss_reconstruction = torch.nn.functional.mse_loss(x_hat1, x1)

        loss = loss_contrastive + lambda_rec * loss_reconstruction

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_target_encoder(student, teacher, alpha)

        total_loss += loss.item()

    return total_loss

def finetune_classifier(encoder, classifier, labeled_loader, optimizer, device='cuda'):
    encoder.train()
    classifier.train()
    total_loss = 0.0
    for data in labeled_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y

        _, features = encoder(x, edge_index)

        logits = classifier(features, batch)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y.type_as(logits).view_as(logits)
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(labeled_loader)
