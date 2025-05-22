
# Anomaly Detection for MDD

This repository contains code for Project II: unsupervised anomaly detection in Major Depressive Disorder (MDD) using rs-fMRI data from the REST-meta-MDD dataset. The goal is to detect deviations from normative functional connectivity patterns using graph-based autoencoder models.

---

## 📁 Directory Structure

```
.
├── preprocessing/
│   └── preprocessing.py       # Preprocessing pipeline (FC matrix, Fisher Z, ComBat)
├── models/
│   ├── gcn_autoencoder.py     # GCN-based autoencoder
│   ├── gat_autoencoder.py     # GAT-based autoencoder
│   └── chebnet_autoencoder.py # ChebNet-based autoencoder
```

---

## 📊 Dataset

- Dataset: [REST-meta-MDD](http://rfmri.org/REST-meta-MDD)
- Data type: rs-fMRI scans from ~1,300 subjects across 25 sites
- Input: 200×200 functional connectivity matrices
- Labels: Site information and MDD/control labels (used only for evaluation)

---

## 🧪 Preprocessing

```bash
python preprocessing/preprocessing.py \
    --data_csv data/subject_list.csv \
    --atlas_path atlas/craddock_200.nii.gz \
    --output_path output/
```

- ROI time series extraction (via NiftiLabelsMasker)
- Pearson correlation → FC matrix
- Fisher Z transformation
- ComBat harmonization

---

## 🔧 Baseline Models

Each model is implemented as an **autoencoder**:
- Input: Graph with ROI node features and FC-based edge weights
- Output: Reconstructed node features
- Anomaly score: MSE between input and output features

| Model      | Layer Type        |
|------------|-------------------|
| GCN        | Spectral convolution |
| GAT        | Attention-based convolution |
| ChebNet    | Chebyshev polynomial convolution |

---

## 🧩 Dependencies

- `nilearn`, `nibabel`
- `neurocombat-sklearn`
- `torch`, `torch_geometric`
- `numpy`, `pandas`, `scikit-learn`


