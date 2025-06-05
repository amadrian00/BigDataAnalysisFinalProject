# Graph Autoencoder Project

## Project 18 

Unsupervised anomaly detection in Major Depressive Disorder (MDD) using rs-fMRI data from the REST-meta-MDD dataset. The goal is to detect deviations from normative functional connectivity patterns using graph-based autoencoder models.

## Overview

This project implements various graph autoencoder architectures for learning node embeddings and reconstructing graph structures. It includes models based on Chebyshev networks (ChebNet), Graph Attention Networks (GAT), and Graph Convolutional Networks (GCN), each with an autoencoder framework. The goal is to explore different graph neural network (GNN) techniques for unsupervised representation learning.

---

## Project Structure
```
├── models/
│   ├── chebnet_autoencoder.py      # Chebyshev Network-based autoencoder model
│   ├── gat_autoencoder.py          # Graph Attention Network autoencoder model
│   ├── gcn_autoencoder.py          # Graph Convolutional Network autoencoder model
│   └── projector.py                # Module projeciton for SimSiam
│
├── preprocessing/
│   └── preprocessing.py            # Data preprocessing utilities
│
├── train_test.py                   # Script to train and evaluate the models
├── main.py                         # Main script to build the graphs and run the project pipeline
└── README.md                       # Project documentation
```

---

## Features

- **Multiple GNN autoencoder models**: Compare ChebNet, GAT, and GCN based autoencoders.
- **Flexible preprocessing**: Includes graph data preparation and feature engineering.
- **Training and evaluation**: Easily train models and evaluate performance on given datasets.
- **Embedding projection**: Additional embedding manipulation via the projector module.

---

## Usage
- Running `main.py` is enough to run all the training and the test battery.

## Contact
amadrian@korea.ac.kr
