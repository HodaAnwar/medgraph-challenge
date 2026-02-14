# 🔬 MedGraph Challenge

<div align="center">

![MedGraph Banner](docs/banner.png)

**Classify Histopathology Cell-Graphs into Tissue Types**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.4+-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)

[🏆 Leaderboard](https://hodaanwar.github.io/medgraph-challenge/) | [📊 Dataset](#dataset) | [🚀 Getting Started](#getting-started) | [📝 Submission](#submission-guide)

</div>

---

## ⚠️ Competition Rules

> **IMPORTANT**: Please read these rules carefully before participating.

| Rule | Description |
|------|-------------|
| 🎯 **One Submission Only** | Each participant is allowed **exactly ONE submission**. Choose wisely! |
| 🏆 **Kaggle-Style Ranking** | Tied scores share the same rank (e.g., if two teams tie for 1st, both get rank 1, next gets rank 3) |
| ⏱️ **CPU Training** | Full training must complete in **< 3 hours on CPU** |
| 🚫 **No Pre-trained GNNs** | Models must be trained from scratch on provided data |

---

## 🎯 Challenge Overview

The **MedGraph Challenge** tasks participants with classifying histopathology images represented as **cell-graphs** into three tissue categories:

| Class | Description | Clinical Significance |
|-------|-------------|----------------------|
| 🟢 **Normal** | Healthy tissue architecture | Baseline reference |
| 🟡 **Benign** | Non-cancerous abnormalities | Monitoring required |
| 🔴 **Malignant** | Cancerous tissue patterns | Immediate intervention |

### Why Graph Neural Networks for Histopathology?

Traditional CNNs analyze pixel patterns, but **cell-graphs** capture the **spatial relationships** between cells—critical for understanding tissue architecture and disease progression.

```
Cell-Graph Representation:
• Nodes = Individual cells (features: morphology, texture, color)
• Edges = Spatial relationships between neighboring cells
• Graph = Tissue microenvironment structure
```

---

## 📊 Dataset

### Graph Specification (Required)

Each graph in this challenge is explicitly defined by:

#### **Adjacency Matrix (A)**
- **Type**: Sparse COO format stored as `edge_index` tensor
- **Shape**: `[2, num_edges]` - each column represents an edge (source, target)
- **Properties**: Undirected graph (edges stored in both directions)
- **Construction**: Delaunay triangulation with 50μm distance threshold
- **Access**: `data.edge_index` in PyG Data object

```python
# Example: Convert to dense adjacency matrix
from torch_geometric.utils import to_dense_adj
A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)  # [1, N, N]
```

#### **Node Feature Matrix (X)**
- **Type**: Dense float tensor
- **Shape**: `[num_nodes, 114]` - one row per cell
- **Access**: `data.x` in PyG Data object

```python
# Example: Access node features
X = data.x  # [N, 114] tensor
```

### Dataset Statistics

| Split | Graphs | Normal | Benign | Malignant | Avg Nodes | Avg Edges |
|-------|--------|--------|--------|-----------|-----------|-----------|
| Train | 1,200  | 400    | 400    | 400       | ~85       | ~340      |
| Val   | 300    | 100    | 100    | 100       | ~85       | ~340      |
| Test  | 500    | ?      | ?      | ?         | ~85       | ~340      |

> **Note**: Dataset is sized for **< 3 hour CPU training** as per competition guidelines.

### Node Features (X) - 114 Dimensions

| Feature Group | Dimensions | Description |
|---------------|------------|-------------|
| Morphology | 32 | Cell shape descriptors (area, perimeter, eccentricity, etc.) |
| Texture | 64 | Haralick texture features (contrast, correlation, energy, etc.) |
| Color | 16 | Color histogram in LAB space |
| Position | 2 | Normalized (x, y) coordinates within tissue region |
| **Total** | **114** | Concatenated feature vector |

### Dataset Challenges

This dataset includes realistic challenges common in medical imaging:

| Challenge | Description |
|-----------|-------------|
| 🔀 **Class Imbalance** | Real-world distribution with slight imbalance |
| 📊 **Feature Noise** | 5% Gaussian noise added to simulate measurement error |
| 🔗 **Edge Sparsity** | Variable graph density (some graphs have fewer edges) |
| 📈 **Distribution Shift** | Test set includes samples from different staining batches |

### Data Format

Each `.pt` file contains a PyTorch Geometric `Data` object:

```python
Data(
    x=[num_nodes, 114],        # Node feature matrix X
    edge_index=[2, num_edges], # Adjacency in COO format (A)
    y=[1],                     # Graph label (0: Normal, 1: Benign, 2: Malignant)
    graph_id='train_0001'      # Unique identifier
)
```

### Download

```bash
# Download and extract dataset
python utils/download_data.py

# Verify data format
python utils/verify_data.py
```

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/HodaAnwar/medgraph-challenge.git
cd medgraph-challenge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torch-geometric>=2.4.0
torch-scatter
torch-sparse
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
wandb>=0.15.0  # Optional: for experiment tracking
```

### Quick Start

```python
from utils.dataset import MedGraphDataset
from baselines.gcn_baseline import GCNClassifier
from evaluation.metrics import evaluate_model

# Load data
train_dataset = MedGraphDataset(root='data', split='train')
val_dataset = MedGraphDataset(root='data', split='val')

# Initialize model
model = GCNClassifier(
    in_channels=114,
    hidden_channels=256,
    num_classes=3,
    num_layers=4
)

# Train and evaluate
# See notebooks/01_baseline_training.ipynb for full example
```

---

## 🏗️ Baseline Models

We provide three baseline implementations:

| Model | Val Accuracy | Val Macro-F1 | Parameters |
|-------|-------------|--------------|------------|
| GCN | 78.3% | 0.776 | 412K |
| GAT | 81.2% | 0.809 | 523K |
| GraphSAGE | 79.8% | 0.794 | 456K |

### Running Baselines

```bash
# Train GCN baseline
python baselines/train.py --model gcn --epochs 100 --lr 0.001

# Train GAT baseline
python baselines/train.py --model gat --epochs 100 --lr 0.0005

# Train GraphSAGE baseline
python baselines/train.py --model sage --epochs 100 --lr 0.001
```

---

## 📝 Submission Guide

### 🔐 Secure Encrypted Submission System

This competition uses an **encrypted submission system** to keep test labels completely hidden while enabling fully automated evaluation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOU (Public)                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Create predictions.csv                                       │
│  2. Encrypt with PUBLIC KEY → your_team.enc                     │
│  3. Submit via Pull Request                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GITHUB ACTIONS (Secret)                       │
├─────────────────────────────────────────────────────────────────┤
│  4. Decrypt with PRIVATE KEY (GitHub Secrets)                   │
│  5. Evaluate against hidden test labels                          │
│  6. Update leaderboard (2-5 minutes)                             │
└─────────────────────────────────────────────────────────────────┘
```

### ⚠️ ONE SUBMISSION ONLY

> **Each participant is allowed exactly ONE submission attempt.**  
> Make sure your submission is final before encrypting and submitting.  
> There are NO exceptions to this rule.

### Step 1: Create Your Predictions

```csv
graph_id,prediction
test_0001,0
test_0002,2
test_0003,1
...
```

Where: `0 = Normal`, `1 = Benign`, `2 = Malignant`

### Step 2: Validate Locally (REQUIRED)

```bash
python submission/validate.py --submission your_predictions.csv
```

### Step 3: Encrypt Your Submission

```bash
# Install cryptography package
pip install cryptography

# Encrypt your predictions
python encryption/encrypt.py \
    your_predictions.csv \
    encryption/public_key.pem \
    submissions/your_team_name.enc
```

### Step 4: Submit via Pull Request

1. **Fork** this repository
2. Add your `.enc` file to `submissions/` folder
3. Create a **Pull Request**
4. Wait 2-5 minutes for automated evaluation
5. Results posted as PR comment + leaderboard update!

### 🔒 Security Guarantees

- ✅ Test labels remain completely hidden
- ✅ Private key never exposed in logs
- ✅ Encrypted submissions unreadable by anyone
- ✅ Fully automated evaluation

📖 [Full Encryption Documentation →](encryption/README.md)

---

## 🏆 Evaluation Metrics

### Primary Metric: Macro F1-Score

$$\text{Macro F1} = \frac{1}{3} \sum_{c \in \{N, B, M\}} F1_c$$

### Secondary Metrics

- **Per-class Accuracy**: Performance breakdown by tissue type
- **Confusion Matrix**: Error analysis visualization
- **Balanced Accuracy**: Accounts for class imbalance

### Evaluation Script

```bash
python evaluation/evaluate.py \
    --predictions your_submission.csv \
    --ground_truth data/test_labels.csv  # Only available after challenge ends
```

---

## 🏅 Leaderboard

| Rank | Team | Model | Macro F1 | Accuracy | Submission Date |
|------|------|-------|----------|----------|-----------------|
| 🥇 | - | - | - | - | - |
| 🥈 | - | - | - | - | - |
| 🥉 | - | - | - | - | - |

**[View Full Leaderboard →](https://hodaanwar.github.io/medgraph-challenge/)**

---

## 📁 Repository Structure

```
medgraph-challenge/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
│
├── data/                        # Dataset directory
│   ├── train/                   # Training graphs (.pt files)
│   ├── val/                     # Validation graphs
│   ├── test_public/             # Test graphs (no labels)
│   ├── test_labels.csv          # Hidden test labels (for CI only)
│   └── README.md                # Data documentation
│
├── baselines/                   # Baseline model implementations
│   ├── gcn_baseline.py          # Graph Convolutional Network
│   ├── gat_baseline.py          # Graph Attention Network
│   ├── sage_baseline.py         # GraphSAGE
│   └── train.py                 # Training script
│
├── evaluation/                  # Evaluation scripts
│   ├── evaluate.py              # Main evaluation script
│   └── metrics.py               # Metric implementations
│
├── encryption/                  # 🔐 Secure submission system
│   ├── encrypt.py               # Encryption script (for participants)
│   ├── decrypt.py               # Decryption script (CI only)
│   ├── public_key.pem           # Public key (use this to encrypt!)
│   └── README.md                # Encryption documentation
│
├── submission/                  # Submission utilities
│   ├── validate.py              # Submission validator
│   └── example_submission.csv   # Example format
│
├── submissions/                 # 📥 Encrypted submissions (.enc files)
│   └── .gitkeep
│
├── leaderboard/                 # Leaderboard files
│   ├── leaderboard.json         # Current standings
│   ├── index.html               # GitHub Pages leaderboard
│   └── update_leaderboard.py    # Leaderboard updater
│
├── utils/                       # Utility functions
│   ├── dataset.py               # PyG dataset class
│   └── generate_sample_data.py  # Data generation
│
├── notebooks/                   # Tutorial notebooks
│   └── 01_baseline_training.ipynb
│
└── .github/workflows/           # GitHub Actions
    └── evaluate.yml             # Auto-evaluation on PR
```
├── baselines/                   # Baseline model implementations
│   ├── __init__.py
│   ├── gcn_baseline.py          # Graph Convolutional Network
│   ├── gat_baseline.py          # Graph Attention Network
│   ├── sage_baseline.py         # GraphSAGE
│   ├── train.py                 # Training script
│   └── README.md                # Model documentation
│
├── evaluation/                  # Evaluation scripts
│   ├── __init__.py
│   ├── evaluate.py              # Main evaluation script
│   └── metrics.py               # Metric implementations
│
├── submission/                  # Submission utilities
│   ├── validate.py              # Submission validator
│   └── example_submission.csv   # Example format
│
├── leaderboard/                 # Leaderboard files
│   ├── leaderboard.json         # Current standings
│   ├── index.html               # GitHub Pages leaderboard
│   └── update_leaderboard.py    # Leaderboard updater
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── dataset.py               # PyG dataset class
│   ├── download_data.py         # Data download script
│   └── visualization.py         # Plotting utilities
│
├── notebooks/                   # Tutorial notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   └── 03_advanced_techniques.ipynb
│
├── docs/                        # Documentation
│   ├── banner.png
│   └── CONTRIBUTING.md
│
└── .github/workflows/           # GitHub Actions
    ├── evaluate.yml             # Auto-evaluation on PR
    └── update_leaderboard.yml   # Leaderboard update
```

---

## 🔬 Research Context

This challenge is inspired by recent advances in **Graph Neural Networks for Computational Pathology**:

- [CGC-Net](https://arxiv.org/abs/1909.01142) - Cell-Graph Convolutional Network
- [HACT-Net](https://arxiv.org/abs/2007.00584) - Hierarchical Cell-to-Tissue Graph
- [GNN-MIL](https://arxiv.org/abs/2103.10115) - Graph-based Multiple Instance Learning

### Citing This Challenge

```bibtex
@misc{medgraph2025,
  title={MedGraph Challenge: Cell-Graph Classification for Histopathology},
  author={Anwar, Hoda},
  year={2025},
  url={https://github.com/HodaAnwar/medgraph-challenge}
}
```

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/HodaAnwar/medgraph-challenge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/HodaAnwar/medgraph-challenge/discussions)
- **Email**: [challenge@example.com](mailto:challenge@example.com)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Good luck! May your gradients flow and your graphs classify! 🧬**

</div>
