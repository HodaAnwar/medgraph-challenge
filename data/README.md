# 📊 MedGraph Challenge Data Documentation

## Overview

This directory contains the cell-graph dataset for the MedGraph Challenge. Each graph represents a tissue region from histopathology images where cells are modeled as nodes and spatial relationships as edges.

---

## Graph Specification

### Adjacency Matrix (A)

The graph structure is represented in **sparse COO format** as `edge_index`:

| Attribute | Description |
|-----------|-------------|
| **Format** | `[2, num_edges]` tensor |
| **Type** | `torch.long` |
| **Access** | `data.edge_index` |
| **Properties** | Undirected (edges in both directions) |

**Construction Method**: 
1. Delaunay triangulation of cell centroids
2. Distance threshold filtering (< 50 μm)
3. Both directions stored for undirected graph

```python
# Access edge_index
edge_index = data.edge_index  # [2, E] where E = number of edges

# Convert to dense adjacency matrix
from torch_geometric.utils import to_dense_adj
A = to_dense_adj(edge_index, max_num_nodes=data.num_nodes)  # [1, N, N]
A = A.squeeze(0)  # [N, N]

# Get adjacency list
from torch_geometric.utils import to_scipy_sparse_matrix
A_sparse = to_scipy_sparse_matrix(edge_index)
```

### Node Feature Matrix (X)

| Attribute | Description |
|-----------|-------------|
| **Format** | `[num_nodes, 114]` tensor |
| **Type** | `torch.float32` |
| **Access** | `data.x` |
| **Range** | `[0, 1]` (normalized) |

**Feature Breakdown (114 dimensions)**:

| Index Range | Dimension | Feature Group | Description |
|-------------|-----------|---------------|-------------|
| 0-31 | 32 | Morphology | Cell shape descriptors |
| 32-95 | 64 | Texture | Haralick texture features |
| 96-111 | 16 | Color | LAB color histogram |
| 112-113 | 2 | Position | Normalized (x, y) coordinates |

```python
# Access node features
X = data.x  # [N, 114] where N = number of nodes

# Split into feature groups
morphology = X[:, 0:32]      # [N, 32]
texture = X[:, 32:96]        # [N, 64]
color = X[:, 96:112]         # [N, 16]
position = X[:, 112:114]     # [N, 2]
```

### Graph Label (Y)

| Value | Class | Description |
|-------|-------|-------------|
| 0 | Normal | Healthy tissue |
| 1 | Benign | Non-cancerous abnormality |
| 2 | Malignant | Cancerous tissue |

```python
# Access label
y = data.y  # [1] tensor with value 0, 1, or 2
class_name = ['Normal', 'Benign', 'Malignant'][y.item()]
```

---

## Data Files

### Directory Structure

```
data/
├── train/                    # Training graphs
│   ├── train_0001.pt
│   ├── train_0002.pt
│   └── ... (1200 files)
├── val/                      # Validation graphs  
│   ├── val_0001.pt
│   ├── val_0002.pt
│   └── ... (300 files)
├── test_public/              # Test graphs (labels hidden)
│   ├── test_0001.pt
│   ├── test_0002.pt
│   ├── test_ids.txt         # List of test graph IDs
│   └── ... (500 files)
├── test_labels.csv           # Ground truth (kept secret)
└── README.md                 # This file
```

### File Format

Each `.pt` file contains a PyTorch Geometric `Data` object:

```python
from torch_geometric.data import Data

data = torch.load('data/train/train_0001.pt')

# Contents:
# data.x           - Node features [N, 114]
# data.edge_index  - Edge indices [2, E]  
# data.y           - Graph label [1]
# data.graph_id    - Unique identifier (string)
```

---

## Statistics

### Dataset Splits

| Split | Graphs | Normal | Benign | Malignant | Avg Nodes | Avg Edges |
|-------|--------|--------|--------|-----------|-----------|-----------|
| Train | 1,200 | 400 | 400 | 400 | ~85 | ~340 |
| Val | 300 | 100 | 100 | 100 | ~85 | ~340 |
| Test | 500 | ? | ? | ? | ~85 | ~340 |

### Node & Edge Statistics

| Metric | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| Nodes per graph | 50 | 120 | 85 | 20 |
| Edges per graph | 100 | 600 | 340 | 80 |
| Node degree | 2 | 12 | 8 | 2 |

---

## Dataset Challenges

This dataset includes realistic challenges:

### 1. Feature Noise (5%)
Gaussian noise added to all features to simulate measurement error.

### 2. Distribution Shift
Test set includes samples with slight feature distribution shift to simulate:
- Different staining batches
- Scanner variations
- Tissue preparation differences

### 3. Variable Graph Sizes
Graphs range from 50 to 120 nodes, requiring models to handle variable-sized inputs.

---

## Loading Data

### Using PyTorch Geometric

```python
from utils.dataset import MedGraphDataset
from torch_geometric.loader import DataLoader

# Load dataset
train_dataset = MedGraphDataset(root='data', split='train')
val_dataset = MedGraphDataset(root='data', split='val')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Iterate
for batch in train_loader:
    x = batch.x              # [total_nodes, 114]
    edge_index = batch.edge_index  # [2, total_edges]
    y = batch.y              # [batch_size]
    batch_idx = batch.batch  # [total_nodes] - maps nodes to graphs
```

### Manual Loading

```python
import torch
from pathlib import Path

data_dir = Path('data/train')
graphs = []

for filepath in sorted(data_dir.glob('*.pt')):
    data = torch.load(filepath)
    graphs.append(data)

print(f"Loaded {len(graphs)} graphs")
```

---

## Verification

Run the data verification script to check integrity:

```bash
python utils/verify_data.py
```

This checks:
- ✅ All required files present
- ✅ Feature dimensions correct (114)
- ✅ Labels valid (0, 1, 2)
- ✅ Edge indices valid
- ✅ No NaN or Inf values

---

## Computational Requirements

The dataset is sized to ensure **< 3 hour CPU training**:

- Total graphs: 2,000 (train + val + test)
- Average nodes per graph: ~85
- Average edges per graph: ~340
- Estimated training time: ~2 hours on modern CPU

---

## Citation

If you use this dataset, please cite:

```bibtex
@misc{medgraph2025,
  title={MedGraph Challenge: Cell-Graph Classification for Histopathology},
  author={Anwar, Hoda},
  year={2025},
  url={https://github.com/HodaAnwar/medgraph-challenge}
}
```
