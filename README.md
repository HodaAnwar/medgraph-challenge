# 🧬 MedGraphDiversity Challenge: Adaptive GNNs for Heterogeneous Medical Graphs

> **GNNs for Rising Stars 2026 - Mini-Competition**
> 
> Inspired by [Graph Mixture of Experts (GMoE)](https://arxiv.org/abs/2304.02806) - Wang et al., NeurIPS 2023

---

## 🎯 Challenge Overview

### The Problem: Real-World Graphs Are Diverse!

In medical imaging, graphs constructed from different organs, imaging modalities, and patient conditions have **vastly different structures**. The GMoE paper showed that **different nodes benefit from different aggregation approaches**.

| Graph Type | Characteristics | Example |
|------------|-----------------|---------|
| **Dense clusters** | High connectivity, many edges | Tumor regions with tightly packed cells |
| **Sparse networks** | Few connections, spread out | Normal tissue with regular spacing |
| **Hub-and-spoke** | Few high-degree nodes | Vascular networks with branching |
| **Chain-like** | Sequential connectivity | Tissue boundaries, membranes |

### Your Task

Build a GNN that can **adaptively handle diverse graph structures** for medical image classification.

- **Input:** Graphs from 4 different medical imaging domains (mixed together)
- **Output:** Disease classification (3 classes: Normal, Benign, Malignant)
- **The twist:** Your model must perform well across **ALL** graph types!

---

## 📊 Dataset

### Four Medical Domains

| Domain | Graph Structure | Avg Nodes | Challenge |
|--------|-----------------|-----------|-----------|
| **Histopathology** | Dense, clustered | 150-250 | High connectivity |
| **Retinal Vessels** | Tree-like, sparse | 80-150 | Long-range dependencies |
| **Brain Connectivity** | Small-world | 100-200 | Community structure |
| **Cell Migration** | Dynamic, irregular | 50-120 | Varying local patterns |

### Statistics

| Split | Graphs | Class Distribution |
|-------|--------|-------------------|
| Train | 1000 | 40% Normal, 35% Benign, 25% Malignant |
| Test | 250 | Hidden |

### Data Files

```
data/
├── train.csv       # Training graph data
└── test.csv        # Test graph data (no labels)
```

---

## 🏆 Evaluation Metric

### Macro F1-Score

```
Macro F1 = (F1_Normal + F1_Benign + F1_Malignant) / 3
```

**Why Macro F1?**
- Handles class imbalance fairly
- Penalizes models that ignore minority classes
- Clinically relevant: missing malignant cases is costly

### Baseline Performance

| Model | Macro F1 | Notes |
|-------|----------|-------|
| Random | 0.33 | - |
| 2-layer GCN | 0.58 | Struggles with diversity |
| **Target** | **0.75+** | Adaptive methods needed |

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/gnn-challenge.git
cd gnn-challenge

# Install dependencies
pip install -r starter_code/requirements.txt

# Run baseline
cd starter_code
python baseline.py
```

---

## 📁 Repository Structure

```
gnn-challenge/
├── data/
│   ├── train.csv
│   └── test.csv
├── submissions/
│   └── sample_submission.csv
├── starter_code/
│   ├── baseline.py
│   └── requirements.txt
├── README.md
├── scoring_script.py
└── LICENSE
```

---

## 💡 Approaches (DGL Lectures 1.1-4.6)

| Lecture | Concept | Application |
|---------|---------|-------------|
| 1.1-1.3 | Graph fundamentals | Graph classification task |
| 2.4-2.5 | GCN message passing | Baseline model |
| 3.3 | Graph pooling | Mean/Max/Sum pooling |
| 3.5 | Aggregation methods | Different aggregators |
| 4.1-4.3 | Mini-batch, sampling | GraphSAGE |
| 4.4-4.5 | BatchNorm, Dropout | Regularization |

### GMoE-Inspired Ideas (from Paper 3)

1. **Multiple Experts** - Different aggregation strategies
2. **Multi-Hop Experts** - 1-hop, 2-hop, 3-hop neighborhoods
3. **Learnable Gating** - Node-specific expert selection

---

## 📜 Rules

1. **No external data** - Only use provided dataset
2. **Parameter limit** - Maximum 1,000,000 parameters
3. **Must use GNN** - At least one message-passing layer
4. **Inference time** - < 120 seconds on CPU

---

## 📋 Submission Format

Submit a CSV file to `submissions/` folder:

```csv
graph_id,label
0,1
1,0
2,2
...
```

Labels: 0 = Normal, 1 = Benign, 2 = Malignant

---

## 🏅 Prizes

- 🥇 **1st Place**: Co-authorship on NeurIPS 2026 paper
- 🥈 **2nd/3rd Place**: Featured solution write-up
- 🏅 **Best MoE Implementation**: Special recognition

---

## 📚 References

1. **GMoE Paper**: [Graph Mixture of Experts](https://arxiv.org/abs/2304.02806) - Wang et al., NeurIPS 2023
2. **DGL Course**: Deep Graph Learning Lectures 1.1-4.6

---

## 📧 Contact

For questions, open an issue in this repository.

---

**Good luck! 🧬**
