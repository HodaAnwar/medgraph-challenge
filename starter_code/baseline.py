"""
MedGraphDiversity Challenge - Baseline Model
=============================================

A simple GCN baseline for the multi-domain medical graph classification task.
Inspired by GMoE paper (Wang et al., NeurIPS 2023).

This baseline demonstrates concepts from DGL Lectures 1.1-4.6:
- Graph classification (Lecture 1.3)
- GCN message passing (Lecture 2.4-2.5)
- Graph pooling (Lecture 3.3)
- Aggregation methods (Lecture 3.5)
- Mini-batch training (Lecture 4.1-4.3)
- Batch normalization & dropout (Lecture 4.4-4.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from pathlib import Path

# PyTorch Geometric imports
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected


# =============================================================================
# DATA LOADING
# =============================================================================

def load_graph_from_csv(csv_path):
    """
    Load graphs from CSV format.
    
    Expected CSV columns:
    - graph_id: Unique identifier for each graph
    - node_id: Node identifier within graph
    - node_features: Comma-separated node features (or multiple columns)
    - edge_src, edge_dst: Edge endpoints
    - label: Graph label (only in train.csv)
    """
    df = pd.read_csv(csv_path)
    
    graphs = []
    graph_ids = df['graph_id'].unique()
    
    for gid in graph_ids:
        graph_df = df[df['graph_id'] == gid]
        
        # Extract node features
        if 'node_features' in graph_df.columns:
            # Features as comma-separated string
            features = graph_df['node_features'].apply(lambda x: [float(f) for f in x.split(',')])
            x = torch.tensor(np.array(features.tolist()), dtype=torch.float)
        else:
            # Features as separate columns (f0, f1, f2, ...)
            feature_cols = [c for c in graph_df.columns if c.startswith('f')]
            x = torch.tensor(graph_df[feature_cols].values, dtype=torch.float)
        
        # Extract edges
        edge_src = graph_df['edge_src'].dropna().astype(int).tolist()
        edge_dst = graph_df['edge_dst'].dropna().astype(int).tolist()
        
        if len(edge_src) > 0:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_index = to_undirected(edge_index)
        else:
            # Create minimal connectivity if no edges
            n_nodes = len(x)
            edge_index = torch.tensor([[i, i+1] for i in range(n_nodes-1)] + 
                                      [[i+1, i] for i in range(n_nodes-1)], dtype=torch.long).t()
        
        # Extract label if available
        label = None
        if 'label' in graph_df.columns:
            label = int(graph_df['label'].iloc[0])
        
        data = Data(x=x, edge_index=edge_index, num_nodes=len(x))
        data.graph_id = gid
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)
        
        graphs.append(data)
    
    return graphs


def create_synthetic_data(n_train=1000, n_test=250, save_dir='../data'):
    """
    Create synthetic multi-domain graph dataset.
    
    Generates graphs from 4 domains with different structural properties,
    following the GMoE paper's insight about graph diversity.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    def generate_graph(label, domain, graph_id):
        """Generate a single graph based on domain and label."""
        
        # Domain-specific parameters
        if domain == 0:  # Histopathology - Dense
            n_nodes = np.random.randint(150, 250)
            edge_prob = 0.08
            feature_base = [0.5 + 0.2*label, 0.5 - 0.1*label, 0.3 + 0.2*label]
        elif domain == 1:  # Retinal - Sparse/Tree
            n_nodes = np.random.randint(80, 150)
            edge_prob = 0.03
            feature_base = [0.4 + 0.15*label, 0.6 - 0.15*label, 0.4 + 0.1*label]
        elif domain == 2:  # Brain - Small-world
            n_nodes = np.random.randint(100, 200)
            edge_prob = 0.05
            feature_base = [0.5 + 0.1*label, 0.5, 0.5 + 0.15*label]
        else:  # Cell Migration - Irregular
            n_nodes = np.random.randint(50, 120)
            edge_prob = 0.04 + 0.02 * np.random.random()
            feature_base = [0.45 + 0.2*label, 0.55 - 0.1*label, 0.35 + 0.2*label]
        
        # Generate 12-dimensional node features
        full_features = feature_base * 4  # Repeat to get 12 features
        x = np.random.normal(full_features, 0.15, (n_nodes, 12))
        x = np.clip(x, 0, 1)
        
        # Generate edges
        edges_src, edges_dst = [], []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.random() < edge_prob:
                    edges_src.extend([i, j])
                    edges_dst.extend([j, i])
        
        # Ensure minimum connectivity
        if len(edges_src) < n_nodes:
            for i in range(n_nodes - 1):
                edges_src.extend([i, i+1])
                edges_dst.extend([i+1, i])
        
        return {
            'graph_id': graph_id,
            'n_nodes': n_nodes,
            'features': x,
            'edges_src': edges_src,
            'edges_dst': edges_dst,
            'label': label,
            'domain': domain
        }
    
    # Generate training data
    train_rows = []
    class_dist = [0.40, 0.35, 0.25]  # Normal, Benign, Malignant
    
    graph_id = 0
    for domain in range(4):
        n_per_domain = n_train // 4
        for label, prob in enumerate(class_dist):
            n_graphs = int(n_per_domain * prob)
            for _ in range(n_graphs):
                g = generate_graph(label, domain, graph_id)
                
                # Convert to CSV rows
                for node_idx in range(g['n_nodes']):
                    row = {
                        'graph_id': g['graph_id'],
                        'node_id': node_idx,
                        'node_features': ','.join(map(str, g['features'][node_idx])),
                        'label': g['label'],
                        'domain': g['domain']
                    }
                    # Add edges for this node
                    node_edges_src = [g['edges_src'][i] for i, s in enumerate(g['edges_src']) if s == node_idx]
                    node_edges_dst = [g['edges_dst'][i] for i, s in enumerate(g['edges_src']) if s == node_idx]
                    
                    if len(node_edges_dst) > 0:
                        row['edge_src'] = node_idx
                        row['edge_dst'] = node_edges_dst[0]
                    else:
                        row['edge_src'] = np.nan
                        row['edge_dst'] = np.nan
                    
                    train_rows.append(row)
                
                graph_id += 1
    
    train_df = pd.DataFrame(train_rows)
    train_df.to_csv(save_dir / 'train.csv', index=False)
    
    # Generate test data (similar process, no labels exposed)
    test_rows = []
    graph_id = 0
    for domain in range(4):
        n_per_domain = n_test // 4
        for label, prob in enumerate(class_dist):
            n_graphs = int(n_per_domain * prob)
            for _ in range(n_graphs):
                g = generate_graph(label, domain, graph_id)
                
                for node_idx in range(g['n_nodes']):
                    row = {
                        'graph_id': g['graph_id'],
                        'node_id': node_idx,
                        'node_features': ','.join(map(str, g['features'][node_idx])),
                    }
                    node_edges_dst = [g['edges_dst'][i] for i, s in enumerate(g['edges_src']) if s == node_idx]
                    
                    if len(node_edges_dst) > 0:
                        row['edge_src'] = node_idx
                        row['edge_dst'] = node_edges_dst[0]
                    else:
                        row['edge_src'] = np.nan
                        row['edge_dst'] = np.nan
                    
                    test_rows.append(row)
                
                graph_id += 1
    
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(save_dir / 'test.csv', index=False)
    
    # Create sample submission
    sample_sub = pd.DataFrame({
        'graph_id': range(n_test),
        'label': [0] * n_test
    })
    sample_sub.to_csv(save_dir.parent / 'submissions' / 'sample_submission.csv', index=False)
    
    print(f"Dataset created in {save_dir}")
    print(f"  Train: {len(train_df['graph_id'].unique())} graphs")
    print(f"  Test: {len(test_df['graph_id'].unique())} graphs")


# =============================================================================
# MODELS
# =============================================================================

class GCNBaseline(nn.Module):
    """
    Basic GCN for graph classification.
    
    Architecture based on DGL Lectures 2.4-2.5 (GCN) and 3.3 (pooling).
    """
    
    def __init__(self, num_features, hidden_dim=64, num_classes=3, dropout=0.5):
        super().__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        
        return x


class MultiPoolModel(nn.Module):
    """
    GCN with multiple pooling strategies (Lecture 3.3, 3.5).
    
    Combines mean, max, and sum pooling for richer graph representation.
    """
    
    def __init__(self, num_features, hidden_dim=64, num_classes=3, dropout=0.5):
        super().__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim * 3, num_classes)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        x = self.classifier(x)
        
        return x


class SimpleMoE(nn.Module):
    """
    Simple Mixture of Experts model inspired by GMoE paper.
    
    Each node dynamically selects aggregation experts based on its features.
    This implements the key insight from GMoE (Paper 3).
    """
    
    def __init__(self, num_features, hidden_dim=64, num_classes=3, num_experts=3, dropout=0.5):
        super().__init__()
        
        self.num_experts = num_experts
        
        # Multiple expert GCN layers
        self.experts = nn.ModuleList([
            GCNConv(num_features, hidden_dim) for _ in range(num_experts)
        ])
        
        # Gating network - decides expert weights per node
        self.gate = nn.Sequential(
            nn.Linear(num_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim * 3, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Compute gating weights
        gate_weights = F.softmax(self.gate(x), dim=1)
        
        # Compute expert outputs
        expert_outputs = [expert(x, edge_index) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Weighted combination
        x = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        x = self.classifier(x)
        
        return x


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y.squeeze())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    
    all_preds, all_labels = [], []
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        
        loss = F.cross_entropy(out, batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs
        
        preds = out.argmax(dim=1).cpu().numpy()
        labels = batch.y.squeeze().cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, macro_f1


def main():
    """Main training script."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data if needed
    data_dir = Path('../data')
    if not (data_dir / 'train.csv').exists():
        print("Creating synthetic dataset...")
        create_synthetic_data(save_dir=str(data_dir))
    
    # Load data
    print("Loading data...")
    train_graphs = load_graph_from_csv(data_dir / 'train.csv')
    
    # Split into train/val
    np.random.shuffle(train_graphs)
    n_val = int(len(train_graphs) * 0.15)
    val_graphs = train_graphs[:n_val]
    train_graphs = train_graphs[n_val:]
    
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")
    
    # Get dimensions
    num_features = train_graphs[0].x.shape[1]
    num_classes = 3
    
    # Create model
    model = GCNBaseline(num_features, hidden_dim=64, num_classes=num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    best_f1 = 0
    for epoch in range(1, 101):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pt')
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    
    print(f"\nBest Validation F1: {best_f1:.4f}")
    
    # Final evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    _, _, _, = evaluate(model, val_loader, device)


if __name__ == '__main__':
    main()
