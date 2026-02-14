"""
Graph Attention Network (GAT) Baseline for MedGraph Challenge

This module implements a GAT-based classifier with multi-head attention
for histopathology cell-graph classification.

Reference:
    Veličković, P., et al. (2018). Graph Attention Networks. ICLR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.nn import BatchNorm


class GATClassifier(nn.Module):
    """
    Graph Attention Network for Graph Classification.
    
    Uses multi-head attention to learn adaptive edge weights,
    allowing the model to focus on the most relevant neighbors.
    
    Architecture:
        Input → [GATConv → BatchNorm → ELU → Dropout] × num_layers 
              → Global Pooling → MLP → Output
    
    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden units per attention head.
        num_classes (int): Number of output classes.
        num_layers (int): Number of GAT layers.
        heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        attention_dropout (float): Dropout on attention weights.
        v2 (bool): Use GATv2 (dynamic attention) instead of GAT.
        
    Example:
        >>> model = GATClassifier(
        ...     in_channels=114,
        ...     hidden_channels=64,
        ...     num_classes=3,
        ...     num_layers=4,
        ...     heads=4
        ... )
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 3,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.5,
        attention_dropout: float = 0.3,
        v2: bool = True,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        
        GATLayer = GATv2Conv if v2 else GATConv
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels * heads)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_channels * heads
            out_dim = hidden_channels
            
            # Last layer: single head, no concat
            if i == num_layers - 1:
                self.convs.append(GATLayer(
                    in_dim, out_dim, 
                    heads=1, 
                    concat=False,
                    dropout=attention_dropout,
                ))
                self.bns.append(BatchNorm(out_dim))
            else:
                self.convs.append(GATLayer(
                    in_dim, out_dim, 
                    heads=heads, 
                    concat=True,
                    dropout=attention_dropout,
                ))
                self.bns.append(BatchNorm(out_dim * heads))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # *2 for concat pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )
    
    def forward(self, x, edge_index, batch, return_attention=False):
        """
        Forward pass.
        
        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels].
            edge_index (Tensor): Edge indices [2, num_edges].
            batch (Tensor): Batch assignment vector [num_nodes].
            return_attention (bool): If True, return attention weights.
            
        Returns:
            Tensor: Class logits [batch_size, num_classes].
            (Optional) List[Tensor]: Attention weights from each layer.
        """
        attention_weights = []
        
        # Input projection
        x = self.input_proj(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if return_attention:
                x, (edge_index_out, alpha) = conv(
                    x, edge_index, return_attention_weights=True
                )
                attention_weights.append(alpha)
            else:
                x = conv(x, edge_index)
            
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling: concatenate mean and max
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=-1)
        
        # Classification
        out = self.classifier(x)
        
        if return_attention:
            return out, attention_weights
        return out
    
    def get_attention_weights(self, x, edge_index, batch):
        """
        Get attention weights for visualization.
        
        Returns:
            List[Tensor]: Attention weights from each layer.
        """
        _, attention_weights = self.forward(x, edge_index, batch, return_attention=True)
        return attention_weights


class HierarchicalGAT(nn.Module):
    """
    Hierarchical GAT with node-level and graph-level attention.
    
    This model applies attention at two levels:
    1. Node-level: Learn which neighbors are important for each node
    2. Graph-level: Learn which nodes are important for graph classification
    
    Inspired by Hierarchical Graph Attention Networks (HAN).
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 3,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.dropout = dropout
        
        # Node-level GAT
        self.input_proj = nn.Linear(in_channels, hidden_channels * heads)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_channels * heads
            self.convs.append(GATv2Conv(in_dim, hidden_channels, heads=heads, concat=True))
            self.bns.append(BatchNorm(hidden_channels * heads))
        
        # Graph-level attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1, bias=False),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )
    
    def forward(self, x, edge_index, batch):
        # Input projection
        x = self.input_proj(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Node-level GAT
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level attention pooling
        # Compute attention scores for each node
        attn_scores = self.attention_pool(x)  # [num_nodes, 1]
        
        # Softmax within each graph
        attn_weights = self._segment_softmax(attn_scores, batch)
        
        # Weighted sum of node features
        x_weighted = x * attn_weights
        
        # Sum over nodes in each graph
        num_graphs = batch.max().item() + 1
        graph_repr = torch.zeros(num_graphs, x.size(1), device=x.device)
        graph_repr.scatter_add_(0, batch.unsqueeze(-1).expand_as(x_weighted), x_weighted)
        
        # Classification
        out = self.classifier(graph_repr)
        
        return out
    
    def _segment_softmax(self, scores, batch):
        """Compute softmax within each segment (graph)."""
        # Subtract max for numerical stability
        max_scores = torch.zeros(batch.max() + 1, device=scores.device)
        max_scores.scatter_reduce_(0, batch, scores.squeeze(), reduce='amax')
        scores = scores - max_scores[batch].unsqueeze(-1)
        
        # Exp and normalize within segments
        exp_scores = torch.exp(scores)
        sum_exp = torch.zeros(batch.max() + 1, device=scores.device)
        sum_exp.scatter_add_(0, batch, exp_scores.squeeze())
        
        return exp_scores / (sum_exp[batch].unsqueeze(-1) + 1e-8)


if __name__ == '__main__':
    from torch_geometric.data import Data, Batch
    
    # Create dummy data
    num_nodes = 100
    num_edges = 500
    num_features = 114
    num_graphs = 4
    
    data_list = []
    for i in range(num_graphs):
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        y = torch.tensor([i % 3])
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    
    batch = Batch.from_data_list(data_list)
    
    # Test GAT
    model = GATClassifier(
        in_channels=num_features,
        hidden_channels=64,
        num_classes=3,
        num_layers=4,
        heads=4,
    )
    
    print(f"GAT parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    out = model(batch.x, batch.edge_index, batch.batch)
    print(f"Output shape: {out.shape}")
    
    # Test with attention weights
    out, attns = model(batch.x, batch.edge_index, batch.batch, return_attention=True)
    print(f"Attention weights shapes: {[a.shape for a in attns]}")
    
    # Test Hierarchical GAT
    model_hier = HierarchicalGAT(
        in_channels=num_features,
        hidden_channels=64,
        num_classes=3,
        num_layers=4,
        heads=4,
    )
    
    print(f"\nHierarchical GAT parameters: {sum(p.numel() for p in model_hier.parameters()):,}")
    out_hier = model_hier(batch.x, batch.edge_index, batch.batch)
    print(f"Output shape: {out_hier.shape}")
