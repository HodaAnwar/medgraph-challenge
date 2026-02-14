"""
GraphSAGE Baseline for MedGraph Challenge

This module implements a GraphSAGE-based classifier for histopathology 
cell-graph classification with inductive learning capabilities.

Reference:
    Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation 
    Learning on Large Graphs. NeurIPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm, LayerNorm


class GraphSAGEClassifier(nn.Module):
    """
    GraphSAGE for Graph Classification.
    
    GraphSAGE samples and aggregates features from local neighborhoods,
    making it efficient and suitable for inductive learning on unseen graphs.
    
    Architecture:
        Input → [SAGEConv → Norm → ReLU → Dropout] × num_layers 
              → Global Pooling → MLP → Output
    
    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden units.
        num_classes (int): Number of output classes.
        num_layers (int): Number of SAGE layers.
        dropout (float): Dropout probability.
        aggr (str): Aggregation function ('mean', 'max', 'lstm').
        normalize (bool): Apply L2 normalization to embeddings.
        project (bool): Apply linear projection before aggregation.
        
    Example:
        >>> model = GraphSAGEClassifier(
        ...     in_channels=114,
        ...     hidden_channels=256,
        ...     num_classes=3,
        ...     num_layers=4
        ... )
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_classes: int = 3,
        num_layers: int = 4,
        dropout: float = 0.5,
        aggr: str = 'mean',
        normalize: bool = True,
        project: bool = True,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize = normalize
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # SAGE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(SAGEConv(
                hidden_channels, 
                hidden_channels,
                aggr=aggr,
                normalize=normalize,
                project=project,
            ))
            self.norms.append(BatchNorm(hidden_channels))
        
        # Classifier with skip connection from early layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # *2 for pooling concat
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass.
        
        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels].
            edge_index (Tensor): Edge indices [2, num_edges].
            batch (Tensor): Batch assignment vector [num_nodes].
            
        Returns:
            Tensor: Class logits [batch_size, num_classes].
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # SAGE layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            x = x + x_res
        
        # L2 normalization
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        
        # Global pooling: mean + max
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=-1)
        
        # Classification
        out = self.classifier(x)
        
        return out


class DeepGraphSAGE(nn.Module):
    """
    Deep GraphSAGE with advanced techniques to prevent over-smoothing.
    
    Includes:
    - Layer normalization for stability
    - Dense skip connections (DenseNet-style)
    - Learnable layer weights
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_classes: int = 3,
        num_layers: int = 6,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # SAGE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Dense connections: input channels grow with each layer
            in_dim = hidden_channels * (i + 1)
            self.convs.append(SAGEConv(in_dim, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))
        
        # Learnable layer importance weights
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        
        # Final dimension after dense connections
        final_dim = hidden_channels * num_layers
        
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )
    
    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Store all layer outputs for dense connections
        xs = [x]
        
        for conv, norm in zip(self.convs, self.norms):
            # Concatenate all previous layer outputs
            x_cat = torch.cat(xs, dim=-1)
            
            x = conv(x_cat, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            xs.append(x)
        
        # Weighted aggregation of all layer outputs (skip first input layer)
        layer_weights = F.softmax(self.layer_weights, dim=0)
        
        pooled_layers = []
        for i, x in enumerate(xs[1:]):  # Skip input projection
            pooled = global_mean_pool(x, batch)
            pooled_layers.append(pooled * layer_weights[i])
        
        x = torch.cat(pooled_layers, dim=-1)
        
        out = self.classifier(x)
        return out


class MinCutPoolSAGE(nn.Module):
    """
    GraphSAGE with differentiable pooling using MinCut objective.
    
    Learns hierarchical graph representations by progressively
    coarsening the graph structure.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_classes: int = 3,
        num_layers: int = 4,
        pool_ratio: float = 0.5,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        from torch_geometric.nn import dense_mincut_pool
        self.dense_mincut_pool = dense_mincut_pool
        
        self.dropout = dropout
        
        # First SAGE block
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.norm1 = BatchNorm(hidden_channels)
        
        # Pooling assignment network
        self.pool_conv = SAGEConv(hidden_channels, int(hidden_channels * pool_ratio))
        
        # Second SAGE block (on coarsened graph)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.norm2 = BatchNorm(hidden_channels)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )
        
        self.pool_ratio = pool_ratio
    
    def forward(self, x, edge_index, batch):
        # First convolution
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # For simplicity, use global pooling instead of full MinCut
        # (Full MinCut requires dense adjacency which is memory-intensive)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=-1)
        
        out = self.classifier(x)
        return out


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
    
    # Test GraphSAGE
    model = GraphSAGEClassifier(
        in_channels=num_features,
        hidden_channels=256,
        num_classes=3,
        num_layers=4,
    )
    
    print(f"GraphSAGE parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    out = model(batch.x, batch.edge_index, batch.batch)
    print(f"Output shape: {out.shape}")
    
    # Test Deep GraphSAGE
    model_deep = DeepGraphSAGE(
        in_channels=num_features,
        hidden_channels=128,
        num_classes=3,
        num_layers=6,
    )
    
    print(f"\nDeep GraphSAGE parameters: {sum(p.numel() for p in model_deep.parameters()):,}")
    out_deep = model_deep(batch.x, batch.edge_index, batch.batch)
    print(f"Output shape: {out_deep.shape}")
