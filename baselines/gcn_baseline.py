"""
Graph Convolutional Network (GCN) Baseline for MedGraph Challenge

This module implements a GCN-based classifier for histopathology cell-graph
classification following Kipf & Welling (2017).

Reference:
    Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with 
    Graph Convolutional Networks. ICLR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm


class GCNClassifier(nn.Module):
    """
    Graph Convolutional Network for Graph Classification.
    
    Architecture:
        Input → [GCNConv → BatchNorm → ReLU → Dropout] × num_layers 
              → Global Pooling → MLP → Output
    
    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden units in each layer.
        num_classes (int): Number of output classes.
        num_layers (int): Number of GCN layers.
        dropout (float): Dropout probability.
        pooling (str): Global pooling method ('mean', 'max', 'add', 'concat').
        
    Example:
        >>> model = GCNClassifier(
        ...     in_channels=114,
        ...     hidden_channels=256,
        ...     num_classes=3,
        ...     num_layers=4
        ... )
        >>> # Forward pass with batched graphs
        >>> out = model(data.x, data.edge_index, data.batch)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_classes: int = 3,
        num_layers: int = 4,
        dropout: float = 0.5,
        pooling: str = 'mean',
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Classifier head
        if pooling == 'concat':
            classifier_input = hidden_channels * 3  # mean + max + add
        else:
            classifier_input = hidden_channels
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_channels),
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
        
        # GCN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_res = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0:
                x = x + x_res
        
        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling == 'concat':
            x = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
                global_add_pool(x, batch),
            ], dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classification
        out = self.classifier(x)
        
        return out
    
    def get_embeddings(self, x, edge_index, batch):
        """
        Get graph-level embeddings before classification.
        
        Useful for visualization and analysis.
        """
        x = self.input_proj(x)
        x = F.relu(x)
        
        for conv, bn in zip(self.convs, self.bns):
            x_res = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = x + x_res
        
        if self.pooling == 'concat':
            x = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
                global_add_pool(x, batch),
            ], dim=-1)
        else:
            x = global_mean_pool(x, batch)
        
        return x


class GCNWithJK(nn.Module):
    """
    GCN with Jumping Knowledge for improved deep graph learning.
    
    Combines representations from all layers for final prediction,
    which can help with over-smoothing in deep GNNs.
    
    Reference:
        Xu, K., et al. (2018). Representation Learning on Graphs with 
        Jumping Knowledge Networks. ICML.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_classes: int = 3,
        num_layers: int = 4,
        dropout: float = 0.5,
        jk_mode: str = 'cat',  # 'cat', 'max', 'lstm'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.jk_mode = jk_mode
        
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Jumping Knowledge aggregation
        if jk_mode == 'cat':
            jk_channels = hidden_channels * num_layers
        elif jk_mode == 'max':
            jk_channels = hidden_channels
        elif jk_mode == 'lstm':
            self.jk_lstm = nn.LSTM(
                hidden_channels, hidden_channels, 
                batch_first=True, bidirectional=True
            )
            jk_channels = hidden_channels * 2
        else:
            raise ValueError(f"Unknown JK mode: {jk_mode}")
        
        self.classifier = nn.Sequential(
            nn.Linear(jk_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )
    
    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Collect representations from each layer
        xs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(global_mean_pool(x, batch))
        
        # Jumping Knowledge aggregation
        if self.jk_mode == 'cat':
            x = torch.cat(xs, dim=-1)
        elif self.jk_mode == 'max':
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.jk_mode == 'lstm':
            x = torch.stack(xs, dim=1)  # [batch, layers, hidden]
            x, _ = self.jk_lstm(x)
            x = x[:, -1, :]  # Take last hidden state
        
        out = self.classifier(x)
        return out


if __name__ == '__main__':
    # Test the model
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
    
    # Test GCN
    model = GCNClassifier(
        in_channels=num_features,
        hidden_channels=256,
        num_classes=3,
        num_layers=4,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    out = model(batch.x, batch.edge_index, batch.batch)
    print(f"Output shape: {out.shape}")  # [4, 3]
    
    # Test with JK
    model_jk = GCNWithJK(
        in_channels=num_features,
        hidden_channels=256,
        num_classes=3,
        num_layers=4,
        jk_mode='cat',
    )
    
    print(f"JK Model parameters: {sum(p.numel() for p in model_jk.parameters()):,}")
    
    out_jk = model_jk(batch.x, batch.edge_index, batch.batch)
    print(f"JK Output shape: {out_jk.shape}")
