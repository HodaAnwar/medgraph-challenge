"""
MedGraph Dataset - PyTorch Geometric Dataset for Histopathology Cell-Graphs

This module provides a standardized PyG dataset class for loading and processing
cell-graph data for the MedGraph Challenge.
"""

import os
import os.path as osp
from typing import Callable, List, Optional, Tuple

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm


class MedGraphDataset(InMemoryDataset):
    """
    MedGraph Cell-Graph Dataset for Histopathology Classification.
    
    Each graph represents a tissue region where:
    - Nodes represent individual cells
    - Edges represent spatial relationships between cells
    - Graph label indicates tissue type (0: Normal, 1: Benign, 2: Malignant)
    
    Args:
        root (str): Root directory where the dataset should be saved.
        split (str): One of 'train', 'val', or 'test'.
        transform (callable, optional): A function/transform that takes in a
            Data object and returns a transformed version.
        pre_transform (callable, optional): A function/transform that takes in
            a Data object and returns a transformed version.
        pre_filter (callable, optional): A function that takes in a Data object
            and returns a boolean value, indicating whether the data object
            should be included in the final dataset.
    
    Node Features (114 dimensions):
        - Morphology features (32): Shape descriptors
        - Texture features (64): Haralick texture features
        - Color features (16): LAB color histogram
        - Position (2): Normalized (x, y) coordinates
    
    Example:
        >>> dataset = MedGraphDataset(root='data', split='train')
        >>> print(f'Number of graphs: {len(dataset)}')
        >>> print(f'Number of classes: {dataset.num_classes}')
        >>> print(f'Node feature dimension: {dataset.num_node_features}')
        >>> 
        >>> # Access a single graph
        >>> data = dataset[0]
        >>> print(f'Nodes: {data.num_nodes}, Edges: {data.num_edges}')
    """
    
    # Class names for reference
    CLASS_NAMES = ['Normal', 'Benign', 'Malignant']
    
    # Feature dimensions
    MORPHOLOGY_DIM = 32
    TEXTURE_DIM = 64
    COLOR_DIM = 16
    POSITION_DIM = 2
    TOTAL_FEATURES = MORPHOLOGY_DIM + TEXTURE_DIM + COLOR_DIM + POSITION_DIM  # 114
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        assert split in ['train', 'val', 'test', 'test_public'], \
            f"Split must be one of 'train', 'val', 'test', 'test_public', got {split}"
        
        self.split = split if split != 'test' else 'test_public'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.split)
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed', self.split)
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw graph files."""
        if osp.exists(self.raw_dir):
            return [f for f in os.listdir(self.raw_dir) if f.endswith('.pt')]
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']
    
    def download(self):
        """Download the dataset if not present."""
        # Data should be downloaded using utils/download_data.py
        if len(self.raw_file_names) == 0:
            raise FileNotFoundError(
                f"No data found in {self.raw_dir}. "
                "Please run 'python utils/download_data.py' first."
            )
    
    def process(self):
        """Process raw graph files into a single processed file."""
        data_list = []
        
        for filename in tqdm(self.raw_file_names, desc=f'Processing {self.split}'):
            filepath = osp.join(self.raw_dir, filename)
            data = torch.load(filepath)
            
            # Validate data format
            if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
                print(f"Warning: Skipping {filename} - invalid format")
                continue
            
            # Apply pre-filter if specified
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            # Apply pre-transform if specified
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)
        
        self.save(data_list, self.processed_paths[0])
    
    @property
    def num_classes(self) -> int:
        return 3
    
    @property
    def num_node_features(self) -> int:
        return self.TOTAL_FEATURES
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.
        
        Returns:
            torch.Tensor: Inverse frequency weights for each class.
        """
        labels = [data.y.item() for data in self]
        class_counts = np.bincount(labels, minlength=self.num_classes)
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_class_distribution(self) -> dict:
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            dict: Dictionary mapping class names to counts.
        """
        labels = [data.y.item() for data in self]
        counts = np.bincount(labels, minlength=self.num_classes)
        return {name: int(count) for name, count in zip(self.CLASS_NAMES, counts)}
    
    def statistics(self) -> dict:
        """
        Compute dataset statistics.
        
        Returns:
            dict: Dictionary containing various statistics.
        """
        num_nodes = [data.num_nodes for data in self]
        num_edges = [data.num_edges for data in self]
        
        return {
            'num_graphs': len(self),
            'num_classes': self.num_classes,
            'num_features': self.num_node_features,
            'class_distribution': self.get_class_distribution(),
            'nodes': {
                'min': min(num_nodes),
                'max': max(num_nodes),
                'mean': np.mean(num_nodes),
                'std': np.std(num_nodes),
            },
            'edges': {
                'min': min(num_edges),
                'max': max(num_edges),
                'mean': np.mean(num_edges),
                'std': np.std(num_edges),
            }
        }


def create_data_loaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        root (str): Root directory of the dataset.
        batch_size (int): Batch size for training.
        num_workers (int): Number of worker processes for data loading.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = MedGraphDataset(root=root, split='train')
    val_dataset = MedGraphDataset(root=root, split='val')
    test_dataset = MedGraphDataset(root=root, split='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Quick test
    import sys
    
    root = sys.argv[1] if len(sys.argv) > 1 else 'data'
    
    print("Loading MedGraph Dataset...")
    
    try:
        dataset = MedGraphDataset(root=root, split='train')
        print(f"\n📊 Dataset Statistics:")
        stats = dataset.statistics()
        
        print(f"  Graphs: {stats['num_graphs']}")
        print(f"  Classes: {stats['num_classes']}")
        print(f"  Features: {stats['num_features']}")
        print(f"\n  Class Distribution:")
        for name, count in stats['class_distribution'].items():
            print(f"    {name}: {count}")
        print(f"\n  Nodes per graph: {stats['nodes']['mean']:.1f} ± {stats['nodes']['std']:.1f}")
        print(f"  Edges per graph: {stats['edges']['mean']:.1f} ± {stats['edges']['std']:.1f}")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nTo generate sample data for testing, run:")
        print("  python utils/generate_sample_data.py")
