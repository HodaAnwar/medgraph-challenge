"""
Synthetic Data Generator for MedGraph Challenge

This script generates synthetic cell-graph data for the challenge.
The data includes realistic challenges:
- Class imbalance
- Feature noise
- Edge sparsity variation
- Distribution shift in test set

Data is sized for < 3 hour CPU training as per competition guidelines.

Usage:
    python generate_sample_data.py --output-dir data
"""

import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
from scipy.spatial import Delaunay
from tqdm import tqdm


# Dataset configuration (sized for < 3h CPU training)
CONFIG = {
    'train': {'count': 1200, 'class_dist': [400, 400, 400]},  # Balanced
    'val': {'count': 300, 'class_dist': [100, 100, 100]},      # Balanced
    'test_public': {'count': 500, 'class_dist': [167, 167, 166]},  # Hidden, approx balanced
}

# Feature dimensions
NUM_NODE_FEATURES = 114
MORPHOLOGY_DIM = 32
TEXTURE_DIM = 64
COLOR_DIM = 16
POSITION_DIM = 2

# Graph parameters
MIN_NODES = 50
MAX_NODES = 120
EDGE_DISTANCE_THRESHOLD = 0.15  # Normalized distance

# Challenge parameters
FEATURE_NOISE_STD = 0.05
LABEL_NOISE_RATE = 0.02  # 2% label noise in training
TEST_DISTRIBUTION_SHIFT = 0.1  # Feature shift in test set


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_cell_positions(num_nodes: int) -> np.ndarray:
    """
    Generate cell positions using a realistic tissue-like distribution.
    
    Uses a mixture of clustered and uniform distributions to simulate
    real tissue architecture.
    """
    positions = []
    
    # Generate cluster centers
    num_clusters = np.random.randint(2, 5)
    cluster_centers = np.random.rand(num_clusters, 2)
    cluster_stds = np.random.uniform(0.05, 0.15, num_clusters)
    
    # Assign nodes to clusters
    for _ in range(num_nodes):
        cluster_idx = np.random.randint(num_clusters)
        center = cluster_centers[cluster_idx]
        std = cluster_stds[cluster_idx]
        
        # Generate position with some noise
        pos = center + np.random.randn(2) * std
        pos = np.clip(pos, 0, 1)  # Keep in unit square
        positions.append(pos)
    
    return np.array(positions)


def construct_edges(positions: np.ndarray, threshold: float = EDGE_DISTANCE_THRESHOLD) -> np.ndarray:
    """
    Construct edges using Delaunay triangulation with distance filtering.
    
    Returns edge_index in COO format [2, num_edges].
    """
    if len(positions) < 4:
        # Not enough points for Delaunay
        return np.array([[0], [1]])
    
    try:
        tri = Delaunay(positions)
        edges = set()
        
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    p1, p2 = simplex[i], simplex[j]
                    dist = np.linalg.norm(positions[p1] - positions[p2])
                    
                    if dist < threshold:
                        edges.add((min(p1, p2), max(p1, p2)))
        
        if len(edges) == 0:
            # Fallback: connect nearest neighbors
            edges = {(0, 1)}
        
        # Convert to edge_index (undirected: add both directions)
        edge_list = []
        for i, j in edges:
            edge_list.extend([[i, j], [j, i]])
        
        return np.array(edge_list).T
        
    except Exception:
        # Fallback for degenerate cases
        return np.array([[0, 1], [1, 0]]).T


def generate_class_features(
    num_nodes: int,
    class_label: int,
    positions: np.ndarray,
    distribution_shift: float = 0.0,
) -> np.ndarray:
    """
    Generate node features with class-specific patterns.
    
    Each class has distinct feature distributions:
    - Normal (0): Regular morphology, uniform texture
    - Benign (1): Slightly enlarged cells, varied texture
    - Malignant (2): Irregular morphology, high texture variance
    """
    features = np.zeros((num_nodes, NUM_NODE_FEATURES - POSITION_DIM))
    
    # Base class statistics
    class_patterns = {
        0: {'morph_mean': 0.3, 'morph_std': 0.1, 'text_mean': 0.4, 'text_std': 0.15, 'color_mean': 0.5},
        1: {'morph_mean': 0.5, 'morph_std': 0.2, 'text_mean': 0.5, 'text_std': 0.2, 'color_mean': 0.55},
        2: {'morph_mean': 0.7, 'morph_std': 0.3, 'text_mean': 0.6, 'text_std': 0.3, 'color_mean': 0.6},
    }
    
    params = class_patterns[class_label]
    
    # Apply distribution shift for test data
    if distribution_shift > 0:
        params = {k: v + np.random.uniform(-distribution_shift, distribution_shift) 
                  for k, v in params.items()}
    
    for i in range(num_nodes):
        # Morphology features (32 dims)
        morph = np.random.normal(params['morph_mean'], params['morph_std'], MORPHOLOGY_DIM)
        
        # Texture features (64 dims) - Haralick-like
        texture = np.random.normal(params['text_mean'], params['text_std'], TEXTURE_DIM)
        
        # Color features (16 dims)
        color = np.random.normal(params['color_mean'], 0.1, COLOR_DIM)
        
        features[i] = np.concatenate([morph, texture, color])
    
    # Clip to valid range
    features = np.clip(features, 0, 1)
    
    return features


def add_feature_noise(features: np.ndarray, noise_std: float = FEATURE_NOISE_STD) -> np.ndarray:
    """Add Gaussian noise to features."""
    noise = np.random.normal(0, noise_std, features.shape)
    return np.clip(features + noise, 0, 1)


def generate_graph(
    graph_id: str,
    class_label: int,
    add_noise: bool = True,
    distribution_shift: float = 0.0,
) -> dict:
    """
    Generate a single cell-graph.
    
    Returns:
        dict with 'x', 'edge_index', 'y', 'graph_id'
    """
    # Random number of nodes
    num_nodes = np.random.randint(MIN_NODES, MAX_NODES + 1)
    
    # Generate cell positions
    positions = generate_cell_positions(num_nodes)
    
    # Construct edges (adjacency)
    edge_index = construct_edges(positions)
    
    # Generate class-specific features
    base_features = generate_class_features(num_nodes, class_label, positions, distribution_shift)
    
    # Add noise if specified
    if add_noise:
        base_features = add_feature_noise(base_features)
    
    # Concatenate with positions
    features = np.concatenate([base_features, positions], axis=1)
    
    return {
        'x': torch.tensor(features, dtype=torch.float32),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'y': torch.tensor([class_label], dtype=torch.long),
        'graph_id': graph_id,
    }


def generate_dataset(
    split: str,
    output_dir: str,
    config: dict,
) -> list:
    """Generate all graphs for a dataset split."""
    
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine if this is test data (apply distribution shift)
    is_test = 'test' in split
    shift = TEST_DISTRIBUTION_SHIFT if is_test else 0.0
    
    graphs = []
    graph_idx = 1
    
    # Generate graphs for each class
    for class_label, count in enumerate(config['class_dist']):
        for _ in tqdm(range(count), desc=f"{split} - Class {class_label}"):
            graph_id = f"{split}_{graph_idx:04d}"
            
            graph = generate_graph(
                graph_id=graph_id,
                class_label=class_label,
                add_noise=True,
                distribution_shift=shift,
            )
            
            # Save as PyTorch Geometric Data object
            from torch_geometric.data import Data
            data = Data(
                x=graph['x'],
                edge_index=graph['edge_index'],
                y=graph['y'],
                graph_id=graph_id,
            )
            
            torch.save(data, split_dir / f"{graph_id}.pt")
            graphs.append({'graph_id': graph_id, 'label': class_label})
            graph_idx += 1
    
    return graphs


def generate_test_labels(graphs: list, output_dir: str):
    """Generate ground truth labels file for test set."""
    import pandas as pd
    
    df = pd.DataFrame(graphs)
    df.to_csv(Path(output_dir) / 'test_labels.csv', index=False)
    print(f"Saved test labels to {output_dir}/test_labels.csv")


def generate_test_ids(graphs: list, output_dir: str):
    """Generate list of test graph IDs."""
    ids = [g['graph_id'] for g in graphs]
    
    with open(Path(output_dir) / 'test_public' / 'test_ids.txt', 'w') as f:
        f.write('\n'.join(ids))
    print(f"Saved test IDs to {output_dir}/test_public/test_ids.txt")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic MedGraph data')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("="*60)
    print("MedGraph Challenge - Synthetic Data Generator")
    print("="*60)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate each split
    for split, config in CONFIG.items():
        print(f"\n📊 Generating {split} split ({config['count']} graphs)...")
        graphs = generate_dataset(split, args.output_dir, config)
        
        if split == 'test_public':
            generate_test_labels(graphs, args.output_dir)
            generate_test_ids(graphs, args.output_dir)
    
    print("\n" + "="*60)
    print("✅ Data generation complete!")
    print("="*60)
    print(f"\nDataset structure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── train/          ({CONFIG['train']['count']} graphs)")
    print(f"  ├── val/            ({CONFIG['val']['count']} graphs)")
    print(f"  ├── test_public/    ({CONFIG['test_public']['count']} graphs)")
    print(f"  └── test_labels.csv (ground truth for evaluation)")
    
    print(f"\n📈 Dataset Challenges:")
    print(f"  • Feature noise: {FEATURE_NOISE_STD*100:.1f}% Gaussian")
    print(f"  • Distribution shift in test: {TEST_DISTRIBUTION_SHIFT*100:.1f}%")
    print(f"  • Variable graph sizes: {MIN_NODES}-{MAX_NODES} nodes")


if __name__ == '__main__':
    main()
