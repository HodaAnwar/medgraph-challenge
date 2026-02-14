"""
Training Script for MedGraph Challenge Baselines

This script provides a complete training pipeline with:
- Multiple model architectures (GCN, GAT, GraphSAGE)
- Learning rate scheduling
- Early stopping
- Experiment tracking (optional W&B)
- Model checkpointing
- Comprehensive logging

Usage:
    python train.py --model gcn --epochs 100 --lr 0.001
    python train.py --model gat --epochs 100 --lr 0.0005 --heads 4
    python train.py --model sage --epochs 100 --lr 0.001
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset import MedGraphDataset
from evaluation.metrics import compute_metrics, print_metrics
from baselines.gcn_baseline import GCNClassifier, GCNWithJK
from baselines.gat_baseline import GATClassifier, HierarchicalGAT
from baselines.sage_baseline import GraphSAGEClassifier, DeepGraphSAGE


def get_model(args, num_features: int, num_classes: int):
    """Initialize model based on arguments."""
    
    model_dict = {
        'gcn': GCNClassifier,
        'gcn_jk': GCNWithJK,
        'gat': GATClassifier,
        'gat_hier': HierarchicalGAT,
        'sage': GraphSAGEClassifier,
        'sage_deep': DeepGraphSAGE,
    }
    
    if args.model not in model_dict:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(model_dict.keys())}")
    
    model_class = model_dict[args.model]
    
    # Common arguments
    kwargs = {
        'in_channels': num_features,
        'hidden_channels': args.hidden_channels,
        'num_classes': num_classes,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
    }
    
    # Model-specific arguments
    if 'gat' in args.model:
        kwargs['heads'] = args.heads
    
    return model_class(**kwargs)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=-1)
        total_correct += (pred == batch.y).sum().item()
        total_samples += batch.num_graphs
    
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        
        total_loss += loss.item() * batch.num_graphs
        
        pred = out.argmax(dim=-1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = avg_loss
    
    return metrics


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"MedGraph Challenge - Training {args.model.upper()}")
    print(f"{'='*60}")
    
    # Load datasets
    print("\n📊 Loading datasets...")
    train_dataset = MedGraphDataset(root=args.data_dir, split='train')
    val_dataset = MedGraphDataset(root=args.data_dir, split='val')
    
    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val: {len(val_dataset)} graphs")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Initialize model
    print("\n🔧 Initializing model...")
    model = get_model(args, train_dataset.num_node_features, train_dataset.num_classes)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.model}")
    print(f"  Parameters: {num_params:,}")
    
    # Loss function with class weights
    if args.use_class_weights:
        class_weights = train_dataset.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    else:
        scheduler = None
    
    # Training loop
    print(f"\n🚀 Training for {args.epochs} epochs...")
    print(f"{'='*60}")
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['macro_f1'])
            else:
                scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Logging
        history['train'].append({'loss': train_loss, 'acc': train_acc})
        history['val'].append(val_metrics)
        
        # Print progress
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['macro_f1']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, output_dir / 'best_model.pt')
            
            print(f"  ✓ New best model saved! F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
            break
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("📊 Final Results")
    print(f"{'='*60}")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_metrics = evaluate(model, val_loader, criterion, device)
    
    print(f"\nBest Epoch: {best_epoch}")
    print_metrics(val_metrics)
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete! Results saved to: {output_dir}")
    
    return val_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MedGraph Challenge baseline models')
    
    # Model
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'gcn_jk', 'gat', 'gat_hier', 'sage', 'sage_deep'],
                        help='Model architecture')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads (for GAT)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    main(args)
