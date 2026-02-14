"""
Evaluation Metrics for MedGraph Challenge

This module provides comprehensive evaluation metrics for histopathology
cell-graph classification, including:
- Macro/Micro F1-Score
- Per-class precision, recall, F1
- Confusion matrix
- Balanced accuracy
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# Class names for the challenge
CLASS_NAMES = ['Normal', 'Benign', 'Malignant']


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        class_names (List[str], optional): Names for each class.
        
    Returns:
        Dict: Dictionary containing all metrics.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # F1 scores
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {}
    for i, name in enumerate(class_names):
        metrics['per_class'][name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def print_metrics(metrics: Dict, class_names: Optional[List[str]] = None):
    """
    Print metrics in a formatted table.
    
    Args:
        metrics (Dict): Metrics dictionary from compute_metrics.
        class_names (List[str], optional): Names for each class.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f} (Primary Metric)")
    print(f"  Micro F1:          {metrics['micro_f1']:.4f}")
    print(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")
    
    print(f"\n📋 Per-Class Metrics:")
    print("-"*55)
    print(f"{'Class':<12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-"*55)
    
    for name in class_names:
        if name in metrics['per_class']:
            cls_metrics = metrics['per_class'][name]
            print(f"{name:<12} {cls_metrics['precision']:>12.4f} "
                  f"{cls_metrics['recall']:>12.4f} {cls_metrics['f1']:>12.4f}")
    
    print("-"*55)
    
    print(f"\n🔢 Confusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    
    # Print header
    header = "Pred →   " + "  ".join(f"{name[:6]:>8}" for name in class_names)
    print(header)
    print("-" * len(header))
    
    # Print rows
    for i, name in enumerate(class_names):
        row = f"{name[:8]:<8} " + "  ".join(f"{cm[i,j]:>8d}" for j in range(len(class_names)))
        print(row)
    
    print("="*60 + "\n")


def compute_metrics_with_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict:
    """
    Compute metrics with confidence intervals using bootstrap.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_proba (np.ndarray, optional): Prediction probabilities.
        n_bootstrap (int): Number of bootstrap iterations.
        confidence (float): Confidence level.
        
    Returns:
        Dict: Metrics with confidence intervals.
    """
    n_samples = len(y_true)
    alpha = (1 - confidence) / 2
    
    metrics = compute_metrics(y_true, y_pred)
    
    # Bootstrap for confidence intervals
    bootstrap_metrics = {
        'accuracy': [],
        'macro_f1': [],
        'balanced_accuracy': [],
    }
    
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        bootstrap_metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
        bootstrap_metrics['macro_f1'].append(f1_score(y_true_boot, y_pred_boot, average='macro'))
        bootstrap_metrics['balanced_accuracy'].append(balanced_accuracy_score(y_true_boot, y_pred_boot))
    
    # Compute confidence intervals
    for metric_name, values in bootstrap_metrics.items():
        values = np.array(values)
        lower = np.percentile(values, alpha * 100)
        upper = np.percentile(values, (1 - alpha) * 100)
        metrics[f'{metric_name}_ci'] = (float(lower), float(upper))
    
    return metrics


def format_leaderboard_entry(
    team_name: str,
    model_name: str,
    metrics: Dict,
    submission_date: str,
) -> Dict:
    """
    Format metrics as a leaderboard entry.
    
    Args:
        team_name (str): Name of the team.
        model_name (str): Name of the model.
        metrics (Dict): Metrics dictionary.
        submission_date (str): Date of submission (ISO format).
        
    Returns:
        Dict: Formatted leaderboard entry.
    """
    return {
        'team': team_name,
        'model': model_name,
        'macro_f1': round(metrics['macro_f1'], 4),
        'accuracy': round(metrics['accuracy'], 4),
        'balanced_accuracy': round(metrics['balanced_accuracy'], 4),
        'per_class_f1': {
            name: round(metrics['per_class'][name]['f1'], 4)
            for name in CLASS_NAMES
        },
        'submission_date': submission_date,
    }


if __name__ == '__main__':
    # Test with random data
    np.random.seed(42)
    
    n_samples = 1000
    y_true = np.random.randint(0, 3, n_samples)
    
    # Simulate predictions with some noise
    y_pred = y_true.copy()
    noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    y_pred[noise_indices] = np.random.randint(0, 3, len(noise_indices))
    
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)
    
    # Test with confidence intervals
    metrics_ci = compute_metrics_with_confidence(y_true, y_pred)
    print(f"\nMacro F1 with 95% CI: {metrics_ci['macro_f1']:.4f} "
          f"({metrics_ci['macro_f1_ci'][0]:.4f}, {metrics_ci['macro_f1_ci'][1]:.4f})")
