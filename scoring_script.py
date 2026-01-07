"""
MedGraphDiversity Challenge - Official Scoring Script
======================================================

Computes Macro F1-Score for submitted predictions.

Usage:
    python scoring_script.py <submission_file>
    
Example:
    python scoring_script.py submissions/my_submission.csv
"""

import sys
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np
from pathlib import Path


def score_submission(submission_path, ground_truth_path='data/test_labels.csv'):
    """
    Score a submission against ground truth.
    
    Parameters:
    -----------
    submission_path : str
        Path to submission CSV with columns: graph_id, label
    ground_truth_path : str
        Path to ground truth labels
        
    Returns:
    --------
    macro_f1 : float
        Official Macro F1 score
    """
    
    # Load files
    submission = pd.read_csv(submission_path)
    ground_truth = pd.read_csv(ground_truth_path)
    
    # Validate submission format
    required_cols = {'graph_id', 'label'}
    if not required_cols.issubset(submission.columns):
        raise ValueError(f"Submission must have columns: {required_cols}")
    
    # Check length
    if len(submission) != len(ground_truth):
        raise ValueError(f"Expected {len(ground_truth)} predictions, got {len(submission)}")
    
    # Merge on graph_id
    merged = submission.merge(ground_truth, on='graph_id', suffixes=('_pred', '_true'))
    
    y_pred = merged['label_pred'].values
    y_true = merged['label_true'].values
    
    # Validate labels
    valid_labels = {0, 1, 2}
    if not set(y_pred).issubset(valid_labels):
        raise ValueError("Labels must be 0, 1, or 2")
    
    # Compute Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    return macro_f1, y_true, y_pred


def main():
    if len(sys.argv) < 2:
        print("Usage: python scoring_script.py <submission_file>")
        print("Example: python scoring_script.py submissions/my_submission.csv")
        sys.exit(1)
    
    submission_path = sys.argv[1]
    
    if not Path(submission_path).exists():
        print(f"Error: File not found: {submission_path}")
        sys.exit(1)
    
    try:
        macro_f1, y_true, y_pred = score_submission(submission_path)
        
        print("=" * 50)
        print("MedGraphDiversity Challenge - Score")
        print("=" * 50)
        print(f"\n  MACRO F1 SCORE: {macro_f1:.4f}\n")
        
        print("Classification Report:")
        print("-" * 50)
        print(classification_report(
            y_true, y_pred,
            target_names=['Normal (0)', 'Benign (1)', 'Malignant (2)'],
            digits=4
        ))
        
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
