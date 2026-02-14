"""
Official Evaluation Script for MedGraph Challenge

This script evaluates submissions against ground truth labels
and produces the official competition metrics.

Usage:
    python evaluate.py --predictions submission.csv --ground_truth labels.csv
    
    # For verbose output with confusion matrix
    python evaluate.py --predictions submission.csv --ground_truth labels.csv --verbose
    
    # Output to JSON
    python evaluate.py --predictions submission.csv --ground_truth labels.csv --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import compute_metrics, print_metrics, compute_metrics_with_confidence


def load_predictions(filepath: str) -> pd.DataFrame:
    """
    Load and validate prediction file.
    
    Expected format:
        graph_id,prediction
        test_0001,0
        test_0002,2
        ...
    
    Args:
        filepath (str): Path to predictions CSV.
        
    Returns:
        pd.DataFrame: Validated predictions.
        
    Raises:
        ValueError: If file format is invalid.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Could not read predictions file: {e}")
    
    # Validate columns
    required_cols = {'graph_id', 'prediction'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns. Expected: {required_cols}, Got: {set(df.columns)}")
    
    # Validate predictions are integers 0-2
    if not df['prediction'].isin([0, 1, 2]).all():
        invalid = df[~df['prediction'].isin([0, 1, 2])]
        raise ValueError(f"Invalid predictions found. Must be 0, 1, or 2. Invalid rows:\n{invalid}")
    
    # Check for duplicates
    if df['graph_id'].duplicated().any():
        duplicates = df[df['graph_id'].duplicated()]['graph_id'].unique()
        raise ValueError(f"Duplicate graph_ids found: {duplicates[:5]}...")
    
    return df


def load_ground_truth(filepath: str) -> pd.DataFrame:
    """
    Load ground truth labels.
    
    Args:
        filepath (str): Path to ground truth CSV.
        
    Returns:
        pd.DataFrame: Ground truth labels.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Could not read ground truth file: {e}")
    
    required_cols = {'graph_id', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns. Expected: {required_cols}, Got: {set(df.columns)}")
    
    return df


def evaluate_submission(
    predictions_file: str,
    ground_truth_file: str,
    verbose: bool = False,
    compute_ci: bool = False,
) -> dict:
    """
    Evaluate a submission against ground truth.
    
    Args:
        predictions_file (str): Path to predictions CSV.
        ground_truth_file (str): Path to ground truth CSV.
        verbose (bool): Print detailed results.
        compute_ci (bool): Compute confidence intervals.
        
    Returns:
        dict: Evaluation metrics.
    """
    # Load files
    predictions = load_predictions(predictions_file)
    ground_truth = load_ground_truth(ground_truth_file)
    
    # Merge on graph_id
    merged = pd.merge(ground_truth, predictions, on='graph_id', how='left')
    
    # Check for missing predictions
    missing = merged['prediction'].isna().sum()
    if missing > 0:
        missing_ids = merged[merged['prediction'].isna()]['graph_id'].tolist()
        raise ValueError(f"Missing predictions for {missing} graphs: {missing_ids[:5]}...")
    
    # Check for extra predictions
    extra = len(predictions) - len(ground_truth)
    if extra > 0:
        print(f"Warning: {extra} extra predictions found (will be ignored)")
    
    # Convert to numpy
    y_true = merged['label'].values
    y_pred = merged['prediction'].values.astype(int)
    
    # Compute metrics
    if compute_ci:
        metrics = compute_metrics_with_confidence(y_true, y_pred)
    else:
        metrics = compute_metrics(y_true, y_pred)
    
    # Add submission info
    metrics['num_samples'] = len(y_true)
    
    if verbose:
        print_metrics(metrics)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Official evaluation script for MedGraph Challenge'
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=True,
        help='Path to predictions CSV file'
    )
    parser.add_argument(
        '--ground_truth', '-g',
        type=str,
        required=True,
        help='Path to ground truth CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save results JSON (optional)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed results'
    )
    parser.add_argument(
        '--ci',
        action='store_true',
        help='Compute 95% confidence intervals via bootstrap'
    )
    
    args = parser.parse_args()
    
    try:
        metrics = evaluate_submission(
            args.predictions,
            args.ground_truth,
            verbose=args.verbose,
            compute_ci=args.ci,
        )
        
        # Print summary
        print(f"\n{'='*40}")
        print("EVALUATION SUMMARY")
        print(f"{'='*40}")
        print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
        print(f"Accuracy:       {metrics['accuracy']:.4f}")
        print(f"{'='*40}\n")
        
        # Save to JSON if requested
        if args.output:
            # Convert numpy types to Python types for JSON
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj
            
            metrics_json = json.loads(
                json.dumps(metrics, default=convert_numpy)
            )
            
            with open(args.output, 'w') as f:
                json.dump(metrics_json, f, indent=2)
            print(f"Results saved to: {args.output}")
        
        return metrics
        
    except ValueError as e:
        print(f"❌ Validation Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
