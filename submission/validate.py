"""
Submission Validator for MedGraph Challenge

This script validates submission format before official evaluation.
It also enforces the ONE SUBMISSION ONLY policy.

Usage:
    python validate.py --submission your_submission.csv
    python validate.py --submission your_submission.csv --check-duplicate --username your_github_username
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np


# Expected number of test samples
EXPECTED_TEST_SAMPLES = 500

# Valid prediction values
VALID_PREDICTIONS = {0, 1, 2}

# Class names
CLASS_NAMES = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}


def load_test_ids(test_ids_file: Optional[str] = None) -> List[str]:
    """
    Load expected test graph IDs.
    
    Args:
        test_ids_file: Path to file with test IDs (one per line)
        
    Returns:
        List of expected graph IDs
    """
    if test_ids_file and os.path.exists(test_ids_file):
        with open(test_ids_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    # Generate expected IDs if file not provided
    return [f'test_{i:04d}' for i in range(1, EXPECTED_TEST_SAMPLES + 1)]


def check_previous_submissions(username: str, submissions_dir: str = 'submissions') -> bool:
    """
    Check if user has already submitted.
    
    Args:
        username: GitHub username
        submissions_dir: Path to submissions directory
        
    Returns:
        True if user has already submitted
    """
    user_submission_dir = Path(submissions_dir) / username
    
    if user_submission_dir.exists():
        submission_file = user_submission_dir / 'submission.csv'
        if submission_file.exists():
            return True
    
    # Also check leaderboard
    leaderboard_file = Path('leaderboard') / 'leaderboard.json'
    if leaderboard_file.exists():
        with open(leaderboard_file, 'r') as f:
            leaderboard = json.load(f)
            for entry in leaderboard.get('entries', []):
                if entry.get('username', '').lower() == username.lower():
                    return True
    
    return False


def validate_submission(
    submission_file: str,
    test_ids_file: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate a submission file.
    
    Args:
        submission_file: Path to submission CSV
        test_ids_file: Optional path to expected test IDs
        verbose: Print detailed validation messages
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    warnings = []
    
    # Check file exists
    if not os.path.exists(submission_file):
        errors.append(f"File not found: {submission_file}")
        return False, errors
    
    # Load submission
    try:
        df = pd.read_csv(submission_file)
    except Exception as e:
        errors.append(f"Could not read CSV file: {e}")
        return False, errors
    
    if verbose:
        print(f"\n{'='*50}")
        print("SUBMISSION VALIDATION")
        print(f"{'='*50}")
        print(f"File: {submission_file}")
        print(f"Rows: {len(df)}")
    
    # Check required columns
    required_columns = {'graph_id', 'prediction'}
    actual_columns = set(df.columns)
    
    if not required_columns.issubset(actual_columns):
        missing = required_columns - actual_columns
        errors.append(f"Missing required columns: {missing}")
        errors.append(f"Expected: {required_columns}")
        errors.append(f"Got: {actual_columns}")
    
    if errors:
        return False, errors
    
    # Check for missing values
    if df['graph_id'].isna().any():
        errors.append("Found missing values in 'graph_id' column")
    
    if df['prediction'].isna().any():
        errors.append("Found missing values in 'prediction' column")
    
    # Check prediction values
    invalid_preds = df[~df['prediction'].isin(VALID_PREDICTIONS)]
    if len(invalid_preds) > 0:
        errors.append(f"Invalid prediction values found. Must be 0, 1, or 2.")
        errors.append(f"Invalid entries:\n{invalid_preds.head(10)}")
    
    # Check for duplicates
    duplicates = df[df['graph_id'].duplicated()]
    if len(duplicates) > 0:
        errors.append(f"Found {len(duplicates)} duplicate graph_ids")
        errors.append(f"Duplicates: {duplicates['graph_id'].tolist()[:10]}")
    
    # Check against expected test IDs
    expected_ids = set(load_test_ids(test_ids_file))
    actual_ids = set(df['graph_id'])
    
    missing_ids = expected_ids - actual_ids
    extra_ids = actual_ids - expected_ids
    
    if missing_ids:
        errors.append(f"Missing predictions for {len(missing_ids)} graphs")
        if len(missing_ids) <= 10:
            errors.append(f"Missing: {sorted(missing_ids)}")
        else:
            errors.append(f"First 10 missing: {sorted(missing_ids)[:10]}")
    
    if extra_ids:
        warnings.append(f"Found {len(extra_ids)} extra graph_ids (will be ignored)")
    
    # Check number of predictions
    if len(df) != EXPECTED_TEST_SAMPLES:
        if len(df) < EXPECTED_TEST_SAMPLES:
            errors.append(f"Expected {EXPECTED_TEST_SAMPLES} predictions, got {len(df)}")
        else:
            warnings.append(f"Expected {EXPECTED_TEST_SAMPLES} predictions, got {len(df)}")
    
    # Print results
    if verbose:
        print(f"\n📊 Summary:")
        print(f"  Total predictions: {len(df)}")
        print(f"  Expected: {EXPECTED_TEST_SAMPLES}")
        
        # Class distribution
        class_dist = df['prediction'].value_counts().sort_index()
        print(f"\n📈 Prediction Distribution:")
        for cls, count in class_dist.items():
            pct = count / len(df) * 100
            print(f"  {CLASS_NAMES.get(cls, cls)}: {count} ({pct:.1f}%)")
        
        if warnings:
            print(f"\n⚠️  Warnings:")
            for w in warnings:
                print(f"  - {w}")
        
        if errors:
            print(f"\n❌ Errors:")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"\n✅ Validation PASSED")
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(
        description='Validate submission for MedGraph Challenge'
    )
    parser.add_argument(
        '--submission', '-s',
        type=str,
        required=True,
        help='Path to submission CSV file'
    )
    parser.add_argument(
        '--test-ids',
        type=str,
        default=None,
        help='Path to file with expected test graph IDs'
    )
    parser.add_argument(
        '--check-duplicate',
        action='store_true',
        help='Check if user has already submitted'
    )
    parser.add_argument(
        '--username',
        type=str,
        default=None,
        help='GitHub username (required if --check-duplicate)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only output errors'
    )
    
    args = parser.parse_args()
    
    # Check for duplicate submission
    if args.check_duplicate:
        if not args.username:
            print("❌ Error: --username required when using --check-duplicate")
            sys.exit(1)
        
        if check_previous_submissions(args.username):
            print(f"\n{'='*50}")
            print("❌ SUBMISSION REJECTED")
            print(f"{'='*50}")
            print(f"\nUser '{args.username}' has already submitted.")
            print("Each participant is allowed only ONE submission.")
            print("\nThis policy is strictly enforced.")
            sys.exit(1)
    
    # Validate submission
    is_valid, errors = validate_submission(
        args.submission,
        args.test_ids,
        verbose=not args.quiet
    )
    
    if not is_valid:
        print(f"\n{'='*50}")
        print("❌ VALIDATION FAILED")
        print(f"{'='*50}")
        print("\nPlease fix the errors above before submitting.")
        sys.exit(1)
    else:
        if not args.quiet:
            print(f"\n{'='*50}")
            print("✅ READY TO SUBMIT")
            print(f"{'='*50}")
            print("\nYour submission is valid!")
            print("\n⚠️  REMINDER: You can only submit ONCE.")
            print("Make sure this is your final submission.")
        sys.exit(0)


if __name__ == '__main__':
    main()
