"""
Leaderboard Update Script for MedGraph Challenge

This script updates the leaderboard with new submissions.
It implements Kaggle-style ranking where tied scores share the same rank.

Usage:
    python update_leaderboard.py --username john_doe --submission submission.csv --ground-truth labels.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import compute_metrics


def kaggle_rank(scores: List[float]) -> List[int]:
    """
    Compute Kaggle-style rankings where ties share ranks.
    
    Example:
        scores = [0.95, 0.90, 0.90, 0.85]
        ranks  = [1,    2,    2,    4]
        
    Args:
        scores: List of scores (higher is better)
        
    Returns:
        List of ranks (1-indexed)
    """
    if not scores:
        return []
    
    # Sort scores in descending order with original indices
    indexed_scores = [(score, i) for i, score in enumerate(scores)]
    indexed_scores.sort(key=lambda x: -x[0])  # Sort descending
    
    ranks = [0] * len(scores)
    current_rank = 1
    
    for i, (score, original_idx) in enumerate(indexed_scores):
        if i > 0 and score < indexed_scores[i-1][0]:
            # Score is different from previous, rank is position + 1
            current_rank = i + 1
        ranks[original_idx] = current_rank
    
    return ranks


def load_leaderboard(filepath: str = 'leaderboard/leaderboard.json') -> Dict:
    """Load leaderboard from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {
        'last_updated': None,
        'total_submissions': 0,
        'primary_metric': 'macro_f1',
        'entries': []
    }


def save_leaderboard(leaderboard: Dict, filepath: str = 'leaderboard/leaderboard.json'):
    """Save leaderboard to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(leaderboard, f, indent=2)


def check_duplicate_submission(leaderboard: Dict, username: str) -> bool:
    """Check if user has already submitted."""
    for entry in leaderboard['entries']:
        if entry['username'].lower() == username.lower():
            return True
    return False


def add_submission(
    leaderboard: Dict,
    username: str,
    model_name: str,
    metrics: Dict,
) -> Dict:
    """
    Add a new submission to the leaderboard.
    
    Args:
        leaderboard: Current leaderboard dictionary
        username: GitHub username
        model_name: Name/description of the model
        metrics: Evaluation metrics dictionary
        
    Returns:
        Updated leaderboard
    """
    # Create new entry
    new_entry = {
        'username': username,
        'model': model_name,
        'macro_f1': round(metrics['macro_f1'], 4),
        'accuracy': round(metrics['accuracy'], 4),
        'balanced_accuracy': round(metrics['balanced_accuracy'], 4),
        'per_class_f1': {
            'Normal': round(metrics['per_class']['Normal']['f1'], 4),
            'Benign': round(metrics['per_class']['Benign']['f1'], 4),
            'Malignant': round(metrics['per_class']['Malignant']['f1'], 4),
        },
        'submission_date': datetime.utcnow().isoformat() + 'Z',
    }
    
    # Add to entries
    leaderboard['entries'].append(new_entry)
    leaderboard['total_submissions'] += 1
    leaderboard['last_updated'] = datetime.utcnow().isoformat() + 'Z'
    
    # Recompute ranks
    leaderboard = update_ranks(leaderboard)
    
    return leaderboard


def update_ranks(leaderboard: Dict) -> Dict:
    """
    Update ranks for all entries using Kaggle-style ranking.
    
    Args:
        leaderboard: Leaderboard dictionary
        
    Returns:
        Updated leaderboard with ranks
    """
    if not leaderboard['entries']:
        return leaderboard
    
    # Get primary metric scores
    primary_metric = leaderboard.get('primary_metric', 'macro_f1')
    scores = [entry[primary_metric] for entry in leaderboard['entries']]
    
    # Compute ranks
    ranks = kaggle_rank(scores)
    
    # Assign ranks to entries
    for entry, rank in zip(leaderboard['entries'], ranks):
        entry['rank'] = rank
    
    # Sort entries by rank
    leaderboard['entries'].sort(key=lambda x: (x['rank'], -x[primary_metric]))
    
    return leaderboard


def format_leaderboard_markdown(leaderboard: Dict) -> str:
    """
    Format leaderboard as Markdown table.
    
    Returns:
        Markdown string
    """
    lines = [
        "# 🏆 MedGraph Challenge Leaderboard",
        "",
        f"*Last updated: {leaderboard.get('last_updated', 'N/A')}*",
        f"*Total submissions: {leaderboard.get('total_submissions', 0)}*",
        "",
        "| Rank | Team | Model | Macro F1 | Accuracy | Normal F1 | Benign F1 | Malignant F1 | Date |",
        "|------|------|-------|----------|----------|-----------|-----------|--------------|------|",
    ]
    
    for entry in leaderboard['entries']:
        rank = entry.get('rank', '-')
        # Add medal emoji for top 3
        if rank == 1:
            rank_str = "🥇 1"
        elif rank == 2:
            rank_str = "🥈 2"
        elif rank == 3:
            rank_str = "🥉 3"
        else:
            rank_str = str(rank)
        
        per_class = entry.get('per_class_f1', {})
        date = entry.get('submission_date', '')[:10]  # Just the date part
        
        line = f"| {rank_str} | {entry['username']} | {entry['model']} | " \
               f"{entry['macro_f1']:.4f} | {entry['accuracy']:.4f} | " \
               f"{per_class.get('Normal', '-')} | {per_class.get('Benign', '-')} | " \
               f"{per_class.get('Malignant', '-')} | {date} |"
        lines.append(line)
    
    if not leaderboard['entries']:
        lines.append("| - | *No submissions yet* | - | - | - | - | - | - | - |")
    
    lines.extend([
        "",
        "---",
        "",
        "**Ranking Policy**: Tied scores share ranks (Kaggle-style). "
        "If two teams tie for 1st place, both receive rank 1, and the next team receives rank 3.",
        "",
        "**Primary Metric**: Macro F1-Score",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Update MedGraph Challenge leaderboard')
    parser.add_argument('--username', type=str, required=True, help='GitHub username')
    parser.add_argument('--model', type=str, default='Unknown Model', help='Model description')
    parser.add_argument('--submission', type=str, required=True, help='Path to submission CSV')
    parser.add_argument('--ground-truth', type=str, required=True, help='Path to ground truth CSV')
    parser.add_argument('--leaderboard', type=str, default='leaderboard/leaderboard.json',
                        help='Path to leaderboard JSON')
    parser.add_argument('--output-md', type=str, default='leaderboard/README.md',
                        help='Path to output Markdown leaderboard')
    
    args = parser.parse_args()
    
    # Load current leaderboard
    leaderboard = load_leaderboard(args.leaderboard)
    
    # Check for duplicate submission
    if check_duplicate_submission(leaderboard, args.username):
        print(f"❌ REJECTED: User '{args.username}' has already submitted.")
        print("Each participant is allowed only ONE submission.")
        sys.exit(1)
    
    # Load and evaluate submission
    try:
        submission = pd.read_csv(args.submission)
        ground_truth = pd.read_csv(args.ground_truth)
        
        # Merge and evaluate
        merged = pd.merge(ground_truth, submission, on='graph_id')
        y_true = merged['label'].values
        y_pred = merged['prediction'].values.astype(int)
        
        metrics = compute_metrics(y_true, y_pred)
        
    except Exception as e:
        print(f"❌ Error evaluating submission: {e}")
        sys.exit(1)
    
    # Add to leaderboard
    leaderboard = add_submission(leaderboard, args.username, args.model, metrics)
    
    # Save updated leaderboard
    save_leaderboard(leaderboard, args.leaderboard)
    
    # Generate Markdown
    md_content = format_leaderboard_markdown(leaderboard)
    with open(args.output_md, 'w') as f:
        f.write(md_content)
    
    # Print results
    print(f"\n{'='*50}")
    print("✅ SUBMISSION ACCEPTED")
    print(f"{'='*50}")
    print(f"\nUser: {args.username}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Find rank
    for entry in leaderboard['entries']:
        if entry['username'] == args.username:
            print(f"Rank: {entry['rank']}")
            break
    
    print(f"\nLeaderboard updated: {args.leaderboard}")


if __name__ == '__main__':
    main()
