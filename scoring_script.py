import pandas as pd
from sklearn.metrics import f1_score
import sys

# Usage: python scoring_script.py submissions/file.csv

submission_file = sys.argv[1]

# Load submission
submission = pd.read_csv(submission_file)

# Load ground truth (hidden)
truth = pd.read_csv('data/test_labels.csv')

# Compute F1 score
score = f1_score(truth['target'], submission['target'], average='macro')
print(f'Submission F1 Score: {score:.4f}')
