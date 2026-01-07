import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Load data
train = pd.read_csv('../data/train.csv')

# For graph data, aggregate node features per graph
feature_cols = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'degree']
graph_features = train.groupby('id')[feature_cols].mean().reset_index()
graph_labels = train.groupby('id')['target'].first().reset_index()
train_agg = graph_features.merge(graph_labels, on='id')

X = train_agg.drop(['id', 'target'], axis=1)
y = train_agg['target']

# Split into train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a baseline model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_val)
score = f1_score(y_val, y_pred, average='macro')
print(f'Validation F1 Score: {score:.4f}')

# Make predictions on test set
test = pd.read_csv('../data/test.csv')
test_agg = test.groupby('id')[feature_cols].mean().reset_index()
test_preds = clf.predict(test_agg.drop('id', axis=1))
pd.DataFrame({'id': test_agg['id'], 'target': test_preds}).to_csv('../submissions/submission.csv', index=False)
print('Submission saved to ../submissions/submission.csv')
