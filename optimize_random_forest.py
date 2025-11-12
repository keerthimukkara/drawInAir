"""
Optimize RandomForest hyperparameters and optionally save best model.

Usage:
    python optimize_random_forest.py --dataset gesture_dataset --out models/optimized_rf.pkl --cv 3

This script will:
- Load dataset from `gesture_dataset` using `eval_recognizer.load_dataset`.
- Extract features and run GridSearchCV over n_estimators and max_depth (and optionally n_jobs).
- Print best parameters and cross-validated score.
- Save the best estimator to the provided output path.
"""
import os
import argparse
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from eval_recognizer import load_dataset, preprocess_sequence, extract_features_from_seq

'''Output:
**After running this script**
Best params: {'rf__max_depth': None, 'rf__max_features': None, 'rf__n_estimators': 100}
Best CV score: 0.97695746144022
Saved optimized model to models/optimized_rf.pkl

'''

def load_features(dataset_dir):
    X_seqs, y, labels = load_dataset(dataset_dir)
    if len(X_seqs) == 0:
        raise SystemExit('No data found in ' + dataset_dir)

    X_feats = []
    for s in X_seqs:
        s_pre = preprocess_sequence(s)
        feat = extract_features_from_seq(s_pre)
        X_feats.append(feat)

    X_feats = np.array(X_feats)
    return X_feats, y, labels


def main(dataset_dir, out_model, cv_splits):
    X, y, labels = load_features(dataset_dir)
    print('Loaded features:', X.shape, 'labels:', len(labels))

    scaler = StandardScaler()
    rf = RandomForestClassifier(random_state=42)
    pipe = Pipeline([('scaler', scaler), ('rf', rf)])

    param_grid = {
        'rf__n_estimators': [50, 100, 200, 400],
        'rf__max_depth': [None, 10, 20, 40],
        'rf__max_features': ['sqrt', 'log2', None]
    }

    cv = StratifiedKFold(n_splits=max(2, min(cv_splits, len(y)//2)), shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring='accuracy', verbose=2)
    gs.fit(X, y)

    print('Best params:', gs.best_params_)
    print('Best CV score:', gs.best_score_)

    if out_model:
        os.makedirs(os.path.dirname(out_model), exist_ok=True)
        with open(out_model, 'wb') as f:
            pickle.dump({'model': gs.best_estimator_, 'labels': labels}, f)
        print('Saved optimized model to', out_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='gesture_dataset')
    parser.add_argument('--out', default='models/optimized_rf.pkl')
    parser.add_argument('--cv', type=int, default=3)
    args = parser.parse_args()
    main(args.dataset, args.out, args.cv)
'''Output:
**After running this script**
Best params: {'rf__max_depth': None, 'rf__max_features': None, 'rf__n_estimators': 100}
Best CV score: 0.97695746144022
Saved optimized model to models/optimized_rf.pkl

'''