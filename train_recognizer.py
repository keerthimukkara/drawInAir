"""
Train a RandomForest gesture recognizer from JSON sessions in `gesture_dataset/`.
Saves trained recognizer to `models/random_forest_recognizer.pkl`.
"""
import os
import json
import numpy as np
import pickle
from gesture_recognition import RandomForestGestureRecognizer


def load_dataset(dataset_dir='gesture_dataset'):
    gestures = {}
    for fname in os.listdir(dataset_dir):
        if not fname.lower().endswith('.json'):
            continue
        path = os.path.join(dataset_dir, fname)
        with open(path, 'r') as f:
            j = json.load(f)
        for entry in j.get('data', []):
            gtype = entry.get('gesture_type') or entry.get('label') or 'unknown'
            lm = entry.get('landmarks')
            if not lm:
                continue
            try:
                arr = np.array(lm).reshape(-1, 3)
                pts = arr[:, :2]
            except Exception:
                continue
            if len(pts) < 4:
                continue
            gestures.setdefault(gtype, []).append(pts)
    return gestures


def preprocess_sequence(pts, n=128):
    pts = np.array(pts)
    old_idx = np.linspace(0, len(pts) - 1, len(pts))
    new_idx = np.linspace(0, len(pts) - 1, n)
    resampled = np.zeros((n, pts.shape[1]))
    for dim in range(pts.shape[1]):
        resampled[:, dim] = np.interp(new_idx, old_idx, pts[:, dim])
    centroid = np.mean(resampled, axis=0)
    resampled = resampled - centroid
    max_dist = np.max(np.linalg.norm(resampled, axis=1))
    if max_dist > 0:
        resampled = resampled / max_dist
    return resampled


if __name__ == '__main__':
    dataset_dir = 'gesture_dataset'
    out_dir = 'models'
    os.makedirs(out_dir, exist_ok=True)

    gestures = load_dataset(dataset_dir)
    print('Found gestures:', {k: len(v) for k, v in gestures.items()})

    X = []
    y = []
    class_names = []
    for i, (gtype, seqs) in enumerate(gestures.items()):
        class_names.append(gtype)
        for s in seqs:
            X.append(preprocess_sequence(s, n=128))
            y.append(i)

    if not X:
        print('No training data found in', dataset_dir)
        raise SystemExit(1)

    rf = RandomForestGestureRecognizer(n_estimators=200)
    rf.train(X, y, class_names)

    outpath = os.path.join(out_dir, 'random_forest_recognizer.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(rf, f)

    print('Saved trained recognizer to', outpath)
