"""
Evaluate RandomForest recognizer with cross-validation on the gesture_dataset.
Prints classification report, confusion matrix and overall accuracy.
"""
import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_dataset(dataset_dir='gesture_dataset'):
    X_seqs = []
    y = []
    labels = []
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
            X_seqs.append(pts)
            if gtype not in labels:
                labels.append(gtype)
            y.append(labels.index(gtype))
    return X_seqs, np.array(y), labels


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


def extract_features_from_seq(seq):
    # Duplicate of RandomForestGestureRecognizer._extract_features but adapted for preprocessed seq
    points = np.array(seq)
    features = []
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    aspect_ratio = width / (height + 1e-6)
    features.extend([width, height, aspect_ratio])
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    total_length = np.sum(distances)
    features.append(total_length)
    if len(points) >= 3:
        angles = []
        for i in range(1, len(points) - 1):
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angles.append(angle)
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        features.extend([mean_angle, std_angle])
    else:
        features.extend([0, 0])
    direction_changes = np.sum(np.abs(np.diff(np.sign(np.diff(points, axis=0)), axis=0)))
    features.append(direction_changes)
    start_end_dist = np.linalg.norm(points[0] - points[-1])
    features.append(start_end_dist)
    if len(points) >= 3:
        area = 0.5 * np.abs(np.sum(points[:-1,0]*points[1:,1] - points[1:,0]*points[:-1,1])) 
        circularity = (4 * np.pi * area) / (total_length ** 2 + 1e-6)
        features.append(circularity)
    else:
        features.append(0)
    velocities = distances / (np.arange(1, len(distances) + 1) + 1e-6)
    features.extend([np.mean(velocities), np.std(velocities)])
    centroid = np.mean(points, axis=0)
    centroid_dists = np.sqrt(np.sum((points - centroid)**2, axis=1))
    features.extend([np.mean(centroid_dists), np.std(centroid_dists)])
    features.append(len(points) / 100.0)
    return np.array(features)


if __name__ == '__main__':
    X_seqs, y, labels = load_dataset('gesture_dataset')
    if len(X_seqs) == 0:
        print('No data found in gesture_dataset')
        raise SystemExit(1)
    print('Classes:', labels)
    X_feats = np.array([extract_features_from_seq(preprocess_sequence(s)) for s in X_seqs])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feats)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    cv = StratifiedKFold(n_splits=min(5, max(2, len(y)//2)), shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    print('Accuracy:', accuracy_score(y, y_pred))
    print('\nClassification Report:\n')
    print(classification_report(y, y_pred, target_names=labels))
    print('\nConfusion Matrix:\n')
    print(confusion_matrix(y, y_pred))
