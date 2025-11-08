"""
Train a RandomForest recognizer focused on 4 easy-to-draw gestures.

Selected gestures: line_horizontal, line_vertical, circle (clockwise), zigzag

This script loads any real examples for these classes from `gesture_dataset/`,
adds large amounts of synthetic samples for each class to reach a large dataset,
trains a RandomForest, saves the model to `models/focused_random_forest.pkl`,
and prints a cross-validated classification report.
"""
import os
import json
import math
import random
import pickle
import numpy as np
from gesture_recognition import RandomForestGestureRecognizer


SELECTED = ['line_horizontal', 'line_vertical', 'circle_clockwise', 'zigzag']


def load_real(dataset_dir='gesture_dataset'):
    out = {k: [] for k in SELECTED}
    for fname in os.listdir(dataset_dir):
        if not fname.lower().endswith('.json'):
            continue
        path = os.path.join(dataset_dir, fname)
        with open(path, 'r') as f:
            j = json.load(f)
        for entry in j.get('data', []):
            gtype = entry.get('gesture_type')
            if gtype not in out:
                continue
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
            out[gtype].append(pts)
    return out


def resample(pts, n=128):
    pts = np.array(pts)
    if len(pts) == 0:
        return pts
    old_idx = np.linspace(0, len(pts) - 1, len(pts))
    new_idx = np.linspace(0, len(pts) - 1, n)
    resampled = np.zeros((n, pts.shape[1]))
    for dim in range(pts.shape[1]):
        resampled[:, dim] = np.interp(new_idx, old_idx, pts[:, dim])
    return resampled


def normalize(pts):
    pts = np.array(pts)
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    return pts


def rotate(pts, angle_rad):
    R = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
                  [math.sin(angle_rad), math.cos(angle_rad)]])
    return pts.dot(R.T)


def jitter(pts, sigma=0.02):
    return pts + np.random.normal(scale=sigma, size=pts.shape)


def gen_line(vertical=False, n=128):
    if vertical:
        x = np.zeros(n)
        y = np.linspace(-1, 1, n)
    else:
        x = np.linspace(-1, 1, n)
        y = np.zeros(n)
    return np.stack([x, y], axis=1)


def gen_circle(n=128, clockwise=True):
    t = np.linspace(0, 2 * math.pi, n)
    if not clockwise:
        t = t[::-1]
    x = np.cos(t)
    y = np.sin(t)
    return np.stack([x, y], axis=1)


def gen_zigzag(n=128, zigs=6):
    xs = np.linspace(-1, 1, zigs * 2 + 1)
    ys = np.array([(-1) ** i * 0.6 for i in range(len(xs))])
    pts = np.stack([xs, ys], axis=1)
    return resample(pts, n=n)


def synth_for(gtype, count):
    out = []
    for i in range(count):
        if gtype == 'line_horizontal':
            pts = gen_line(vertical=False)
            angle = np.random.uniform(-0.15, 0.15)
        elif gtype == 'line_vertical':
            pts = gen_line(vertical=True)
            angle = np.random.uniform(-0.15, 0.15)
        elif gtype == 'circle_clockwise':
            pts = gen_circle()
            angle = np.random.uniform(-0.4, 0.4)
        elif gtype == 'zigzag':
            pts = gen_zigzag()
            angle = np.random.uniform(-0.4, 0.4)
        else:
            pts = gen_line()
            angle = 0.0

        pts = rotate(pts, angle)
        scale = np.random.uniform(0.7, 1.3)
        pts = pts * scale
        pts = jitter(pts, sigma=0.02)
        pts = resample(pts, n=128)
        pts = normalize(pts)
        out.append(pts)
    return out


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)

    os.makedirs('models', exist_ok=True)

    real = load_real('gesture_dataset')
    print('Real counts:', {k: len(v) for k, v in real.items()})

    # target per-class size
    target = 2000

    combined = {k: list(real.get(k, [])) for k in SELECTED}
    for k in SELECTED:
        have = len(combined.get(k, []))
        need = max(0, target - have)
        if need > 0:
            print(f'Generating {need} synthetic samples for {k} (have {have})')
            combined[k].extend(synth_for(k, need))

    # Flatten and prepare training arrays
    X = []
    y = []
    class_names = []
    for i, k in enumerate(SELECTED):
        class_names.append(k)
        for seq in combined[k]:
            X.append(seq)
            y.append(i)

    print('Final counts:', {k: len(combined[k]) for k in SELECTED})

    if not X:
        print('No data; aborting')
        raise SystemExit(1)

    rf = RandomForestGestureRecognizer(n_estimators=400)
    rf.train(X, y, class_names)

    outpath = os.path.join('models', 'focused_random_forest.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(rf, f)
    print('Saved focused model to', outpath)

    # Quick cross-validated evaluation
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        # extract features
        X_feats = np.array([rf._extract_features(s) for s in X])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_feats)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        y_arr = np.array(y)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X_scaled, y_arr, cv=cv)
        print('\nEvaluation:')
        print('Accuracy:', accuracy_score(y_arr, y_pred))
        print('\nClassification Report:\n')
        print(classification_report(y_arr, y_pred, target_names=class_names))
        print('\nConfusion Matrix:\n')
        print(confusion_matrix(y_arr, y_pred))
    except Exception as e:
        print('Evaluation skipped:', e)
