"""
Train RandomForest recognizer augmented with synthetic gesture samples.

This script:
- Loads real gestures from `gesture_dataset/`.
- Generates parametric synthetic gestures for missing classes (circle, star, triangle, etc.)
- Augments the dataset with those synthetic samples (with random rotation/scale/jitter)
- Trains RandomForestGestureRecognizer and saves to `models/random_forest_recognizer.pkl`
- Runs a cross-validated evaluation and prints a report.

This is intended as a quick way to bootstrap multi-class recognition when the
recorded dataset lacks examples for some gesture types. Synthetic examples are
not a perfect substitute for real human-drawn samples but help the recognizer
learn shape priors.
"""
import os
import json
import math
import random
import pickle
import numpy as np
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


def jitter(pts, sigma=0.005):
    noise = np.random.normal(scale=sigma, size=pts.shape)
    return pts + noise


def generate_circle(n=64, clockwise=True):
    t = np.linspace(0, 2 * math.pi, n)
    if not clockwise:
        t = t[::-1]
    x = np.cos(t)
    y = np.sin(t)
    pts = np.stack([x, y], axis=1)
    return pts


def generate_star(n=128, points=5):
    # Generate a star by alternating radii
    verts = []
    for i in range(points * 2):
        angle = i * math.pi / points
        r = 1.0 if i % 2 == 0 else 0.4
        verts.append((r * math.cos(angle), r * math.sin(angle)))
    verts = np.array(verts)
    # Interpolate to n points
    old_idx = np.linspace(0, len(verts) - 1, len(verts))
    new_idx = np.linspace(0, len(verts) - 1, n)
    res = np.zeros((n, 2))
    for d in range(2):
        res[:, d] = np.interp(new_idx, old_idx, verts[:, d])
    return res


def generate_triangle(n=64):
    verts = np.array([[0, 1], [-0.866, -0.5], [0.866, -0.5], [0, 1]])
    old_idx = np.linspace(0, len(verts) - 1, len(verts))
    new_idx = np.linspace(0, len(verts) - 1, n)
    res = np.zeros((n, 2))
    for d in range(2):
        res[:, d] = np.interp(new_idx, old_idx, verts[:, d])
    return res


def generate_rectangle(n=64):
    verts = np.array([[-1, -0.5], [-1, 0.5], [1, 0.5], [1, -0.5], [-1, -0.5]])
    old_idx = np.linspace(0, len(verts) - 1, len(verts))
    new_idx = np.linspace(0, len(verts) - 1, n)
    res = np.zeros((n, 2))
    for d in range(2):
        res[:, d] = np.interp(new_idx, old_idx, verts[:, d])
    return res


def generate_zigzag(n=64, zigs=4):
    xs = np.linspace(-1, 1, zigs * 2 + 1)
    ys = np.array([(-1) ** i * 0.6 for i in range(len(xs))])
    pts = np.stack([xs, ys], axis=1)
    return resample(pts, n=n)


def generate_wave(n=128, cycles=2):
    t = np.linspace(0, 2 * math.pi * cycles, n)
    x = np.linspace(-1, 1, n)
    y = 0.6 * np.sin(t)
    return np.stack([x, y], axis=1)


def generate_spiral(n=128, turns=3):
    t = np.linspace(0, 2 * math.pi * turns, n)
    r = np.linspace(0.1, 1.0, n)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.stack([x, y], axis=1)


def generate_line(vertical=False, n=64):
    if vertical:
        x = np.zeros(n)
        y = np.linspace(-1, 1, n)
    else:
        x = np.linspace(-1, 1, n)
        y = np.zeros(n)
    return np.stack([x, y], axis=1)


SYNTHETIC_GENERATORS = {
    'circle_clockwise': lambda: generate_circle(clockwise=True),
    'circle_counterclockwise': lambda: generate_circle(clockwise=False),
    'star': lambda: generate_star(),
    'triangle': lambda: generate_triangle(),
    'rectangle': lambda: generate_rectangle(),
    'zigzag': lambda: generate_zigzag(),
    'wave': lambda: generate_wave(),
    'spiral': lambda: generate_spiral(),
    'line_horizontal': lambda: generate_line(vertical=False),
    'line_vertical': lambda: generate_line(vertical=True),
}


def make_synthetic_samples(gesture_name, count=200):
    out = []
    gen = SYNTHETIC_GENERATORS.get(gesture_name)
    if gen is None:
        return out
    for i in range(count):
        pts = gen()
        # random rotation
        angle = random.uniform(-0.4, 0.4)
        pts = rotate(pts, angle)
        # random scaling
        s = random.uniform(0.8, 1.2)
        pts = pts * s
        # jitter
        pts = jitter(pts, sigma=0.02)
        # resample and normalize
        pts = resample(pts, n=128)
        pts = normalize(pts)
        out.append(pts)
    return out


def preprocess_sequence(pts, n=128):
    return normalize(resample(pts, n=n))


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    dataset_dir = 'gesture_dataset'
    out_dir = 'models'
    os.makedirs(out_dir, exist_ok=True)

    real = load_dataset(dataset_dir)
    print('Real classes found:', {k: len(v) for k, v in real.items()})

    # Decide which classes we want to train for
    desired = list(SYNTHETIC_GENERATORS.keys())

    # Build combined dataset (real + synthetic for missing/underrepresented)
    combined = {k: list(v) for k, v in real.items()}
    for g in desired:
        have = len(combined.get(g, []))
        if have < 50:
            need = max(100 - have, 200)
            print(f"Generating {need} synthetic samples for '{g}' (have={have})")
            synthetic = make_synthetic_samples(g, count=need)
            combined.setdefault(g, []).extend(synthetic)

    # Prepare training arrays
    X = []
    y = []
    class_names = []
    for i, (gtype, seqs) in enumerate(sorted(combined.items())):
        class_names.append(gtype)
        for s in seqs:
            X.append(preprocess_sequence(s, n=128))
            y.append(i)

    print('Final class counts:', {cn: sum(1 for label in y if class_names[label] == cn) for cn in class_names})

    if not X:
        print('No training data found')
        raise SystemExit(1)

    rf = RandomForestGestureRecognizer(n_estimators=300)
    rf.train(X, y, class_names)

    outpath = os.path.join(out_dir, 'random_forest_recognizer.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(rf, f)

    print('Saved trained recognizer to', outpath)

    # Optional: quick evaluation using existing eval script logic (lightweight)
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        # Build features from X (use RandomForest internal extractor)
        def extract_features(seq):
            return rf._extract_features(seq)

        X_feats = np.array([extract_features(s) for s in X])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_feats)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        y_arr = np.array(y)
        cv = StratifiedKFold(n_splits=min(5, max(2, len(y_arr)//20)), shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X_scaled, y_arr, cv=cv)
        print('\nEvaluation:')
        print('Accuracy:', accuracy_score(y_arr, y_pred))
        print('\nClassification Report:\n')
        print(classification_report(y_arr, y_pred, target_names=class_names))
        print('\nConfusion Matrix:\n')
        print(confusion_matrix(y_arr, y_pred))
    except Exception as e:
        print('Evaluation skipped due to error:', e)
