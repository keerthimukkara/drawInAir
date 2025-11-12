"""
Visualize preprocessing steps for gesture sequences.

Produces PNG images for each sample showing:
- raw trajectory
- resampled (n=128)
- centered
- scaled (unit max distance)
- feature vector bar chart

Usage:
    python visualize_preprocessing.py --dataset gesture_dataset --out preproc_visualizations --max 20

"""
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from eval_recognizer import load_dataset, preprocess_sequence, extract_features_from_seq


def plot_sequence(ax, pts, title=None, connect=True):
    pts = np.array(pts)
    if pts.size == 0:
        return
    ax.plot(pts[:,0], pts[:,1], '-o' if connect else 'o', markersize=3)
    ax.invert_yaxis()
    ax.set_aspect('equal', 'box')
    if title:
        ax.set_title(title)


def visualize_sample(idx, pts, label, out_dir, n=128):
    os.makedirs(out_dir, exist_ok=True)
    raw = np.array(pts)

    # Resample
    resampled = preprocess_sequence(raw, n=n)

    # Center
    centroid = np.mean(resampled, axis=0)
    centered = resampled - centroid

    # Scale
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist > 0:
        scaled = centered / max_dist
    else:
        scaled = centered

    # Features
    features = extract_features_from_seq(resampled)

    # Create figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.ravel()
    plot_sequence(axs[0], raw, title=f'Raw (label: {label})')
    plot_sequence(axs[1], resampled, title=f'Resampled (n={n})')
    plot_sequence(axs[2], centered, title='Centered')
    plot_sequence(axs[3], scaled, title='Scaled (unit max)')

    # Feature bar chart
    axs[4].bar(np.arange(len(features)), features)
    axs[4].set_title('Feature vector')

    # Empty / metadata
    axs[5].axis('off')
    axs[5].text(0.1, 0.5, f'Sample: {idx}\nPoints: {len(raw)}', fontsize=12)

    plt.tight_layout()
    fname = os.path.join(out_dir, f'sample_{idx}_{label}.png')
    fig.savefig(fname)
    plt.close(fig)


def main(dataset_dir, out_dir, max_samples, per_class=0, do_all=False):
    X_seqs, y, labels = load_dataset(dataset_dir)
    if len(X_seqs) == 0:
        print('No sequences found in', dataset_dir)
        return

    print(f'Found {len(X_seqs)} sequences across {len(labels)} classes: {labels}')

    os.makedirs(out_dir, exist_ok=True)
    if do_all:
        # visualize every sequence (be careful: may be large)
        count = len(X_seqs)
        indices = list(range(count))
    elif per_class > 0:
        # select up to per_class samples for each label
        indices = []
        for li, lab in enumerate(labels):
            found = [i for i, yy in enumerate(y) if yy == li]
            if not found:
                continue
            indices.extend(found[:per_class])
    else:
        count = min(max_samples, len(X_seqs))
        indices = list(range(count))

    for idx in indices:
        seq = X_seqs[idx]
        label = labels[y[idx]] if idx < len(y) else 'unknown'
        try:
            visualize_sample(idx, seq, label, out_dir)
        except Exception as e:
            print('Failed for sample', idx, e)

    print('Saved visualizations to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='gesture_dataset')
    parser.add_argument('--out', default='preproc_visualizations')
    parser.add_argument('--max', type=int, default=20)
    parser.add_argument('--per-class', type=int, default=0,
                        help='Number of samples to visualize per gesture class (overrides --max)')
    parser.add_argument('--all', action='store_true', dest='do_all',
                        help='Visualize all sequences in the dataset (may be large)')
    args = parser.parse_args()
    main(args.dataset, args.out, args.max, per_class=args.per_class, do_all=args.do_all)
