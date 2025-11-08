# Draw in Air
🎨 Draw in Air — Gesture-Based Drawing & Recognition with AI

Draw in Air is an interactive computer vision project that turns your hand gestures into digital art.
Using MediaPipe and OpenCV, it tracks your index finger in real time and lets you “draw” mid-air — no pen, no screen, just movement.

The system recognizes your drawn shapes (like circles, lines, or zigzags) using multiple AI models — from Dynamic Time Warping (DTW) to a Random Forest classifier — and even allows you to collect your own gesture dataset and retrain models directly from the built-in Tkinter UI.

✨ Think of it as an AI-powered sketchpad in the air.

---

## Features
- Live webcam drawing (point with index finger, open palm to start, closed fist to stop by default)
- Save drawings as images and export drawn paths as JSON
- In-UI labeled data collection: record gestures and save timestamped JSON samples to `gesture_dataset/`
- Multiple recognizers: template-matching, DTW, and a trained RandomForest classifier
- Scripts to generate synthetic data, train focused models and evaluate performance
- Retrain button in the UI (runs trainer in background) with a progress indicator and training log

---

## Quick Start (Windows / PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the UI:

```powershell
python .\ui_application.py
```

The app window will open. Use the left panel controls to start/stop drawing, record labeled gestures, save drawings, or retrain the model.

---

## Important Files

- `ui_application.py` — Tkinter application. Main UI, drawing, recording, and recognition logic. It now prefers the focused model `models/focused_random_forest.pkl` when available.
- `gesture_recognition.py` — Recognition algorithms (TemplateMatching, DynamicTimeWarping, HMM, RandomForest, KNN). Feature extraction logic for RandomForest lives here.
- `train_focus_gestures.py` — Trains a RandomForest focused on the selected 4 easy gestures (line_horizontal, line_vertical, circle_clockwise, zigzag). Uses real samples found in `gesture_dataset/` and generates synthetic samples to reach a target class size; saves `models/focused_random_forest.pkl`.
- `train_with_synthetic.py` — General-purpose trainer that seeds recognizers with synthetic templates and trains a RandomForest on many synthetic examples.
- `train_recognizer.py` — Simpler trainer that trains a RandomForest from existing dataset files without heavy synthetic augmentation.
- `eval_recognizer.py` — Evaluates dataset using the same feature extractor as RandomForest and prints classification report and confusion matrix.
- `dataset_collection.py` — (Optional) scripts/helpers for collecting datasets (if present/used).

Model/artifacts directory:
- `models/` — trained models and logs (e.g. `focused_random_forest.pkl`, `random_forest_recognizer.pkl`, `retrain_log.txt`).

Dataset directory:
- `gesture_dataset/` — JSON session files with recorded samples. Use the UI to create new labeled samples.

Saved drawings:
- `saved_drawings/` — PNG images saved from the UI when you press the Save button or press `s`.

---

## Recording labeled samples (in the UI)

1. Click "🎙️ Record Labeled Gesture" and type a label (e.g. `circle_clockwise` or `line_horizontal`).
2. Draw the gesture using the camera view (open palm to start drawing; closed fist to stop; or use the Start Drawing button).
3. Click "🔴 Save Labeled Sample" to save a timestamped JSON file to `gesture_dataset/`.

Saved JSON format (single-sample session):

```json
{
  "session_id": "user_<label>_<timestamp>",
  "total_samples": 1,
  "gesture_types": ["<label>"],
  "data": [
    {
      "gesture_type": "<label>",
      "landmarks": [x1, y1, z1, x2, y2, z2, ...],
      "index_tip": [x_last, y_last, z_last],
      "timestamp": "..."
    }
  ]
}
```

Notes:
- Coordinates are normalized relative to the camera resolution (x in [0..1], y in [0..1], z=0.0 placeholder).
- Keep at least ~20–30 samples per class to start seeing improvements. 100+ samples per class is ideal for robust performance.

---

## Retraining from the UI

- Click the "🔁 Retrain Model" button in the File Operations panel.
- The UI will ask for confirmation and then run `train_focus_gestures.py` in a background thread.
- A simple determinate progress bar and percentage label appear while training runs and the trainer output is written to `models/retrain_log.txt`.
- When training completes the app attempts to reload `models/focused_random_forest.pkl` automatically.

If you prefer to retrain from the command line:

```powershell
python .\train_focus_gestures.py
# or for the synthetic-heavy general trainer
python .\train_with_synthetic.py
```

Training logs and output model:

- `models/retrain_log.txt` — stdout/stderr from the trainer (useful for debugging)
- `models/focused_random_forest.pkl` — the trained focused RandomForest model used by the UI

---

## Evaluation

Run the evaluation script to get a classification report and confusion matrix for the dataset:

```powershell
python .\eval_recognizer.py
```

By default the evaluation script resamples gestures to a fixed length and computes geometric features identical to those used by the RandomForest. If you want to evaluate only real (non-synthetic) samples, see the troubleshooting section below or ask me to run a custom evaluation for you.

---

## How recognition works (short)

- TemplateMatching: simple normalized distance to a set of stored templates (fast, no training).
- DynamicTimeWarping (DTW): aligns time series to handle speed differences.
- RandomForest: feature-based classifier using geometric & temporal features extracted from resampled traces. This is the main model used in the UI when available.

Preprocessing for all recognizers: resample to fixed length (128 points), center to centroid and scale to unit size to handle speed and scale variations.

---

## Tips to improve accuracy

1. Collect real samples for every gesture you want the app to recognize. Synthetic data helps but real samples are essential.
2. Keep drawing speed/scale consistent during collection or rely on the resampling/normalization steps.
3. Use the same labels consistently (e.g. `circle_clockwise`, `line_horizontal`).
4. If recognition is noisy, try collecting more samples with variation (angles, speed, hand orientation).
5. After collecting samples, click Retrain Model and test the model in the UI.

---

## Troubleshooting

- If the UI errors on import, check that dependencies are installed and your Python interpreter matches the virtualenv where libraries were installed.
- If retraining takes too long, you can edit `train_focus_gestures.py` to reduce `target` per-class size or reduce `n_estimators` in RandomForest for faster but less accurate training.
- If `models/focused_random_forest.pkl` does not load after retraining, inspect `models/retrain_log.txt` for errors.

---

## Development notes

- The UI code is intentionally self-contained in `ui_application.py` to simplify running the app. Recognition algorithms are in `gesture_recognition.py` for easier experimentation.
- The project includes multiple training scripts so you can choose the workflow that matches your data situation (real-only vs. synthetic augmentation).

---


