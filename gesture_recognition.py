"""
Gesture Recognition Algorithms for Draw in Air
Author: Swapna
Purpose: Classify and recognize drawing gestures
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
#import tensorflow as tf
#from tensorflow import keras
from hmmlearn import hmm


class TemplateMatching:
    """
    Algorithm 1: Distance-based Template Matching
    
    Why: Simple and interpretable baseline
    - Fast computation: O(n) comparison
    - No training required
    - Works well for consistent gestures
    
    How it works: Compare input to stored templates using distance metric
    """
    def __init__(self):
        self.templates = {}
    
    def add_template(self, gesture_name, points):
        """Store normalized template"""
        normalized = self._normalize_gesture(points)
        self.templates[gesture_name] = normalized
    
    def _normalize_gesture(self, points):
        """Normalize to unit scale and center"""
        points = np.array(points)

        if points.size == 0:
            return points

        # Resample to fixed length to make templates comparable
        points = self._resample(points, 128)

        # Center at origin
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # Scale to unit size (preserve aspect)
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered

        return normalized
    
    def recognize(self, points):
        """Match input gesture to closest template"""
        if not self.templates:
            return None, 0.0
        
        normalized = self._normalize_gesture(points)
        
        best_match = None
        best_distance = float('inf')
        
        for name, template in self.templates.items():
            # Resample to same length
            resampled_input = self._resample(normalized, len(template))
            
            # Calculate distance
            distance = np.mean([
                euclidean(p1, p2) 
                for p1, p2 in zip(resampled_input, template)
            ])
            
            if distance < best_distance:
                best_distance = distance
                best_match = name
        
        confidence = 1.0 / (1.0 + best_distance)
        return best_match, confidence
    
    def _resample(self, points, n):
        """Resample points to n samples using linear interpolation"""
        points = np.array(points)
        if points.size == 0:
            return points

        old_indices = np.linspace(0, len(points) - 1, len(points))
        new_indices = np.linspace(0, len(points) - 1, n)

        resampled = np.zeros((n, points.shape[1]))
        for dim in range(points.shape[1]):
            resampled[:, dim] = np.interp(new_indices, old_indices, points[:, dim])

        return resampled


class DynamicTimeWarping:
    """
    Algorithm 2: Dynamic Time Warping (DTW)
    
    Why: Handles variable speed gestures optimally
    - Aligns sequences with different timings
    - Robust to speed variations (drawing fast vs slow)
    - Standard for time-series comparison
    
    How it works: Finds optimal alignment between sequences
    Compute DTW distance using fastdtw, which aligns sequences of different lengths/speeds.
    Select gesture with minimum distance
    """
    def __init__(self):
        self.templates = {}
    
    def add_template(self, gesture_name, points):
        """Store template for DTW comparison"""
        pts = np.array(points)
        if pts.size == 0:
            self.templates[gesture_name] = pts
            return

        # resample to fixed length
        n = 128
        old_idx = np.linspace(0, len(pts) - 1, len(pts))
        new_idx = np.linspace(0, len(pts) - 1, n)
        resampled = np.zeros((n, pts.shape[1]))
        for dim in range(pts.shape[1]):
            resampled[:, dim] = np.interp(new_idx, old_idx, pts[:, dim])

        # center and scale
        centroid = np.mean(resampled, axis=0)
        resampled = resampled - centroid
        max_dist = np.max(np.linalg.norm(resampled, axis=1))
        if max_dist > 0:
            resampled = resampled / max_dist

        self.templates[gesture_name] = resampled
    
    def recognize(self, points):
        """Use DTW to find best matching gesture"""
        if not self.templates:
            return None, 0.0
        pts = np.array(points)
        if pts.size == 0:
            return None, 0.0

        # resample and normalize input same way as templates
        n = 128
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

        best_match = None
        best_distance = float('inf')

        for name, template in self.templates.items():
            try:
                distance, _ = fastdtw(resampled, template, dist=euclidean)
            except Exception:
                # fallback: mean euclidean distance
                distance = np.mean([euclidean(a, b) for a, b in zip(resampled, template)])

            if distance < best_distance:
                best_distance = distance
                best_match = name

        # Confidence: inverse scaled distance
        confidence = 1.0 / (1.0 + best_distance)
        return best_match, confidence


class HMMGestureRecognizer:
    """
    Algorithm 3: Hidden Markov Model (HMM)
    
    Why: Models temporal sequences probabilistically
    - Captures gesture dynamics (start, middle, end)
    - Handles noise and variations well
    - Used in speech recognition (proven technology)
    
    How it works: Models gesture as sequence of hidden states
    Train an HMM for each gesture type using hmmlearn.GaussianHMM.
    Each model learns the “gesture dynamics” — how features evolve over time.
    For recognition, compute log-likelihood of input under each HMM → choose max likelihood.
    """
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.models = {}
    
    def train(self, gesture_name, sequences):
        """Train HMM for a specific gesture"""
        # Concatenate all training sequences
        X = np.concatenate(sequences)
        lengths = [len(seq) for seq in sequences]
        
        # Train Gaussian HMM
        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=100
        )
        
        model.fit(X, lengths)
        self.models[gesture_name] = model
    
    def recognize(self, points):
        """Classify gesture using trained HMMs"""
        if not self.models:
            return None, 0.0
        pts = np.array(points)
        if pts.size == 0:
            return None, 0.0

        # Normalize/resample like other recognizers to reduce scale/speed sensitivity
        n = 128
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

        best_match = None
        best_score = float('-inf')

        for name, model in self.models.items():
            try:
                score = model.score(resampled)
                if score > best_score:
                    best_score = score
                    best_match = name
            except Exception:
                continue

        # Convert log-likelihood to confidence (sigmoid-like)
        confidence = 1.0 / (1.0 + np.exp(-best_score / 50.0))
        return best_match, confidence


class RandomForestGestureRecognizer:
    """
    Algorithm 5: Random Forest Classifier
    
    Why: Ensemble learning with handcrafted features
    - Fast training and prediction
    - Handles non-linear relationships
    - Less data hungry than deep learning
    - Feature importance analysis
    
    How it works: Extract features, train ensemble of decision tree
    """
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.scaler = StandardScaler()
        self.class_names = []
    
    def _extract_features(self, points):
        """Extract geometric and statistical features"""
        if len(points) < 2:
            return np.zeros(15)
        
        points = np.array(points)
        
        features = []
        
        # 1. Bounding box features
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = width / (height + 1e-6)
        features.extend([width, height, aspect_ratio])
        
        # 2. Path length
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        total_length = np.sum(distances)
        features.append(total_length)
        
        # 3. Curvature features
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
        
        # 4. Direction changes
        direction_changes = np.sum(np.abs(np.diff(np.sign(np.diff(points, axis=0)), axis=0)))
        features.append(direction_changes)
        
        # 5. Start-end distance
        start_end_dist = euclidean(points[0], points[-1])
        features.append(start_end_dist)
        
        # 6. Circularity (ratio of area to perimeter^2)
        # Approximate area using shoelace formula
        if len(points) >= 3:
            area = 0.5 * np.abs(np.sum(
                points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1]
            ))
            circularity = (4 * np.pi * area) / (total_length ** 2 + 1e-6)
            features.append(circularity)
        else:
            features.append(0)
        
        # 7. Velocity statistics
        velocities = distances / (np.arange(1, len(distances) + 1) + 1e-6)
        features.extend([np.mean(velocities), np.std(velocities)])
        
        # 8. Centroid distance statistics
        centroid = np.mean(points, axis=0)
        centroid_dists = np.sqrt(np.sum((points - centroid)**2, axis=1))
        features.extend([np.mean(centroid_dists), np.std(centroid_dists)])
        
        # 9. Number of points (normalized)
        features.append(len(points) / 100.0)
        
        return np.array(features)
    
    def train(self, X_train, y_train, class_names):
        """Train Random Forest on extracted features"""
        self.class_names = class_names
        
        # Extract features from all training samples
        X_features = np.array([self._extract_features(gesture) for gesture in X_train])
        
        # Normalize features
        X_features = self.scaler.fit_transform(X_features)
        
        # Train model
        self.model.fit(X_features, y_train)
    
    def recognize(self, points):
        """Classify gesture using Random Forest"""
        features = self._extract_features(points).reshape(1, -1)
        features = self.scaler.transform(features)
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        gesture_name = self.class_names[prediction] if self.class_names else str(prediction)
        confidence = probabilities[prediction]
        
        return gesture_name, float(confidence)
    
    def get_feature_importance(self):
        """Return feature importance scores"""
        return self.model.feature_importances_


class KNNGestureRecognizer:
    """
    Algorithm 6: K-Nearest Neighbors (KNN)
    
    Why: Non-parametric, instance-based learning
    - No training phase (lazy learning)
    - Simple and intuitive
    - Good baseline for comparison
    - Works well with small datasets
    
    How it works: Find k closest training examples, vote for class
    """
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.class_names = []
    
    def _extract_features(self, points):
        """Use same features as Random Forest for consistency"""
        recognizer = RandomForestGestureRecognizer()
        return recognizer._extract_features(points)
    
    def train(self, X_train, y_train, class_names):
        """Train KNN on feature vectors"""
        self.class_names = class_names
        
        X_features = np.array([self._extract_features(gesture) for gesture in X_train])
        X_features = self.scaler.fit_transform(X_features)
        
        self.model.fit(X_features, y_train)
    
    def recognize(self, points):
        """Classify using KNN"""
        features = self._extract_features(points).reshape(1, -1)
        features = self.scaler.transform(features)
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        gesture_name = self.class_names[prediction] if self.class_names else str(prediction)
        confidence = probabilities[prediction]
        
        return gesture_name, float(confidence)


# Algorithm Comparison and Selection Guide
"""
ALGORITHM COMPARISON FOR DRAW IN AIR:

┌─────────────────────┬──────────┬────────────┬─────────────┬──────────────┐
│ Algorithm           │ Accuracy │ Speed      │ Data Needed │ Complexity   │
├─────────────────────┼──────────┼────────────┼─────────────┼──────────────┤
│ Template Matching   │ Low      │ Very Fast  │ Minimal     │ Very Simple  │
│ DTW                 │ Medium   │ Fast       │ Minimal     │ Simple       │
│ HMM                 │ Medium   │ Medium     │ Medium      │ Medium       │
│ Random Forest       │ High     │ Fast       │ Medium      │ Medium       │
│ KNN                 │ Medium   │ Very Fast  │ Small       │ Simple       │
└─────────────────────┴──────────┴────────────┴─────────────┴──────────────┘

"""