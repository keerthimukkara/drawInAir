"""
Smoothing Algorithms for Hand Gesture Tracking
Author: Swapana
Purpose: Reduce noise and jitter in hand tracking coordinates
"""

import numpy as np
from collections import deque
from scipy.signal import savgol_filter
import time

class MovingAverageFilter:
    """
    Algorithm 1: Moving Average Filter
    
    Why: Simple sliding window average
    - Reduces high-frequency jitter
    - Computationally efficient: O(1) per update
    - Easy to implement and understand
    
    Limitation: Introduces lag proportional to window size
    Use case: When simplicity is priority
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer_x = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)
    
    def filter(self, x, y):
        self.buffer_x.append(x)
        self.buffer_y.append(y)
        return np.mean(self.buffer_x), np.mean(self.buffer_y)


class ExponentialMovingAverage:
    """
    Algorithm 2: Exponential Moving Average (EMA)
    
    Why: Weighted average with more importance to recent values
    - Less lag than simple moving average
    - Single parameter (alpha) controls smoothness
    - Memory efficient: only stores last value
    
    Formula: smoothed = alpha * current + (1 - alpha) * previous
    
    Why not simple average: EMA responds faster to direction changes
    Use case: When responsiveness is important
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # 0 < alpha < 1 (lower = smoother)
        self.prev_x = None
        self.prev_y = None
    
    def filter(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return x, y
        
        smoothed_x = self.alpha * x + (1 - self.alpha) * self.prev_x
        smoothed_y = self.alpha * y + (1 - self.alpha) * self.prev_y
        
        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        return smoothed_x, smoothed_y


class KalmanFilter:
    """
    Algorithm 3: Kalman Filter
    
    Why: Optimal state estimation for linear systems with Gaussian noise
    - Predicts future positions based on velocity
    - Handles measurement noise optimally
    - Used in aerospace, robotics (proven technology)
    
    Components:
    - State: [x, y, vx, vy] (position + velocity)
    - Prediction: Estimate next position
    - Update: Correct prediction with measurement
    
    Why not simpler filters: Kalman considers motion dynamics
    Use case: When predictive tracking is needed
    """
    def __init__(self):
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # State covariance (uncertainty)
        self.P = np.eye(4) * 1000
        
        # Transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise
        self.Q = np.eye(4) * 0.1
        
        # Measurement noise
        self.R = np.eye(2) * 10
        
        self.initialized = False
    
    def filter(self, x, y):
        measurement = np.array([x, y])
        
        if not self.initialized:
            self.state[:2] = measurement
            self.initialized = True
            return x, y
        
        # Prediction step
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update step
        y_residual = measurement - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y_residual
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.state[0], self.state[1]


class SavitzkyGolayFilter:
    """
    Algorithm 4: Savitzky-Golay Filter
    
    Why: Polynomial regression-based smoothing
    - Preserves peaks and shape features
    - Better for shape recognition (circles, lines)
    - Smooths while maintaining feature integrity
    
    How it works: Fits polynomial to window, replaces center point
    
    Why not moving average: Preserves sharp features better
    Use case: When shape accuracy is critical
    """
    def __init__(self, window_size=11, poly_order=3):
        self.window_size = window_size
        self.poly_order = poly_order
        self.buffer_x = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)
    
    def filter(self, x, y):
        self.buffer_x.append(x)
        self.buffer_y.append(y)
        
        if len(self.buffer_x) < self.window_size:
            return x, y
        
        smoothed_x = savgol_filter(list(self.buffer_x), 
                                   self.window_size, self.poly_order)[-1]
        smoothed_y = savgol_filter(list(self.buffer_y), 
                                   self.window_size, self.poly_order)[-1]
        
        return smoothed_x, smoothed_y


class OneEuroFilter:
    """
    Algorithm 5: One Euro Filter
    
    Why: Adaptive filter that adjusts to motion speed
    - Low lag during fast movements
    - High smoothing during slow movements
    - Single tuning parameter (min_cutoff)
    
    Used in: VR/AR systems, game controllers
    
    How it works: Cutoff frequency increases with velocity
    
    Why not fixed smoothing: Adapts to user behavior
    Use case: Best overall performance for gesture tracking
    """
    def __init__(self, min_cutoff=1.0, beta=0.007):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.x_filter = LowPassFilter(self._alpha(min_cutoff))
        self.y_filter = LowPassFilter(self._alpha(min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(min_cutoff))
        self.dy_filter = LowPassFilter(self._alpha(min_cutoff))
        self.prev_time = None
    
    def _alpha(self, cutoff):
        te = 1.0 / 60.0  # Assume 60 FPS
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
    def filter(self, x, y):
        curr_time = time.time()
        
        if self.prev_time is None:
            self.prev_time = curr_time
            self.x_filter.filter(x)
            self.y_filter.filter(y)
            return x, y
        
        dt = curr_time - self.prev_time
        self.prev_time = curr_time
        
        # Estimate velocity
        dx = self.dx_filter.filter((x - self.x_filter.last_value) / dt 
                                   if dt > 0 else 0)
        dy = self.dy_filter.filter((y - self.y_filter.last_value) / dt 
                                   if dt > 0 else 0)
        
        # Adaptive cutoff based on velocity
        cutoff = self.min_cutoff + self.beta * np.sqrt(dx**2 + dy**2)
        
        # Update alpha based on velocity
        self.x_filter.alpha = self._alpha(cutoff)
        self.y_filter.alpha = self._alpha(cutoff)
        
        return self.x_filter.filter(x), self.y_filter.filter(y)


class LowPassFilter:
    """Helper class for OneEuroFilter"""
    def __init__(self, alpha):
        self.alpha = alpha
        self.last_value = None
    
    def filter(self, value):
        if self.last_value is None:
            self.last_value = value
            return value
        
        filtered = self.alpha * value + (1 - self.alpha) * self.last_value
        self.last_value = filtered
        return filtered


class MedianFilter:
    """
    Algorithm 6: Median Filter
    
    Why: Non-linear filter that removes outliers
    - Excellent for removing spikes/jumps
    - Preserves edges better than averaging
    - Robust to extreme values
    
    How it works: Takes median of window (middle value when sorted)
    
    Why not mean: Outliers don't affect median
    Use case: When tracking has occasional large errors
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer_x = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)
    
    def filter(self, x, y):
        self.buffer_x.append(x)
        self.buffer_y.append(y)
        return np.median(self.buffer_x), np.median(self.buffer_y)


class SmoothingComparator:
    """
    Utility class to compare all smoothing algorithms
    """
    def __init__(self):
        self.filters = {
            'moving_average': MovingAverageFilter(window_size=5),
            'ema': ExponentialMovingAverage(alpha=0.3),
            'kalman': KalmanFilter(),
            'savitzky_golay': SavitzkyGolayFilter(window_size=11, poly_order=3),
            'one_euro': OneEuroFilter(min_cutoff=1.0, beta=0.007),
            'median': MedianFilter(window_size=5)
        }
        
        self.results = {name: [] for name in self.filters.keys()}
    
    def process_point(self, x, y):
        """Apply all filters to a point"""
        for name, filter_obj in self.filters.items():
            smoothed_x, smoothed_y = filter_obj.filter(x, y)
            self.results[name].append((smoothed_x, smoothed_y))
        
        return self.results
    
    def get_comparison_metrics(self, ground_truth):
        """Calculate performance metrics"""
        metrics = {}
        
        for name, smoothed_points in self.results.items():
            if len(smoothed_points) != len(ground_truth):
                continue
            
            # Calculate Mean Squared Error
            mse = np.mean([
                (gt[0] - sp[0])**2 + (gt[1] - sp[1])**2
                for gt, sp in zip(ground_truth, smoothed_points)
            ])
            
            # Calculate lag (cross-correlation)
            metrics[name] = {
                'mse': mse,
                'smoothness': self._calculate_smoothness(smoothed_points)
            }
        
        return metrics
    
    def _calculate_smoothness(self, points):
        """Calculate path smoothness (lower is smoother)"""
        if len(points) < 2:
            return 0
        
        accelerations = []
        for i in range(1, len(points) - 1):
            ax = points[i+1][0] - 2*points[i][0] + points[i-1][0]
            ay = points[i+1][1] - 2*points[i][1] + points[i-1][1]
            accelerations.append(np.sqrt(ax**2 + ay**2))
        
        return np.mean(accelerations) if accelerations else 0


# Algorithm Selection Guidelines
"""
RECOMMENDATION FOR DRAW IN AIR PROJECT:

Primary Choice: ONE EURO FILTER
- Best balance of smoothness and responsiveness
- Adapts to drawing speed automatically
- Industry standard for gesture interfaces

Backup Choice: KALMAN FILTER
- Good prediction capability
- Handles consistent motion well

For Shape Recognition: SAVITZKY-GOLAY
- Preserves shape features
- Better for pattern matching

Avoid:
- Simple Moving Average: Too much lag
- Median Filter: Can distort smooth curves

Implementation Strategy:
1. Use One Euro Filter for real-time drawing
2. Apply Savitzky-Golay post-processing for shape recognition
3. Use Kalman Filter for gesture prediction/anticipation
"""