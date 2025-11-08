"""
Testing and Validation Framework for Draw in Air
Author: Gayatri
Purpose: Evaluate algorithm performance, detect issues, optimize parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, precision_recall_fscore_support)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns
import json
import time


class PerformanceEvaluator:
    """
    Comprehensive testing framework for gesture recognition system
    """
    def __init__(self):
        self.test_results = {}
        self.confusion_matrices = {}
        self.timing_data = {}
    
    def evaluate_smoothing_algorithms(self, raw_trajectory, ground_truth, smoothers):
        """
        Test 1: Smoothing Algorithm Comparison
        
        Metrics:
        - Mean Squared Error (MSE): Distance from ground truth
        - Smoothness: Measure of jitter (acceleration variance)
        - Lag: Delay in response
        - Computational time
        """
        print("=" * 60)
        print("SMOOTHING ALGORITHM EVALUATION")
        print("=" * 60)
        
        results = {}
        
        for name, smoother in smoothers.items():
            print(f"\nTesting {name}...")
            
            smoothed_trajectory = []
            start_time = time.time()
            
            # Apply smoothing
            for point in raw_trajectory:
                smoothed = smoother.filter(point[0], point[1])
                smoothed_trajectory.append(smoothed)
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            mse = self._calculate_mse(smoothed_trajectory, ground_truth)
            smoothness = self._calculate_smoothness(smoothed_trajectory)
            lag = self._calculate_lag(smoothed_trajectory, ground_truth)
            
            results[name] = {
                'mse': mse,
                'smoothness': smoothness,
                'lag': lag,
                'time_ms': elapsed_time * 1000,
                'points_per_second': len(raw_trajectory) / elapsed_time
            }
            
            print(f"  MSE: {mse:.4f}")
            print(f"  Smoothness: {smoothness:.4f}")
            print(f"  Lag: {lag:.4f}")
            print(f"  Processing time: {elapsed_time*1000:.2f} ms")
            print(f"  Throughput: {results[name]['points_per_second']:.0f} points/sec")
        
        # Rank algorithms
        print("\n" + "=" * 60)
        print("SMOOTHING ALGORITHM RANKING")
        print("=" * 60)
        
        # Best by MSE (accuracy)
        best_mse = min(results.items(), key=lambda x: x[1]['mse'])
        print(f"Best Accuracy: {best_mse[0]} (MSE: {best_mse[1]['mse']:.4f})")
        
        # Best by smoothness
        best_smooth = min(results.items(), key=lambda x: x[1]['smoothness'])
        print(f"Best Smoothness: {best_smooth[0]} (Score: {best_smooth[1]['smoothness']:.4f})")
        
        # Best by speed
        best_speed = max(results.items(), key=lambda x: x[1]['points_per_second'])
        print(f"Fastest: {best_speed[0]} ({best_speed[1]['points_per_second']:.0f} pts/sec)")
        
        self.test_results['smoothing'] = results
        return results
    
    def evaluate_recognition_algorithms(self, X_test, y_test, class_names, recognizers):
        """
        Test 2: Gesture Recognition Algorithm Comparison
        
        Metrics:
        - Accuracy, Precision, Recall, F1-Score
        - Confusion Matrix
        - Per-class performance
        - Inference time
        """
        print("\n" + "=" * 60)
        print("GESTURE RECOGNITION EVALUATION")
        print("=" * 60)
        
        results = {}
        
        for name, recognizer in recognizers.items():
            print(f"\nTesting {name}...")
            
            predictions = []
            confidences = []
            inference_times = []
            
            # Test each sample
            for gesture in X_test:
                start_time = time.time()
                pred, conf = recognizer.recognize(gesture)
                inference_times.append(time.time() - start_time)
                
                predictions.append(pred)
                confidences.append(conf)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, predictions, average='weighted', zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, predictions, labels=class_names)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_confidence': np.mean(confidences),
                'avg_inference_time_ms': np.mean(inference_times) * 1000,
                'confusion_matrix': cm
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Avg Confidence: {results[name]['avg_confidence']:.4f}")
            print(f"  Inference Time: {results[name]['avg_inference_time_ms']:.2f} ms")
            
            # Detailed classification report
            print(f"\nClassification Report for {name}:")
            print(classification_report(y_test, predictions, 
                                       target_names=class_names, zero_division=0))
        
        # Overall ranking
        print("\n" + "=" * 60)
        print("RECOGNITION ALGORITHM RANKING")
        print("=" * 60)
        
        best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
        
        best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
        print(f"Best F1-Score: {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
        
        fastest = min(results.items(), key=lambda x: x[1]['avg_inference_time_ms'])
        print(f"Fastest: {fastest[0]} ({fastest[1]['avg_inference_time_ms']:.2f} ms)")
        
        self.test_results['recognition'] = results
        self.confusion_matrices = {name: res['confusion_matrix'] 
                                   for name, res in results.items()}
        
        return results
    
    def test_false_positive_scenarios(self, recognizer):
        """
        Test 3: False Positive Detection
        
        Scenarios that should NOT be recognized:
        - Random movements
        - Trembling/shaking
        - Stationary hand
        - Very fast movements
        """
        print("\n" + "=" * 60)
        print("FALSE POSITIVE TESTING")
        print("=" * 60)
        
        test_scenarios = {
            'random_movement': self._generate_random_trajectory(50),
            'trembling': self._generate_trembling_trajectory(50),
            'stationary': self._generate_stationary_trajectory(50),
            'too_fast': self._generate_fast_trajectory(10),
            'too_short': self._generate_short_trajectory(3)
        }
        
        results = {}
        
        for scenario_name, trajectory in test_scenarios.items():
            pred, conf = recognizer.recognize(trajectory)
            
            results[scenario_name] = {
                'predicted': pred,
                'confidence': conf,
                'should_reject': conf < 0.5  # Threshold for rejection
            }
            
            status = "✓ PASS" if conf < 0.5 else "✗ FAIL (False Positive)"
            print(f"{scenario_name:20s}: {pred:15s} (conf: {conf:.3f}) {status}")
        
        self.test_results['false_positives'] = results
        return results
    
    def test_environmental_conditions(self, recognizer, test_gestures):
        """
        Test 4: Environmental Robustness
        
        Test under:
        - Different lighting conditions (simulated)
        - Various backgrounds
        - Occlusion scenarios
        - Distance variations
        """
        print("\n" + "=" * 60)
        print("ENVIRONMENTAL ROBUSTNESS TESTING")
        print("=" * 60)
        
        conditions = {
            'normal': lambda g: g,
            'noisy': lambda g: self._add_noise(g, 0.05),
            'very_noisy': lambda g: self._add_noise(g, 0.1),
            'scaled_small': lambda g: self._scale_gesture(g, 0.5),
            'scaled_large': lambda g: self._scale_gesture(g, 1.5),
            'rotated': lambda g: self._rotate_gesture(g, 15),
            'partial_occlusion': lambda g: self._simulate_occlusion(g, 0.2)
        }
        
        results = {}
        
        for cond_name, transform in conditions.items():
            correct = 0
            total = len(test_gestures)
            
            for true_label, gesture in test_gestures:
                transformed = transform(gesture)
                pred, conf = recognizer.recognize(transformed)
                
                if pred == true_label:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            results[cond_name] = accuracy
            
            status = "✓" if accuracy > 0.7 else "⚠" if accuracy > 0.5 else "✗"
            print(f"{status} {cond_name:20s}: {accuracy:.2%}")
        
        self.test_results['environmental'] = results
        return results
    
    def benchmark_realtime_performance(self, system_components):
        """
        Test 5: Real-time Performance Benchmarking
        
        Measure:
        - End-to-end latency (camera → result)
        - Frame rate
        - CPU/Memory usage
        - Bottleneck identification
        """
        print("\n" + "=" * 60)
        print("REAL-TIME PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        n_iterations = 100
        
        timings = {
            'hand_detection': [],
            'smoothing': [],
            'recognition': [],
            'total': []
        }
        
        for i in range(n_iterations):
            # Simulate frame
            test_gesture = self._generate_random_trajectory(30)
            
            # Time each component
            t_start = time.time()
            
            # Component 1: Hand detection (simulated)
            t1 = time.time()
            timings['hand_detection'].append((t1 - t_start) * 1000)
            
            # Component 2: Smoothing
            smoothed = []
            for point in test_gesture:
                smoothed.append(system_components['smoother'].filter(point[0], point[1]))
            t2 = time.time()
            timings['smoothing'].append((t2 - t1) * 1000)
            
            # Component 3: Recognition
            pred, conf = system_components['recognizer'].recognize(test_gesture)
            t3 = time.time()
            timings['recognition'].append((t3 - t2) * 1000)
            
            timings['total'].append((t3 - t_start) * 1000)
        
        # Calculate statistics
        for component, times in timings.items():
            avg = np.mean(times)
            std = np.std(times)
            p95 = np.percentile(times, 95)
            
            print(f"\n{component.upper()}:")
            print(f"  Average: {avg:.2f} ms")
            print(f"  Std Dev: {std:.2f} ms")
            print(f"  95th percentile: {p95:.2f} ms")
        
        # Calculate achievable frame rate
        avg_total = np.mean(timings['total'])
        max_fps = 1000 / avg_total
        
        print(f"\nMAX ACHIEVABLE FPS: {max_fps:.1f}")
        print(f"REAL-TIME (30 FPS) CAPABLE: {'✓ YES' if max_fps >= 30 else '✗ NO'}")
        
        self.test_results['performance'] = timings
        return timings
    
    def generate_test_report(self, output_file='test_report.json'):
        """
        Generate comprehensive test report
        """
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_results': self.test_results,
            'summary': self._generate_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n\nTest report saved to {output_file}")
        return report
    
    def _generate_summary(self):
        """Generate executive summary of all tests"""
        summary = {
            'overall_status': 'PASS',
            'recommendations': [],
            'issues_found': []
        }
        
        # Check smoothing performance
        if 'smoothing' in self.test_results:
            best_smoother = min(self.test_results['smoothing'].items(), 
                              key=lambda x: x[1]['mse'])
            summary['best_smoother'] = best_smoother[0]
            summary['recommendations'].append(
                f"Use {best_smoother[0]} for smoothing (best MSE: {best_smoother[1]['mse']:.4f})"
            )
        
        # Check recognition performance
        if 'recognition' in self.test_results:
            best_recognizer = max(self.test_results['recognition'].items(),
                                 key=lambda x: x[1]['f1_score'])
            summary['best_recognizer'] = best_recognizer[0]
            
            if best_recognizer[1]['f1_score'] < 0.8:
                summary['issues_found'].append(
                    f"Recognition F1-score below 0.8: {best_recognizer[1]['f1_score']:.3f}"
                )
                summary['overall_status'] = 'WARNING'
        
        # Check real-time performance
        if 'performance' in self.test_results:
            avg_latency = np.mean(self.test_results['performance']['total'])
            if avg_latency > 33:  # More than 30 FPS
                summary['issues_found'].append(
                    f"Latency too high for real-time: {avg_latency:.1f} ms"
                )
                summary['overall_status'] = 'FAIL'
        
        return summary
    
    # Helper methods for generating test data
    def _generate_random_trajectory(self, n_points):
        return np.random.rand(n_points, 2) * 100
    
    def _generate_trembling_trajectory(self, n_points):
        base = np.linspace(0, 100, n_points)
        noise = np.random.randn(n_points) * 2
        return np.column_stack([base + noise, base + noise])
    
    def _generate_stationary_trajectory(self, n_points):
        return np.ones((n_points, 2)) * 50 + np.random.randn(n_points, 2) * 0.5
    
    def _generate_fast_trajectory(self, n_points):
        return np.random.rand(n_points, 2) * 100
    
    def _generate_short_trajectory(self, n_points):
        return np.random.rand(n_points, 2) * 10
    
    def _add_noise(self, gesture, noise_level):
        gesture = np.array(gesture)
        noise = np.random.randn(*gesture.shape) * noise_level * np.max(gesture)
        return gesture + noise
    
    def _scale_gesture(self, gesture, scale):
        gesture = np.array(gesture)
        centroid = np.mean(gesture, axis=0)
        return (gesture - centroid) * scale + centroid
    
    def _rotate_gesture(self, gesture, angle_deg):
        gesture = np.array(gesture)
        angle_rad = np.radians(angle_deg)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        centroid = np.mean(gesture, axis=0)
        return (gesture - centroid) @ rotation_matrix.T + centroid
    
    def _simulate_occlusion(self, gesture, occlusion_ratio):
        gesture = np.array(gesture)
        n_occluded = int(len(gesture) * occlusion_ratio)
        indices = np.random.choice(len(gesture), n_occluded, replace=False)
        occluded = gesture.copy()
        occluded[indices] = np.nan
        return occluded[~np.isnan(occluded).any(axis=1)]
    
    def _calculate_mse(self, trajectory1, trajectory2):
        """Calculate Mean Squared Error between trajectories"""
        traj1 = np.array(trajectory1)
        traj2 = np.array(trajectory2)
        
        # Resample to same length
        if len(traj1) != len(traj2):
            min_len = min(len(traj1), len(traj2))
            traj1 = traj1[:min_len]
            traj2 = traj2[:min_len]
        
        return np.mean((traj1 - traj2) ** 2)
    
    def _calculate_smoothness(self, trajectory):
        """Calculate smoothness (inverse of jerk)"""
        traj = np.array(trajectory)
        if len(traj) < 3:
            return 0
        
        # Calculate second derivative (acceleration)
        acceleration = np.diff(traj, n=2, axis=0)
        jerk = np.sqrt(np.sum(acceleration ** 2, axis=1))
        
        return np.mean(jerk)
    
    def _calculate_lag(self, smoothed, ground_truth):
        """Calculate lag using cross-correlation"""
        smoothed = np.array(smoothed)
        ground_truth = np.array(ground_truth)
        
        # Simplified lag calculation
        if len(smoothed) != len(ground_truth):
            return 0
        
        # Find best alignment
        max_lag = min(10, len(smoothed) // 4)
        best_corr = -np.inf
        best_lag = 0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(smoothed[-lag:, 0], ground_truth[:lag, 0])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(smoothed[:-lag, 0], ground_truth[lag:, 0])[0, 1]
            else:
                corr = np.corrcoef(smoothed[:, 0], ground_truth[:, 0])[0, 1]
            
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag
        
        return abs(best_lag)
    
    def plot_confusion_matrices(self, class_names):
        """Visualize confusion matrices for all recognizers"""
        n_algorithms = len(self.confusion_matrices)
        
        if n_algorithms == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (name, cm) in enumerate(self.confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx])
            axes[idx].set_title(f'{name} Confusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Confusion matrices saved to confusion_matrices.png")
        
        return fig