from testing_framework import PerformanceEvaluator
from smoothing_algorithms import *
import numpy as np

evaluator = PerformanceEvaluator()

# Generate test data
t = np.linspace(0, 2*np.pi, 100)
ground_truth = [(np.cos(ti)*100+200, np.sin(ti)*100+200) for ti in t]
noisy = [(x+np.random.randn()*10, y+np.random.randn()*10) for x, y in ground_truth]

# Test smoothers
smoothers = {
    'one_euro': OneEuroFilter(),
    'kalman': KalmanFilter(),
    'ema': ExponentialMovingAverage()
}

results = evaluator.evaluate_smoothing_algorithms(noisy, ground_truth, smoothers)
evaluator.generate_test_report('results/test_report.json')
print("Test complete! Check results/test_report.json")