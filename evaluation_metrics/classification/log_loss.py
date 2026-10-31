"""Binary cross entropy with numerically clipped positive-class probabilities."""

import numpy as np

from evaluation_metrics._validation import label_probability_pair


def calculate_log_loss(y_true, y_probability) -> float:
    """Calculate mean binary log loss; p is clipped to [1e-15, 1-1e-15]."""
    actual, probability = label_probability_pair(y_true, y_probability)
    epsilon = 1e-15
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    return float(-np.mean(actual * np.log(clipped) + (1 - actual) * np.log1p(-clipped)))
