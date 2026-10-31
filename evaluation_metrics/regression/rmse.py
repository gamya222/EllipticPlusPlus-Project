"""Root Mean Square Error: sqrt(mean((y_true - y_pred)^2))."""

import numpy as np

from evaluation_metrics._validation import numeric_pair


def calculate_rmse(y_true, y_pred) -> float:
    """Calculate RMSE for equal-length finite one-dimensional arrays."""
    actual, predicted = numeric_pair(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))
