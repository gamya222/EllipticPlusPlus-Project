"""Mean Absolute Error: mean(abs(y_true - y_pred))."""

import numpy as np

from evaluation_metrics._validation import numeric_pair


def calculate_mae(y_true, y_pred) -> float:
    """Calculate MAE for equal-length finite one-dimensional arrays."""
    actual, predicted = numeric_pair(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))
