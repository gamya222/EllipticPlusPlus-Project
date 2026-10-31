"""Median Absolute Error: median(abs(y_true - y_pred))."""

import numpy as np

from evaluation_metrics._validation import numeric_pair


def calculate_medae(y_true, y_pred) -> float:
    """Calculate MedAE for equal-length finite one-dimensional arrays."""
    actual, predicted = numeric_pair(y_true, y_pred)
    return float(np.median(np.abs(actual - predicted)))
