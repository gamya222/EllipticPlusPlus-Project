"""Coefficient of determination: 1 - SSE/SST."""

import numpy as np

from evaluation_metrics._validation import UndefinedMetricError, numeric_pair


def calculate_r2(y_true, y_pred) -> float:
    """Calculate R²; it is undefined when the target is constant."""
    actual, predicted = numeric_pair(y_true, y_pred)
    total = float(np.sum(np.square(actual - np.mean(actual))))
    if total == 0.0:
        raise UndefinedMetricError("R² is undefined for constant y_true")
    return float(1.0 - np.sum(np.square(actual - predicted)) / total)
