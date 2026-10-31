"""Mean Absolute Percentage Error; undefined when any target is zero."""

import numpy as np

from evaluation_metrics._validation import UndefinedMetricError, numeric_pair


def calculate_mape(y_true, y_pred) -> float:
    """Calculate percentage MAPE; reject zero targets rather than omit them."""
    actual, predicted = numeric_pair(y_true, y_pred)
    if np.any(actual == 0):
        raise UndefinedMetricError("MAPE is undefined when any y_true value is zero")
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)
