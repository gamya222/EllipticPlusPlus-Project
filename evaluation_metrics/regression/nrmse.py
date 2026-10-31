"""Range-normalized RMSE: RMSE / (max(y_true) - min(y_true))."""

import numpy as np

from evaluation_metrics._validation import UndefinedMetricError, numeric_pair
from .rmse import calculate_rmse


def calculate_nrmse(y_true, y_pred) -> float:
    """Calculate NRMSE using the target range as its documented scale."""
    actual, predicted = numeric_pair(y_true, y_pred)
    target_range = float(np.max(actual) - np.min(actual))
    if target_range == 0.0:
        raise UndefinedMetricError("Range-normalized RMSE is undefined for zero target range")
    return calculate_rmse(actual, predicted) / target_range
