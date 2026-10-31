"""Mean Error / bias: mean(y_pred - y_true)."""

import numpy as np

from evaluation_metrics._validation import numeric_pair


def calculate_bias(y_true, y_pred) -> float:
    """Calculate signed bias; positive values mean overprediction."""
    actual, predicted = numeric_pair(y_true, y_pred)
    return float(np.mean(predicted - actual))
