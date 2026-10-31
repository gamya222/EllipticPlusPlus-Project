"""Cohen's kappa: (observed agreement - expected agreement)/(1-expected)."""

import numpy as np

from evaluation_metrics._validation import UndefinedMetricError, binary_pair


def calculate_cohens_kappa(y_true, y_pred) -> float:
    """Calculate binary Cohen's kappa; undefined if expected agreement is 1."""
    actual, predicted = binary_pair(y_true, y_pred)
    observed = float(np.mean(actual == predicted))
    true_positive_rate = float(np.mean(actual == 1))
    predicted_positive_rate = float(np.mean(predicted == 1))
    expected = (true_positive_rate * predicted_positive_rate) + ((1 - true_positive_rate) * (1 - predicted_positive_rate))
    if expected == 1.0:
        raise UndefinedMetricError("Cohen's kappa is undefined when expected agreement is 1")
    return (observed - expected) / (1.0 - expected)
