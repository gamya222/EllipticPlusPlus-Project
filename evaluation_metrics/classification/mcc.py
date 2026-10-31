"""Matthews Correlation Coefficient from the binary confusion counts."""

import math

from evaluation_metrics._validation import UndefinedMetricError, confusion_counts


def calculate_mcc(y_true, y_pred) -> float:
    """Calculate MCC; undefined when any confusion-matrix marginal is zero."""
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        raise UndefinedMetricError("MCC is undefined when a confusion-matrix marginal is zero")
    return ((tp * tn) - (fp * fn)) / denominator
