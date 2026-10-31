"""Recall / sensitivity: TP / (TP + FN)."""

from evaluation_metrics._validation import UndefinedMetricError, confusion_counts


def calculate_recall(y_true, y_pred) -> float:
    """Calculate positive-class recall; undefined when no positives are present."""
    _, _, fn, tp = confusion_counts(y_true, y_pred)
    denominator = tp + fn
    if denominator == 0:
        raise UndefinedMetricError("Recall is undefined when y_true has no positive class")
    return tp / denominator
