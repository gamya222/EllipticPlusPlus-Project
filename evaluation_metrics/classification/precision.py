"""Positive predictive value: TP / (TP + FP)."""

from evaluation_metrics._validation import UndefinedMetricError, confusion_counts


def calculate_precision(y_true, y_pred) -> float:
    """Calculate positive-class precision; undefined if no positives are predicted."""
    _, fp, _, tp = confusion_counts(y_true, y_pred)
    denominator = tp + fp
    if denominator == 0:
        raise UndefinedMetricError("Precision is undefined when no positive classes are predicted")
    return tp / denominator
