"""Balanced accuracy: mean of positive-class recall and negative-class recall."""

from evaluation_metrics._validation import UndefinedMetricError, confusion_counts


def calculate_balanced_accuracy(y_true, y_pred) -> float:
    """Calculate balanced accuracy; undefined unless both classes are present."""
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    positive_total = tp + fn
    negative_total = tn + fp
    if positive_total == 0 or negative_total == 0:
        raise UndefinedMetricError("Balanced accuracy requires both true classes")
    return 0.5 * (tp / positive_total + tn / negative_total)
