"""Specificity / true negative rate: TN / (TN + FP)."""

from evaluation_metrics._validation import UndefinedMetricError, confusion_counts


def calculate_specificity(y_true, y_pred) -> float:
    """Calculate true negative rate; undefined when y_true has no negatives."""
    tn, fp, _, _ = confusion_counts(y_true, y_pred)
    denominator = tn + fp
    if denominator == 0:
        raise UndefinedMetricError("Specificity is undefined when y_true has no negative class")
    return tn / denominator
