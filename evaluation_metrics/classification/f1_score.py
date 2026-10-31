"""F1 score: 2 TP / (2 TP + FP + FN)."""

from evaluation_metrics._validation import UndefinedMetricError, confusion_counts


def calculate_f1_score(y_true, y_pred) -> float:
    """Calculate positive-class F1; undefined when its denominator is zero."""
    _, fp, fn, tp = confusion_counts(y_true, y_pred)
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        raise UndefinedMetricError("F1 is undefined when there are no actual or predicted positives")
    return (2 * tp) / denominator
