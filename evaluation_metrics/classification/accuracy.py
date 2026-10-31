"""Accuracy: (TP + TN) / (TP + TN + FP + FN)."""

from evaluation_metrics._validation import binary_pair


def calculate_accuracy(y_true, y_pred) -> float:
    """Return the fraction of correctly classified binary examples."""
    actual, predicted = binary_pair(y_true, y_pred)
    return float((actual == predicted).mean())
