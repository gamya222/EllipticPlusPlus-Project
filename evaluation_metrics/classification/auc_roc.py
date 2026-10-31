"""ROC AUC using the Mann–Whitney rank statistic with average tie ranks."""

import numpy as np

from evaluation_metrics._validation import UndefinedMetricError, label_probability_pair


def calculate_auc_roc(y_true, y_probability) -> float:
    """Calculate ROC AUC from positive-class probabilities, not hard labels."""
    actual, probability = label_probability_pair(y_true, y_probability)
    positives = int(np.sum(actual == 1))
    negatives = int(np.sum(actual == 0))
    if positives == 0 or negatives == 0:
        raise UndefinedMetricError("AUC-ROC is undefined when y_true contains only one class")
    order = np.argsort(probability, kind="mergesort")
    sorted_scores = probability[order]
    ranks = np.empty(actual.size, dtype=np.float64)
    start = 0
    while start < actual.size:
        end = start + 1
        while end < actual.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    positive_rank_sum = float(np.sum(ranks[actual == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
