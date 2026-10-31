"""Shared input validation used by independently callable metric functions."""

from __future__ import annotations

import numpy as np


class MetricInputError(ValueError):
    """Raised when metric inputs do not satisfy the documented contract."""


class UndefinedMetricError(ValueError):
    """Raised when a metric has no finite mathematical definition for inputs."""


def numeric_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, nonempty, equal-length one-dimensional float arrays."""
    try:
        actual = np.asarray(y_true, dtype=np.float64)
        predicted = np.asarray(y_pred, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MetricInputError("Inputs must contain numeric values") from exc
    if actual.ndim != 1 or predicted.ndim != 1:
        raise MetricInputError("Inputs must be one-dimensional arrays")
    if actual.size == 0 or predicted.size == 0:
        raise MetricInputError("Inputs must not be empty")
    if actual.size != predicted.size:
        raise MetricInputError("y_true and y_pred must have matching lengths")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise MetricInputError("Inputs must not contain NaN or infinite values")
    return actual, predicted


def binary_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-length binary label/class arrays encoded as 0 and 1."""
    actual, predicted = numeric_pair(y_true, y_pred)
    if not np.isin(actual, (0.0, 1.0)).all():
        raise MetricInputError("y_true must contain only binary labels 0 and 1")
    if not np.isin(predicted, (0.0, 1.0)).all():
        raise MetricInputError("y_pred must contain only binary classes 0 and 1")
    return actual.astype(np.int8), predicted.astype(np.int8)


def label_probability_pair(y_true, y_probability) -> tuple[np.ndarray, np.ndarray]:
    """Return binary labels and finite positive-class probabilities in [0, 1]."""
    try:
        actual = np.asarray(y_true, dtype=np.float64)
        probability = np.asarray(y_probability, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MetricInputError("Labels and probabilities must be numeric") from exc
    if actual.ndim != 1 or probability.ndim != 1:
        raise MetricInputError("Labels and probabilities must be one-dimensional")
    if actual.size == 0 or probability.size == 0:
        raise MetricInputError("Labels and probabilities must not be empty")
    if actual.size != probability.size:
        raise MetricInputError("Labels and probabilities must have matching lengths")
    if not np.isfinite(actual).all() or not np.isfinite(probability).all():
        raise MetricInputError("Inputs must not contain NaN or infinite values")
    if not np.isin(actual, (0.0, 1.0)).all():
        raise MetricInputError("y_true must contain only binary labels 0 and 1")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise MetricInputError("Positive-class probabilities must be in [0, 1]")
    return actual.astype(np.int8), probability


def confusion_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    """Return (tn, fp, fn, tp) after validating binary arrays."""
    actual, predicted = binary_pair(y_true, y_pred)
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    return tn, fp, fn, tp
