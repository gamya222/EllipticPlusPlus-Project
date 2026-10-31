"""Independent tests for every binary classification metric."""

import unittest

import numpy as np

from evaluation_metrics._validation import MetricInputError, UndefinedMetricError
from evaluation_metrics.classification import (
    calculate_accuracy, calculate_auc_roc, calculate_balanced_accuracy,
    calculate_cohens_kappa, calculate_f1_score, calculate_log_loss,
    calculate_mcc, calculate_precision, calculate_recall, calculate_specificity,
)


class ClassificationMetricTests(unittest.TestCase):
    def setUp(self):
        self.actual = np.array([0, 0, 1, 1])
        self.predicted = np.array([0, 1, 1, 1])
        self.probability = np.array([0.1, 0.8, 0.8, 0.9])
        self.class_metrics = (
            calculate_accuracy, calculate_precision, calculate_recall,
            calculate_f1_score, calculate_specificity, calculate_mcc,
            calculate_cohens_kappa, calculate_balanced_accuracy,
        )
        self.probability_metrics = (calculate_auc_roc, calculate_log_loss)

    def test_expected_values(self):
        self.assertEqual(calculate_accuracy(self.actual, self.predicted), 0.75)
        self.assertAlmostEqual(calculate_precision(self.actual, self.predicted), 2 / 3)
        self.assertEqual(calculate_recall(self.actual, self.predicted), 1.0)
        self.assertEqual(calculate_f1_score(self.actual, self.predicted), 0.8)
        self.assertEqual(calculate_specificity(self.actual, self.predicted), 0.5)
        self.assertAlmostEqual(calculate_auc_roc(self.actual, self.probability), 0.875)
        self.assertAlmostEqual(calculate_log_loss(self.actual, self.probability), -np.mean(np.log([0.9, 0.2, 0.8, 0.9])))
        self.assertAlmostEqual(calculate_mcc(self.actual, self.predicted), 1 / np.sqrt(3))
        self.assertEqual(calculate_cohens_kappa(self.actual, self.predicted), 0.5)
        self.assertEqual(calculate_balanced_accuracy(self.actual, self.predicted), 0.75)

    def test_hard_class_input_validation(self):
        for metric in self.class_metrics:
            for actual, predicted in (([], []), ([0, 1], [0]), ([[0, 1]], [[0, 1]]), ([0, np.nan], [0, 1]), ([0, 1], [0, np.inf])):
                with self.subTest(metric=metric.__name__, actual=actual):
                    with self.assertRaises(MetricInputError):
                        metric(actual, predicted)

    def test_probability_validation(self):
        for metric in self.probability_metrics:
            for actual, probability in (([], []), ([0, 1], [0.2]), ([[0, 1]], [[0.2, 0.8]]), ([0, np.nan], [0.2, 0.8]), ([0, 1], [-0.1, 0.8]), ([0, 1], [0.2, 1.1]), ([0, 1], [0.2, np.inf])):
                with self.subTest(metric=metric.__name__, probability=probability):
                    with self.assertRaises(MetricInputError):
                        metric(actual, probability)

    def test_single_class_and_zero_denominator_cases(self):
        with self.assertRaises(UndefinedMetricError):
            calculate_auc_roc([0, 0], [0.1, 0.2])
        with self.assertRaises(UndefinedMetricError):
            calculate_recall([0, 0], [0, 1])
        with self.assertRaises(UndefinedMetricError):
            calculate_specificity([1, 1], [0, 1])
        with self.assertRaises(UndefinedMetricError):
            calculate_balanced_accuracy([1, 1], [0, 1])
        with self.assertRaises(UndefinedMetricError):
            calculate_precision([0, 1], [0, 0])
        with self.assertRaises(UndefinedMetricError):
            calculate_f1_score([0, 0], [0, 0])
        with self.assertRaises(UndefinedMetricError):
            calculate_mcc([0, 1], [0, 0])
        with self.assertRaises(UndefinedMetricError):
            calculate_cohens_kappa([1, 1], [1, 1])

    def test_log_loss_handles_exact_zero_and_one_probabilities(self):
        value = calculate_log_loss([0, 1], [0.0, 1.0])
        self.assertTrue(np.isfinite(value))
        self.assertLess(value, 1e-12)

    def test_imbalance_does_not_hide_recall(self):
        actual = [0] * 99 + [1]
        predicted = [0] * 99 + [0]
        self.assertAlmostEqual(calculate_accuracy(actual, predicted), 0.99)
        self.assertEqual(calculate_recall(actual, predicted), 0.0)
        self.assertAlmostEqual(calculate_balanced_accuracy(actual, predicted), 0.5)


if __name__ == "__main__":
    unittest.main()
