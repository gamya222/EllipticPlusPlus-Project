"""Independent tests for every regression-style metric."""

import unittest

import numpy as np

from evaluation_metrics._validation import MetricInputError, UndefinedMetricError
from evaluation_metrics.regression import (
    calculate_bias, calculate_mae, calculate_mape, calculate_medae,
    calculate_mse, calculate_nrmse, calculate_r2, calculate_rmse,
)


class RegressionMetricTests(unittest.TestCase):
    def setUp(self):
        self.actual = np.array([1.0, 2.0, 3.0])
        self.predicted = np.array([1.0, 2.0, 2.0])
        self.metrics = (
            calculate_rmse, calculate_mse, calculate_mae, calculate_mape,
            calculate_r2, calculate_nrmse, calculate_medae, calculate_bias,
        )

    def test_expected_values(self):
        self.assertAlmostEqual(calculate_rmse(self.actual, self.predicted), np.sqrt(1 / 3))
        self.assertAlmostEqual(calculate_mse(self.actual, self.predicted), 1 / 3)
        self.assertAlmostEqual(calculate_mae(self.actual, self.predicted), 1 / 3)
        self.assertAlmostEqual(calculate_mape(self.actual, self.predicted), 100 / 9)
        self.assertAlmostEqual(calculate_r2(self.actual, self.predicted), 0.5)
        self.assertAlmostEqual(calculate_nrmse(self.actual, self.predicted), np.sqrt(1 / 3) / 2)
        self.assertAlmostEqual(calculate_medae(self.actual, self.predicted), 0.0)
        self.assertAlmostEqual(calculate_bias(self.actual, self.predicted), -1 / 3)

    def test_input_validation_for_all_metrics(self):
        for metric in self.metrics:
            with self.subTest(metric=metric.__name__, case="empty"):
                with self.assertRaises(MetricInputError):
                    metric([], [])
            with self.subTest(metric=metric.__name__, case="mismatch"):
                with self.assertRaises(MetricInputError):
                    metric([1, 2], [1])
            with self.subTest(metric=metric.__name__, case="dimension"):
                with self.assertRaises(MetricInputError):
                    metric([[1, 2]], [[1, 2]])
            with self.subTest(metric=metric.__name__, case="nan"):
                with self.assertRaises(MetricInputError):
                    metric([1, np.nan], [1, 2])
            with self.subTest(metric=metric.__name__, case="infinity"):
                with self.assertRaises(MetricInputError):
                    metric([1, np.inf], [1, 2])

    def test_undefined_denominators(self):
        with self.assertRaisesRegex(UndefinedMetricError, "zero"):
            calculate_mape([0, 1], [0.1, 0.9])
        with self.assertRaisesRegex(UndefinedMetricError, "constant"):
            calculate_r2([1, 1], [0.8, 1.0])
        with self.assertRaisesRegex(UndefinedMetricError, "zero target range"):
            calculate_nrmse([1, 1], [0.8, 1.0])


if __name__ == "__main__":
    unittest.main()
