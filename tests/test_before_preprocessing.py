"""Focused invariants for the isolated before-preprocessing experiment."""

import unittest

import numpy as np
import torch

from experiments.before_preprocessing.run_experiment import (
    RawData,
    calculate_metrics,
    fit_train_medians,
    graph_prefix,
    impute_only,
)


class BeforePreprocessingTests(unittest.TestCase):
    def test_imputer_fits_training_timesteps_only_and_does_not_scale(self):
        raw = np.array([[2.0, np.nan], [4.0, 8.0], [9000.0, 7000.0]], dtype=np.float32)
        times = np.array([1, 2, 35], dtype=np.int16)
        medians = fit_train_medians(raw, times, through=2)
        self.assertTrue(np.allclose(medians, [3.0, 8.0]))
        transformed = impute_only(raw, medians)
        self.assertTrue(np.allclose(transformed[0], [2.0, 8.0]))
        self.assertEqual(float(transformed[2, 0]), 9000.0)

    def test_graph_prefix_excludes_future_endpoint(self):
        data = RawData(
            ids=np.array([10, 20, 30]),
            times=np.array([1, 2, 3], dtype=np.int16),
            labels=np.array([0, 1, -1], dtype=np.int8),
            raw_features=np.ones((3, 1), dtype=np.float32),
            feature_names=["feature"],
            source_edges=np.array([0, 1]),
            target_edges=np.array([1, 2]),
        )
        nodes, adjacency = graph_prefix(data, cutoff=2, device=torch.device("cpu"))
        self.assertTrue(np.array_equal(nodes, [0, 1]))
        self.assertEqual(tuple(adjacency.shape), (2, 2))
        self.assertEqual(int(adjacency._indices().max()), 1)

    def test_classification_metrics_are_probability_based_and_complete(self):
        y = np.array([0, 0, 1, 1], dtype=np.int8)
        probability = np.array([0.1, 0.7, 0.4, 0.9], dtype=np.float32)
        metrics, confusion, prediction = calculate_metrics(y, probability)
        required = {"accuracy", "precision", "recall", "specificity", "f1", "pr_auc", "roc_auc", "fpr", "fnr", "error_rate"}
        self.assertTrue(required.issubset(metrics))
        self.assertEqual(confusion.tolist(), [[1, 1], [1, 1]])
        self.assertTrue(np.array_equal(prediction, [0, 1, 0, 1]))
        self.assertTrue(all(np.isfinite(metrics[key]) for key in required))


if __name__ == "__main__":
    unittest.main()
