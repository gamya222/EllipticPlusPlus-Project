"""Small deterministic checks for preprocessing's fitted missing-value transform."""

from __future__ import annotations

import unittest

import numpy as np

from models.gcn.data import FeatureTransform


class FeatureTransformInvariantTests(unittest.TestCase):
    def test_imputation_and_scaling_fit_only_through_training_cutoff(self):
        raw = np.array(
            [[1.0, 10.0], [3.0, 30.0], [np.nan, 50.0], [10_000.0, 1_000_000.0]],
            dtype=np.float64,
        )
        times = np.array([1, 2, 2, 3], dtype=np.int16)
        transform = FeatureTransform.fit(raw, times, fit_through=2)

        np.testing.assert_allclose(transform.medians, [2.0, 30.0])
        np.testing.assert_allclose(transform.centers, [2.0, 30.0])
        self.assertLess(transform.centers[1], 1_000_000.0)

        transformed = transform.transform(raw)
        self.assertTrue(np.isfinite(transformed).all())
        # The missing training value is median-imputed; after centering it is 0.
        self.assertEqual(float(transformed[2, 0]), 0.0)
        # The held-out extreme is transformed with saved training parameters.
        self.assertGreater(float(transformed[3, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
