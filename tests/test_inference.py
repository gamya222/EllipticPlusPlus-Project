"""Focused contract tests for the reusable batch inference interface."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from models.inference import EllipticInference, SUPPORTED_MODELS


class InferenceInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = EllipticInference(device="cpu")

    def test_loads_all_saved_model_and_risk_artifacts(self):
        self.assertEqual(set(self.scorer.models), set(SUPPORTED_MODELS))
        self.assertEqual(self.scorer.risk_artifact["artifact_version"], 1)
        for transform in [*self.scorer.transforms.values(), self.scorer.risk_transform]:
            self.assertEqual(transform.medians.shape, (182,))
            self.assertTrue((transform.scales > 0).all())
        for name, transform in self.scorer.transforms.items():
            checkpoint = self.scorer.checkpoints[name]
            self.assertTrue((transform.medians == checkpoint["feature_medians"]).all())
            self.assertTrue((transform.centers == checkpoint["feature_centers"]).all())
            self.assertTrue((transform.scales == checkpoint["feature_scales"]).all())
        self.assertTrue(
            (self.scorer.risk_transform.medians == self.scorer.risk_artifact["feature_medians"]).all()
        )

    def test_unknown_and_known_but_unsupported_ids_are_explicit(self):
        training_row = next(
            i for i, timestep in enumerate(self.scorer.data.times) if timestep <= 34
        )
        unknown, training = self.scorer.score_batch(
            [-1, int(self.scorer.data.ids[training_row])]
        )
        self.assertEqual(unknown["status"], "unknown_transaction_id")
        self.assertIsNone(unknown["model_output"]["score"])
        self.assertIsNone(unknown["risk"]["score"])
        self.assertEqual(training["status"], "unsupported_timestep")
        self.assertIsNone(training["graph_statistics"])

    def test_known_test_batch_schema_and_determinism(self):
        index = next(i for i, timestep in enumerate(self.scorer.data.times) if timestep == 35)
        tx_id = int(self.scorer.data.ids[index])
        first = self.scorer.score_batch([tx_id], model="logistic")[0]
        second = self.scorer.score_batch([tx_id], model="logistic")[0]
        expected = {
            "txId", "Time step", "status", "requested_model", "model_output", "risk",
            "graph_statistics", "community", "suspicious_connection",
        }
        self.assertEqual(set(first), expected)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "scored")
        self.assertIs(type(first["txId"]), int)
        self.assertIs(type(first["Time step"]), int)
        self.assertIn(first["status"], {"scored", "unknown_transaction_id", "unsupported_timestep"})
        self.assertIs(type(first["model_output"]["score"]), float)
        self.assertGreaterEqual(first["model_output"]["score"], 0.0)
        self.assertLessEqual(first["model_output"]["score"], 1.0)
        self.assertIs(type(first["risk"]["score"]), float)
        self.assertGreaterEqual(first["risk"]["score"], 0.0)
        self.assertLessEqual(first["risk"]["score"], 1.0)
        self.assertIs(type(first["graph_statistics"]["in_degree"]), int)
        self.assertIs(type(first["community"]["id"]), int)
        self.assertIs(type(first["suspicious_connection"]["rank"]), int)
        self.assertEqual(first["graph_statistics"]["scope"], "full_graph_descriptive_includes_later_timesteps")
        self.assertEqual(first["suspicious_connection"]["scope"], "as_of_transaction_timestep")

    def test_gcn_scores_a_small_known_batch(self):
        indices = [i for i, timestep in enumerate(self.scorer.data.times) if timestep == 35][:2]
        ids = [int(self.scorer.data.ids[i]) for i in indices]
        results = self.scorer.score_batch(ids, model="gcn")
        self.assertEqual([item["status"] for item in results], ["scored", "scored"])
        self.assertTrue(all(0.0 <= item["model_output"]["score"] <= 1.0 for item in results))

    def test_graphsage_and_gru_score_a_known_transaction(self):
        index = next(i for i, timestep in enumerate(self.scorer.data.times) if timestep == 35)
        tx_id = int(self.scorer.data.ids[index])
        for model in ("graphsage", "gru"):
            with self.subTest(model=model):
                result = self.scorer.score_batch([tx_id], model=model)[0]
                self.assertEqual(result["status"], "scored")
                self.assertTrue(0.0 <= result["model_output"]["score"] <= 1.0)
                self.assertTrue(0.0 <= result["risk"]["score"] <= 1.0)

    def test_missing_analytics_records_remain_null(self):
        index = next(i for i, timestep in enumerate(self.scorer.data.times) if timestep == 35)
        tx_id = int(self.scorer.data.ids[index])
        stats = self.scorer.graph_stats.pop(tx_id, None)
        suspicious = self.scorer.suspicious.pop(tx_id, None)
        try:
            result = self.scorer.score_batch([tx_id], model="logistic")[0]
        finally:
            if stats is not None:
                self.scorer.graph_stats[tx_id] = stats
            if suspicious is not None:
                self.scorer.suspicious[tx_id] = suspicious
        self.assertIsNone(result["graph_statistics"])
        self.assertIsNone(result["community"])
        self.assertIsNone(result["suspicious_connection"])

    def test_versioned_batch_document_writes_without_overwriting(self):
        index = next(i for i, timestep in enumerate(self.scorer.data.times) if timestep == 35)
        tx_id = int(self.scorer.data.ids[index])
        document = self.scorer.batch_document([tx_id, -1], model="logistic")
        self.assertEqual(document["schema_version"], "elliptic-inference-v1")
        self.assertEqual(document["record_count"], 2)
        self.assertEqual(document["status_counts"], {
            "scored": 1, "unknown_transaction_id": 1, "unsupported_timestep": 0,
        })
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "batch.json"
            self.scorer.write_batch_document(document, output)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved, document)
            with self.assertRaises(FileExistsError):
                self.scorer.write_batch_document(document, output)


if __name__ == "__main__":
    unittest.main()
