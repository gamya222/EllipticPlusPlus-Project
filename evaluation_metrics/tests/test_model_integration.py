"""Tests ID/timestep alignment and label mapping without synthetic evaluation outputs."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation_metrics.calculate_all import align_prediction_records, evaluate_arrays
from evaluation_metrics.definitions import METRIC_DEFINITIONS, definitions_as_json, export_definitions


class PredictionAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.truth = [
            {"txId": 12, "Time step": 35, "class": 2},
            {"txId": 10, "Time step": 35, "class": 1},
            {"txId": 11, "Time step": 35, "class": 3},
        ]
        self.predictions = [
            {"txId": 11, "Time step": 35, "status": "scored", "model_output": {"score": 0.4}},
            {"txId": 10, "Time step": 35, "status": "scored", "model_output": {"score": 0.8}},
            {"txId": 12, "Time step": 35, "status": "scored", "model_output": {"score": 0.2}},
        ]

    def test_aligns_by_id_and_maps_only_labeled_classes(self):
        aligned = align_prediction_records(self.predictions, self.truth)
        self.assertEqual([(record["txId"], label) for record, label in aligned], [(12, 0), (10, 1), (11, None)])

    def test_rejects_duplicate_ids_and_missing_predictions(self):
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            align_prediction_records(self.predictions + [self.predictions[0]], self.truth)
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            align_prediction_records(self.predictions[:-1], self.truth)

    def test_rejects_timestep_mismatch_and_unscored_record(self):
        mismatched = [dict(row) for row in self.predictions]
        mismatched[0] = {**mismatched[0], "Time step": 36}
        with self.assertRaisesRegex(ValueError, "Timestep mismatch"):
            align_prediction_records(mismatched, self.truth)
        unscored = [dict(row) for row in self.predictions]
        unscored[0] = {**unscored[0], "status": "unknown_transaction_id"}
        with self.assertRaisesRegex(ValueError, "was not scored"):
            align_prediction_records(unscored, self.truth)

    def test_evaluation_statuses_and_all_definitions_are_machine_readable(self):
        regression, classification = evaluate_arrays([0, 1, 0, 1], [0.1, 0.9, 0.7, 0.8], [0, 1, 1, 1])
        self.assertEqual(regression["mape"]["status"], "undefined")
        self.assertTrue(all(result["status"] == "calculated" for name, result in classification.items() if name != "mape"))
        self.assertEqual(len(METRIC_DEFINITIONS), 18)
        self.assertEqual(set(json.loads(definitions_as_json())), set(METRIC_DEFINITIONS))
        with tempfile.TemporaryDirectory() as directory:
            output = export_definitions(Path(directory) / "definitions.json")
            self.assertEqual(set(json.loads(output.read_text())), set(METRIC_DEFINITIONS))


if __name__ == "__main__":
    unittest.main()
