"""Focused tests for the dashboard's read-only evaluation artifact endpoints."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from dashboard.server import DashboardHandler


class DashboardEvaluationApiTests(unittest.TestCase):
    def setUp(self):
        self.handler = DashboardHandler.__new__(DashboardHandler)
        self.json_response = None
        self.file_response = None
        self.error_response = None
        self.handler._send_json = lambda data, status=200: setattr(
            self, "json_response", (data, status)
        )
        self.handler._send_file = lambda path, content_type: setattr(
            self, "file_response", (Path(path), content_type)
        )
        self.handler.send_error = lambda status, message: setattr(
            self, "error_response", (status, message)
        )

    def test_evaluation_endpoint_returns_csv_metrics_and_explicit_undefined(self):
        self.handler._handle_api_evaluation()
        data, status = self.json_response

        self.assertEqual(status, 200)
        self.assertEqual(data["schema_version"], "elliptic-evaluation-api-v1")
        self.assertEqual(data["status"], "available")
        self.assertEqual(len(data["models"]), 5)
        self.assertEqual(len(data["groups"]["classification"]), 10)
        self.assertEqual(len(data["groups"]["regression"]), 8)

        mape = next(row for row in data["groups"]["regression"] if row["metric"] == "mape")
        self.assertEqual(mape["results"]["gcn"]["status"], "undefined")
        self.assertIsNone(mape["results"]["gcn"]["value"])

        accuracy = next(row for row in data["groups"]["classification"] if row["metric"] == "accuracy")
        self.assertEqual(accuracy["results"]["gcn"]["status"], "calculated")
        self.assertAlmostEqual(accuracy["results"]["gcn"]["value"], 0.9092981403719256)
        self.assertEqual(accuracy["results"]["gcn"]["sample_count"], 16670)

    def test_graph_manifest_lists_only_existing_generated_pngs(self):
        self.handler._handle_api_graphs()
        data, status = self.json_response

        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "available")
        self.assertEqual(data["generated_count"], 17)
        self.assertEqual(len(data["graphs"]), 17)
        self.assertEqual(data["skipped_count"], 1)
        self.assertEqual(data["skipped_graphs"][0]["metric"], "mape")
        self.assertIn("undefined", data["skipped_graphs"][0]["reason"])
        self.assertTrue(all(row["url"].startswith("/api/metric-graphs/") for row in data["graphs"]))

    def test_graph_endpoint_serves_existing_png_and_rejects_unlisted_mape(self):
        self.handler._handle_metric_graph("classification/auc_roc_comparison.png")
        path, content_type = self.file_response
        self.assertEqual(content_type, "image/png")
        self.assertTrue(path.is_file())

        self.file_response = None
        self.handler._handle_metric_graph("regression/mape_comparison.png")
        self.assertIsNone(self.file_response)
        self.assertEqual(self.error_response[0], 404)

    def test_missing_gcn_prediction_stays_null_and_known_tx_is_not_called_unsupported(self):
        known_tx = 123
        row = {
            "txId": str(known_tx),
            "Time step": "35",
            "rank": "1",
            "calibrated_illicit_risk": "0.72",
            "in_degree_asof": "2",
            "out_degree_asof": "3",
            "directed_total_degree_asof": "5",
            "undirected_degree_asof": "4",
            "scored_neighbor_count_asof": "2",
            "mean_scored_neighbor_risk_asof": "0.61",
            "structural_suspicion_rank_score": "0.83",
        }
        store = type("Store", (), {})()
        store.loaded = True
        store.load_data = lambda: None
        store.suspicious_rows = [row]
        store.suspicious_by_id = {known_tx: row}
        store.model_predictions_by_id = {}
        store.in_edges = {}
        store.out_edges = {}

        with patch("dashboard.server.STORE", store), patch("dashboard.server.HAS_NATIVE_INFERENCE", False):
            self.handler._handle_api_suspicious({"page": ["1"], "limit": ["25"]})
            result = self.json_response[0]["records"][0]
            self.assertIsNone(result["probability"])
            self.assertIsNone(result["model"])

            score = self.handler._execute_inference(known_tx, "gcn")
            self.assertEqual(score["status"], "inference_unavailable")
            self.assertEqual(score["record"]["status"], "inference_unavailable")
            self.assertEqual(score["record"]["timestep"], 35)


if __name__ == "__main__":
    unittest.main()
