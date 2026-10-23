"""Synthetic graph invariants and read-only checks for saved analytics outputs."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

import numpy as np

from models.graph_analytics.analyze_graph import (
    degree_statistics,
    louvain_communities,
    validate_degree_statistics,
    weak_components,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "models/graph_analytics/artifacts"


class GraphAnalyticsUnitTests(unittest.TestCase):
    def test_degree_and_component_invariants_on_small_graph(self):
        ids = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
        src = np.array([0, 1, 3], dtype=np.int64)
        dst = np.array([1, 2, 4], dtype=np.int64)
        incoming, outgoing, degree, neighbor_mean, pairs = degree_statistics(
            len(ids), src, dst
        )
        np.testing.assert_array_equal(incoming, [0, 1, 1, 0, 1, 0])
        np.testing.assert_array_equal(outgoing, [1, 1, 0, 1, 0, 0])
        np.testing.assert_array_equal(degree, [1, 2, 1, 1, 1, 0])
        np.testing.assert_allclose(neighbor_mean, [2, 1, 2, 1, 1, 0])
        validate_degree_statistics(
            incoming, outgoing, degree, neighbor_mean,
            directed_edges=3, undirected_edges=3,
        )

        component_ids, component_sizes, edge_components = weak_components(
            ids, pairs, src, dst
        )
        np.testing.assert_array_equal(component_ids, [10, 10, 10, 40, 40, 60])
        np.testing.assert_array_equal(component_sizes, [3, 3, 3, 2, 2, 1])
        self.assertEqual(edge_components.size, 3)

    def test_louvain_is_deterministic_and_assigns_every_node(self):
        ids = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
        src = np.array([0, 1, 2, 3, 4, 5, 2], dtype=np.int64)
        dst = np.array([1, 2, 0, 4, 5, 3, 3], dtype=np.int64)
        _, _, _, _, pairs = degree_statistics(len(ids), src, dst)
        first = louvain_communities(ids, pairs)
        second = louvain_communities(ids, pairs)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])
        self.assertTrue(np.isfinite(first[3]))
        self.assertEqual(int(first[2].sum()), len(ids))


class ExistingGraphArtifactTests(unittest.TestCase):
    def test_saved_graph_schemas_counts_missingness_and_summaries(self):
        stats = json.loads((ARTIFACTS / "graph_statistics.json").read_text())
        self.assertEqual(stats["nodes"], 203_769)
        self.assertEqual(stats["directed_edges"], 234_355)
        self.assertEqual(stats["in_degree_sum"], stats["directed_edges"])
        self.assertEqual(stats["out_degree_sum"], stats["directed_edges"])
        self.assertEqual(stats["undirected_degree_sum"], 468_710)
        self.assertEqual(stats["cross_timestep_edges"], 0)
        self.assertFalse(stats["labels_loaded_or_used"])

        node_header = [
            "txId", "Time step", "in_degree", "out_degree", "directed_total_degree",
            "undirected_degree", "mean_neighbor_undirected_degree", "weak_component_id",
            "weak_component_size", "community_id", "community_size",
        ]
        seen_ids: set[int] = set()
        in_sum = out_sum = row_count = 0
        with (ARTIFACTS / "node_graph_statistics.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, node_header)
            for row in reader:
                row_count += 1
                self.assertTrue(all(value != "" for value in row.values()))
                tx_id = int(row["txId"])
                self.assertNotIn(tx_id, seen_ids)
                seen_ids.add(tx_id)
                timestep = int(row["Time step"])
                self.assertIn(timestep, range(1, 50))
                self.assertEqual(int(row["directed_total_degree"]), int(row["in_degree"]) + int(row["out_degree"]))
                in_sum += int(row["in_degree"])
                out_sum += int(row["out_degree"])
                self.assertTrue(np.isfinite(float(row["mean_neighbor_undirected_degree"])))
        self.assertEqual(row_count, 203_769)
        self.assertEqual(in_sum, 234_355)
        self.assertEqual(out_sum, 234_355)

        with (ARTIFACTS / "component_summaries.csv").open(newline="", encoding="utf-8") as stream:
            component_reader = csv.DictReader(stream)
            self.assertEqual(component_reader.fieldnames, ["weak_component_id", "node_count", "directed_edge_count"])
            components = list(component_reader)
        self.assertEqual(len(components), stats["weak_connected_components"])
        self.assertEqual(sum(int(row["node_count"]) for row in components), 203_769)
        self.assertEqual(sum(int(row["directed_edge_count"]) for row in components), 234_355)
        self.assertTrue(all(all(value != "" for value in row.values()) for row in components))

        with (ARTIFACTS / "community_summaries.csv").open(newline="", encoding="utf-8") as stream:
            community_reader = csv.DictReader(stream)
            self.assertEqual(community_reader.fieldnames, ["community_id", "node_count", "mean_undirected_degree", "internal_undirected_edges"])
            communities = list(community_reader)
        self.assertEqual(len(communities), stats["community_detection"]["communities"])
        self.assertEqual(sum(int(row["node_count"]) for row in communities), 203_769)
        self.assertTrue(all(float(row["mean_undirected_degree"]) >= 0 for row in communities))
        self.assertTrue(all(all(value != "" for value in row.values()) for row in communities))

    def test_saved_suspicious_ranking_schema_and_values(self):
        expected_header = [
            "rank", "txId", "Time step", "calibrated_illicit_risk", "in_degree_asof",
            "out_degree_asof", "directed_total_degree_asof", "undirected_degree_asof",
            "scored_neighbor_count_asof", "mean_scored_neighbor_risk_asof",
            "max_scored_neighbor_risk_asof", "structural_suspicion_rank_score",
        ]
        ranks: set[int] = set()
        tx_ids: set[int] = set()
        count = 0
        with (ARTIFACTS / "suspicious_transactions.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, expected_header)
            for row in reader:
                count += 1
                self.assertTrue(all(value != "" for value in row.values()))
                rank = int(row["rank"])
                tx_id = int(row["txId"])
                self.assertNotIn(rank, ranks)
                self.assertNotIn(tx_id, tx_ids)
                ranks.add(rank)
                tx_ids.add(tx_id)
                self.assertIn(int(row["Time step"]), range(35, 50))
                self.assertTrue(0 <= float(row["calibrated_illicit_risk"]) <= 1)
                self.assertTrue(0 <= float(row["structural_suspicion_rank_score"]) <= 1)
                self.assertGreaterEqual(int(row["scored_neighbor_count_asof"]), 0)
        self.assertEqual(count, 67_504)


if __name__ == "__main__":
    unittest.main()
