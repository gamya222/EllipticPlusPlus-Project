"""Reusable batch inference for existing Elliptic++ test-period transactions.

Run from the repository root with the project's WSL environment:
    TORCH_USE_RTLD_GLOBAL=1 python -m models.inference --tx-ids 13385567 12345678

Only known transactions at timesteps 35-49 are supported. This module reads no
class column and never fits preprocessing, calibration, or decision thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
import torch
from torch import nn

from models.gcn.data import (
    FeatureTransform,
    GraphData,
    PROCESSED_ROOT,
    prefix_graph,
    prefix_mean_graph,
)
from models.gcn.model import GCN
from models.graphsage.model import GraphSAGE
from models.gru.model import TemporalGRU
from models.gru.train import make_timestep_summaries


ROOT = Path(__file__).resolve().parents[1]
ANALYTICS = ROOT / "models/graph_analytics/artifacts"
CHECKPOINTS = {
    "gcn": ROOT / "models/gcn/artifacts/gcn_baseline.pt",
    "graphsage": ROOT / "models/graphsage/artifacts/graphsage_baseline.pt",
    "gru": ROOT / "models/gru/artifacts/temporal_gru_baseline.pt",
    "logistic": ROOT / "models/logistic_baseline/artifacts/logistic_baseline.pt",
}
RISK_ARTIFACT = ROOT / "models/risk_predictor/artifacts/risk_scorer.pt"
SUPPORTED_MODELS = tuple(CHECKPOINTS)
TEST_START, TEST_END = 35, 49


def _load_parquet(directory: Path, columns: list[str] | None = None):
    if not directory.is_dir():
        raise FileNotFoundError(f"Required Parquet directory is missing: {directory}")
    return ds.dataset(str(directory), format="parquet").to_table(columns=columns)


def _numpy(table, name: str, dtype) -> np.ndarray:
    column = table[name].combine_chunks()
    if column.null_count:
        if np.issubdtype(np.dtype(dtype), np.floating):
            import pyarrow.compute as pc

            column = pc.fill_null(column, float("nan"))
        else:
            raise ValueError(f"Unexpected nulls in required column {name!r}")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)


def _read_graph_data() -> GraphData:
    """Read supported features and edges only; deliberately never read labels."""
    train_dir = PROCESSED_ROOT / "transactions_train"
    test_dir = PROCESSED_ROOT / "transactions_test"
    train_schema = ds.dataset(str(train_dir), format="parquet").schema
    test_schema = ds.dataset(str(test_dir), format="parquet").schema
    if train_schema != test_schema:
        raise ValueError("Train/test transaction schemas differ")
    names = train_schema.names
    features = [name for name in names if name not in {"txId", "Time step", "class"}]
    if len(features) != 182:
        raise ValueError(f"Expected 182 numeric features, found {len(features)}")
    columns = ["txId", "Time step", *features]
    train = _load_parquet(train_dir, columns)
    test = _load_parquet(test_dir, columns)
    table = __import__("pyarrow").concat_tables([train, test]).sort_by([("txId", "ascending")])
    ids = _numpy(table, "txId", np.int64)
    times = _numpy(table, "Time step", np.int16)
    if np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Transaction IDs are not unique")
    raw = np.column_stack([_numpy(table, name, np.float64) for name in features])
    edge_table = _load_parquet(PROCESSED_ROOT / "graph/edges", ["txId1", "txId2"])
    source_ids = _numpy(edge_table, "txId1", np.int64)
    target_ids = _numpy(edge_table, "txId2", np.int64)
    src = np.searchsorted(ids, source_ids)
    dst = np.searchsorted(ids, target_ids)
    if (src >= ids.size).any() or (dst >= ids.size).any():
        raise ValueError("Graph edge references an absent transaction")
    if not np.array_equal(ids[src], source_ids) or not np.array_equal(ids[dst], target_ids):
        raise ValueError("Graph edge references an absent transaction")
    return GraphData(ids, times, np.full(ids.size, -1, np.int8), raw, features, src, dst)


def _transform(checkpoint: dict[str, Any]) -> FeatureTransform:
    required = ("feature_medians", "feature_centers", "feature_scales")
    if any(key not in checkpoint for key in required):
        raise ValueError("Checkpoint lacks saved feature transformation parameters")
    result = FeatureTransform(*(np.asarray(checkpoint[key], dtype=np.float64) for key in required))
    if not (result.medians.shape == result.centers.shape == result.scales.shape):
        raise ValueError("Saved transform arrays have inconsistent shapes")
    if not np.isfinite(result.medians).all() or not np.isfinite(result.centers).all():
        raise ValueError("Saved transform contains non-finite imputation/center parameters")
    if not np.isfinite(result.scales).all() or (result.scales <= 0).any():
        raise ValueError("Saved transform contains invalid scales")
    return result


class EllipticInference:
    """Batch scorer for known test-period nodes; no label-based runtime path."""

    def __init__(self, device: str | torch.device | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.data = _read_graph_data()
        self.id_to_row = {int(tx_id): i for i, tx_id in enumerate(self.data.ids)}
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.transforms: dict[str, FeatureTransform] = {}
        self.models: dict[str, nn.Module] = {}
        self.feature_cache: dict[str, torch.Tensor] = {}
        self._risk_cache: np.ndarray | None = None
        self._load_models()
        self.risk_artifact = torch.load(RISK_ARTIFACT, map_location="cpu", weights_only=False)
        self._load_risk_model()
        self.graph_stats = self._load_csv_map(
            ANALYTICS / "node_graph_statistics.csv", "txId"
        )
        self.suspicious = self._load_csv_map(
            ANALYTICS / "suspicious_transactions.csv", "txId"
        )
        self._validate_analytics_alignment()

    @staticmethod
    def _load_csv_map(path: Path, key: str) -> dict[int, dict[str, str]]:
        if not path.is_file():
            raise FileNotFoundError(f"Required graph analytics output is missing: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            if key not in (rows.fieldnames or []):
                raise ValueError(f"{path.name} has no {key} field")
            output = {}
            for row in rows:
                tx_id = int(row[key])
                if tx_id in output:
                    raise ValueError(f"Duplicate transaction {tx_id} in {path.name}")
                output[tx_id] = row
        return output

    def _validate_analytics_alignment(self) -> None:
        if self.data.ids.size != 203_769 or self.data.source_edges.size != 234_355:
            raise ValueError(
                "Processed graph counts differ from the validated artifacts: "
                f"nodes={self.data.ids.size}, edges={self.data.source_edges.size}"
            )
        if len(self.graph_stats) != self.data.ids.size:
            raise ValueError("Node graph-statistics rows do not match transaction count")
        expected_test_ids = set(map(int, self.data.ids[(self.data.times >= TEST_START) & (self.data.times <= TEST_END)]))
        if set(self.suspicious) != expected_test_ids:
            raise ValueError("Suspicious-ranking IDs do not exactly match the test period")
        for source, rows in (("node statistics", self.graph_stats), ("suspicious ranking", self.suspicious)):
            for tx_id, values in rows.items():
                index = self.id_to_row.get(tx_id)
                if index is None or int(values["Time step"]) != int(self.data.times[index]):
                    raise ValueError(f"{source} transaction/timestep mismatch for {tx_id}")

    def _load_models(self) -> None:
        for name, path in CHECKPOINTS.items():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if len(checkpoint.get("feature_names", [])) != 182:
                raise ValueError(f"{name} checkpoint has unexpected feature order/schema")
            if checkpoint["feature_names"] != self.data.feature_names:
                raise ValueError(f"{name} feature order differs from transaction Parquet")
            config = checkpoint.get("config", {})
            dim, hidden, dropout = len(self.data.feature_names), config.get("hidden_dim", 64), config.get("dropout", 0.5)
            if name == "gcn":
                model = GCN(dim, hidden, dropout)
            elif name == "graphsage":
                model = GraphSAGE(dim, hidden, dropout)
            elif name == "gru":
                model = TemporalGRU(dim, hidden, dropout)
            else:
                model = nn.Linear(dim, 2)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(self.device).eval()
            self.checkpoints[name] = checkpoint
            self.transforms[name] = _transform(checkpoint)
            self.models[name] = model

    def _load_risk_model(self) -> None:
        artifact = self.risk_artifact
        gru_path = ROOT / artifact["gru_checkpoint"]
        digest = hashlib.sha256(gru_path.read_bytes()).hexdigest()
        if digest != artifact["gru_checkpoint_sha256"]:
            raise ValueError("Risk artifact GRU checkpoint checksum mismatch")
        gru_checkpoint = self.checkpoints["gru"]
        if artifact["feature_names"] != self.data.feature_names:
            raise ValueError("Risk artifact feature order differs from Parquet")
        if artifact["gru_config"] != gru_checkpoint["config"]:
            raise ValueError("Risk artifact GRU configuration mismatch")
        self.risk_transform = _transform(artifact)
        config = artifact["gru_config"]
        model = TemporalGRU(len(self.data.feature_names), config["hidden_dim"], config["dropout"])
        model.load_state_dict(gru_checkpoint["validation_state_dict"], strict=True)
        self.risk_model = model.to(self.device).eval()

    def _features(self, name: str) -> torch.Tensor:
        if name not in self.feature_cache:
            values = self.transforms[name].transform(self.data.raw_features)
            self.feature_cache[name] = torch.from_numpy(values).to(self.device)
        return self.feature_cache[name]

    def _gru_logits(self, model: nn.Module, features: torch.Tensor) -> torch.Tensor:
        summaries_np = make_timestep_summaries(
            features.detach().cpu().numpy(), self.data.times
        )
        summaries = torch.from_numpy(summaries_np).to(self.device)
        timestep_ids = torch.from_numpy(self.data.times.astype(np.int64) - 1).to(self.device)
        return model(features, timestep_ids, summaries)

    @torch.inference_mode()
    def _score_rows(self, model_name: str, rows: list[int]) -> dict[int, float]:
        if not rows:
            return {}
        model = self.models[model_name]
        features = self._features(model_name)
        result: dict[int, float] = {}
        if model_name in {"gcn", "graphsage"}:
            graph_builder = prefix_graph if model_name == "gcn" else prefix_mean_graph
            for timestep in sorted({int(self.data.times[row]) for row in rows}):
                global_nodes, adjacency = graph_builder(self.data, timestep, self.device)
                local = {int(global_row): pos for pos, global_row in enumerate(global_nodes)}
                model_logits = model(features[global_nodes], adjacency)
                wanted = [row for row in rows if int(self.data.times[row]) == timestep]
                selected = torch.tensor([local[row] for row in wanted], device=self.device)
                probabilities = torch.softmax(model_logits[selected], dim=1)[:, 1].cpu().numpy()
                result.update({row: float(score) for row, score in zip(wanted, probabilities, strict=True)})
        elif model_name == "gru":
            logits = self._gru_logits(model, features)
            for row in rows:
                result[row] = float(torch.softmax(logits[row], dim=0)[1].cpu())
        else:
            probabilities = torch.softmax(model(features[rows]), dim=1)[:, 1].cpu().numpy()
            result.update({row: float(score) for row, score in zip(rows, probabilities, strict=True)})
        return result

    @torch.inference_mode()
    def _risk_scores(self, rows: list[int]) -> dict[int, float]:
        if not rows:
            return {}
        if self._risk_cache is None:
            raw = self.risk_transform.transform(self.data.raw_features)
            features = torch.from_numpy(raw).to(self.device)
            logits = self._gru_logits(self.risk_model, features)
            margin = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy().astype(np.float64)
            slope = float(self.risk_artifact["calibrator"]["slope"])
            intercept = float(self.risk_artifact["calibrator"]["intercept"])
            self._risk_cache = 1.0 / (1.0 + np.exp(-np.clip(slope * margin + intercept, -40, 40)))
        return {row: float(self._risk_cache[row]) for row in rows}

    @staticmethod
    def _as_int(row: dict[str, str], key: str) -> int | None:
        value = row.get(key)
        return int(value) if value not in (None, "") else None

    def score_batch(self, tx_ids: list[int], model: str = "gcn") -> list[dict[str, Any]]:
        """Score existing test-period IDs; unsupported/unknown IDs get explicit nulls."""
        model = model.lower()
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"model must be one of {SUPPORTED_MODELS}")
        resolved = [(int(tx_id), self.id_to_row.get(int(tx_id))) for tx_id in tx_ids]
        supported_rows = [
            row for _, row in resolved
            if row is not None and TEST_START <= int(self.data.times[row]) <= TEST_END
        ]
        model_scores = self._score_rows(model, supported_rows)
        risks = self._risk_scores(supported_rows)
        outputs = []
        for tx_id, row_idx in resolved:
            base: dict[str, Any] = {
                "txId": tx_id,
                "Time step": int(self.data.times[row_idx]) if row_idx is not None else None,
                "status": "scored" if row_idx is not None and TEST_START <= int(self.data.times[row_idx]) <= TEST_END else ("unknown_transaction_id" if row_idx is None else "unsupported_timestep"),
                "requested_model": model,
                "model_output": {"score_type": "illicit_class_probability", "score": None, "prediction": None, "threshold": 0.5},
                "risk": {"score_type": "validation_calibrated_gru_illicit_probability", "score": None, "threshold": float(self.risk_artifact["threshold"])},
                "graph_statistics": None,
                "community": None,
                "suspicious_connection": None,
            }
            if base["status"] == "scored":
                score = model_scores[row_idx]
                base["model_output"].update(score=score, prediction="illicit" if score >= 0.5 else "licit")
                base["risk"]["score"] = risks[row_idx]
                stat = self.graph_stats.get(tx_id)
                if stat is not None:
                    base["graph_statistics"] = {
                        "scope": "full_graph_descriptive_includes_later_timesteps",
                        "in_degree": self._as_int(stat, "in_degree"),
                        "out_degree": self._as_int(stat, "out_degree"),
                        "directed_total_degree": self._as_int(stat, "directed_total_degree"),
                        "undirected_degree": self._as_int(stat, "undirected_degree"),
                        "mean_neighbor_undirected_degree": float(stat["mean_neighbor_undirected_degree"]) if stat.get("mean_neighbor_undirected_degree") else None,
                        "weak_component_id": self._as_int(stat, "weak_component_id"),
                        "weak_component_size": self._as_int(stat, "weak_component_size"),
                    }
                    base["community"] = {
                        "id": self._as_int(stat, "community_id"),
                        "size": self._as_int(stat, "community_size"),
                        "scope": "full_graph_descriptive_includes_later_timesteps",
                    }
                suspicious = self.suspicious.get(tx_id)
                if suspicious is not None:
                    base["suspicious_connection"] = {
                        "rank": self._as_int(suspicious, "rank"),
                        "heuristic_score": float(suspicious["structural_suspicion_rank_score"]),
                        "heuristic_name": "structural_suspicion_rank_score",
                        "scope": "as_of_transaction_timestep",
                        "scored_neighbor_count": self._as_int(suspicious, "scored_neighbor_count_asof"),
                        "mean_scored_neighbor_risk": float(suspicious["mean_scored_neighbor_risk_asof"]) if suspicious.get("mean_scored_neighbor_risk_asof") else None,
                    }
            outputs.append(base)
        return outputs

    def batch_document(self, tx_ids: list[int], model: str = "gcn") -> dict[str, Any]:
        """Create a versioned, visualization-friendly JSON document."""
        records = self.score_batch(tx_ids, model)
        return {
            "schema_version": "elliptic-inference-v1",
            "requested_model": model.lower(),
            "supported_timesteps": [TEST_START, TEST_END],
            "population": "known transactions in the existing held-out dataset",
            "record_count": len(records),
            "status_counts": {
                status: sum(record["status"] == status for record in records)
                for status in ("scored", "unknown_transaction_id", "unsupported_timestep")
            },
            "graph_scope": {
                "graph_statistics_and_community": "full-graph descriptive; includes later timesteps",
                "suspicious_connection": "as-of transaction timestep ranking heuristic",
            },
            "records": records,
        }

    @staticmethod
    def write_batch_document(document: dict[str, Any], output_path: str | Path) -> Path:
        """Write JSON without replacing an existing file."""
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--tx-ids", nargs="+", type=int)
    selection.add_argument(
        "--all-test", action="store_true",
        help="score every known transaction at timesteps 35-49 (larger batch)",
    )
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="gcn")
    parser.add_argument("--device", default=None, help="cpu, cuda, or omitted for automatic selection")
    parser.add_argument("--output", help="write versioned JSON to a new file; existing files are never replaced")
    args = parser.parse_args()
    scorer = EllipticInference(device=args.device)
    tx_ids = args.tx_ids
    if args.all_test:
        mask = (scorer.data.times >= TEST_START) & (scorer.data.times <= TEST_END)
        tx_ids = [int(tx_id) for tx_id in scorer.data.ids[mask]]
    document = scorer.batch_document(tx_ids, args.model)
    if args.output:
        path = scorer.write_batch_document(document, args.output)
        print(f"WROTE {path} records={document['record_count']} schema={document['schema_version']}")
    else:
        print(json.dumps(document, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
