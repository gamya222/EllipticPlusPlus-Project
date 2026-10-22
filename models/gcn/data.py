"""Load the generated Parquet graph and prepare train-only feature transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from pyspark.sql import SparkSession


PROCESSED_ROOT = Path("data/processed/elliptic")
METADATA_COLUMNS = {"txId", "Time step", "class"}


@dataclass
class GraphData:
    ids: np.ndarray
    times: np.ndarray
    labels: np.ndarray  # 1=illicit, 0=licit, -1=unknown
    raw_features: np.ndarray
    feature_names: list[str]
    source_edges: np.ndarray
    target_edges: np.ndarray


def _numpy_column(table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
    column = table[name].combine_chunks()
    if column.null_count:
        if pa.types.is_floating(column.type):
            column = pc.fill_null(column, float("nan"))
        else:
            raise ValueError(f"Unexpected NULLs in metadata column {name!r}")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)


def load_graph_data() -> GraphData:
    """Read the documented transaction/graph Parquet outputs without rewriting them."""
    spark = (
        SparkSession.builder.appName("EllipticPlusPlusGCN")
        .master("local[*]")
        .config("spark.driver.memory", "3g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        train = spark.read.parquet(str(PROCESSED_ROOT / "transactions_train"))
        test = spark.read.parquet(str(PROCESSED_ROOT / "transactions_test"))
        if train.schema != test.schema:
            raise ValueError("Train and test Parquet schemas do not match")
        tx = train.unionByName(test).orderBy("txId")
        feature_names = [name for name in tx.columns if name not in METADATA_COLUMNS]
        if len(feature_names) != 182:
            raise ValueError(f"Expected 182 numeric feature columns, found {len(feature_names)}")
        table = tx.select("txId", "Time step", *feature_names, "class").toArrow()

        ids = _numpy_column(table, "txId", np.dtype("int64"))
        times = _numpy_column(table, "Time step", np.dtype("int16"))
        classes = _numpy_column(table, "class", np.dtype("int8"))
        if np.any(ids[1:] <= ids[:-1]):
            raise ValueError("txId ordering is not unique and strictly increasing")
        if not np.isin(classes, [1, 2, 3]).all():
            raise ValueError("Unexpected class value in processed transactions")

        # Elliptic convention: class 1 illicit, class 2 licit, class 3 unknown.
        labels = np.full(classes.shape, -1, dtype=np.int8)
        labels[classes == 1] = 1
        labels[classes == 2] = 0
        raw_features = np.column_stack(
            [_numpy_column(table, name, np.dtype("float64")) for name in feature_names]
        )

        edge_table = spark.read.parquet(str(PROCESSED_ROOT / "graph" / "edges")).toArrow()
        source_ids = _numpy_column(edge_table, "txId1", np.dtype("int64"))
        target_ids = _numpy_column(edge_table, "txId2", np.dtype("int64"))
        src = np.searchsorted(ids, source_ids)
        dst = np.searchsorted(ids, target_ids)
        if (src >= ids.size).any() or (dst >= ids.size).any():
            raise ValueError("An edge endpoint is absent from transaction data")
        if not np.array_equal(ids[src], source_ids) or not np.array_equal(ids[dst], target_ids):
            raise ValueError("An edge endpoint is absent from transaction data")
        if len(np.unique(np.stack([src, dst], axis=1), axis=0)) != source_ids.size:
            raise ValueError("Duplicate directed edge rows in graph Parquet")

        train_graph_ids = _numpy_column(
            spark.read.parquet(str(PROCESSED_ROOT / "graph" / "train_nodes"))
            .select("txId").orderBy("txId").toArrow(),
            "txId",
            np.dtype("int64"),
        )
        test_graph_ids = _numpy_column(
            spark.read.parquet(str(PROCESSED_ROOT / "graph" / "test_nodes"))
            .select("txId").orderBy("txId").toArrow(),
            "txId",
            np.dtype("int64"),
        )
        if not np.array_equal(train_graph_ids, ids[times <= 34]):
            raise ValueError("Training graph nodes do not exactly match transaction training IDs")
        if not np.array_equal(test_graph_ids, ids[times >= 35]):
            raise ValueError("Testing graph nodes do not exactly match transaction testing IDs")
        if ids.size != 203_769 or source_ids.size != 234_355:
            raise ValueError(
                "Generated transaction/edge totals differ from validated raw counts: "
                f"transactions={ids.size}, edges={source_ids.size}"
            )
        if train_graph_ids.size != 136_265 or test_graph_ids.size != 67_504:
            raise ValueError(
                "Generated temporal node counts differ from the expected split: "
                f"train={train_graph_ids.size}, test={test_graph_ids.size}"
            )
        print(
            "Parquet schema checks passed: train/test rows="
            f"{train_graph_ids.size}/{test_graph_ids.size}, edges={source_ids.size}, "
            f"columns={len(train.columns)} (txId int, Time step int, "
            "182 double features, class int)."
        )

        return GraphData(ids, times, labels, raw_features, feature_names, src, dst)
    finally:
        spark.stop()


@dataclass
class FeatureTransform:
    medians: np.ndarray
    centers: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, raw: np.ndarray, times: np.ndarray, fit_through: int) -> "FeatureTransform":
        fit_rows = times <= fit_through
        if not fit_rows.any():
            raise ValueError(f"No rows available to fit through timestep {fit_through}")
        train_values = raw[fit_rows]
        with np.errstate(all="ignore"):
            medians = np.nanmedian(train_values, axis=0)
        if not np.isfinite(medians).all():
            missing = np.flatnonzero(~np.isfinite(medians)).tolist()
            raise ValueError(f"Features have no finite training observations: {missing}")
        filled = np.where(np.isfinite(train_values), train_values, medians)
        centers = np.median(filled, axis=0)
        q25, q75 = np.percentile(filled, [25, 75], axis=0)
        scales = q75 - q25
        scales[~np.isfinite(scales) | (scales == 0)] = 1.0
        return cls(medians, centers, scales)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(raw), raw, self.medians)
        scaled = (filled - self.centers) / self.scales
        # Elliptic amount/count features have extreme tails after robust scaling.
        # Signed log compression limits their influence without fitting anything
        # on validation or test data and preserves sign and rank ordering.
        scaled = np.sign(scaled) * np.log1p(np.abs(scaled))
        if not np.isfinite(scaled).all():
            raise ValueError("Feature transformation produced non-finite values")
        return scaled.astype(np.float32)


def normalized_adjacency(
    num_nodes: int,
    source_edges: np.ndarray,
    target_edges: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Build binary, undirected A-hat = D^-1/2 (A + I) D^-1/2."""
    nodes = np.arange(num_nodes, dtype=np.int64)
    rows = np.concatenate((source_edges, target_edges, nodes))
    cols = np.concatenate((target_edges, source_edges, nodes))
    indices = torch.from_numpy(np.stack((rows, cols)))
    values = torch.ones(rows.size, dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    indices = adjacency.indices()
    # Coalesce first so a reverse-listed edge pair remains one undirected connection.
    degrees = torch.bincount(indices[0], minlength=num_nodes).to(torch.float32)
    inv_sqrt = degrees.clamp_min(1).rsqrt()
    weights = inv_sqrt[indices[0]] * inv_sqrt[indices[1]]
    return torch.sparse_coo_tensor(indices, weights, (num_nodes, num_nodes)).coalesce().to(device)


def mean_adjacency(
    num_nodes: int,
    source_edges: np.ndarray,
    target_edges: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Build row-normalized binary undirected adjacency for mean aggregation."""
    rows = np.concatenate((source_edges, target_edges))
    cols = np.concatenate((target_edges, source_edges))
    indices = torch.from_numpy(np.stack((rows, cols)))
    adjacency = torch.sparse_coo_tensor(
        indices,
        torch.ones(rows.size, dtype=torch.float32),
        (num_nodes, num_nodes),
    ).coalesce()
    indices = adjacency.indices()
    degrees = torch.bincount(indices[0], minlength=num_nodes).to(torch.float32)
    weights = degrees[indices[0]].clamp_min(1).reciprocal()
    return torch.sparse_coo_tensor(
        indices, weights, (num_nodes, num_nodes)
    ).coalesce().to(device)


def prefix_graph(data: GraphData, cutoff: int, device: torch.device):
    """Return nodes and adjacency containing no transaction later than cutoff."""
    global_indices = np.flatnonzero(data.times <= cutoff)
    global_to_local = np.full(data.ids.size, -1, dtype=np.int64)
    global_to_local[global_indices] = np.arange(global_indices.size, dtype=np.int64)
    src = global_to_local[data.source_edges]
    dst = global_to_local[data.target_edges]
    keep = (src >= 0) & (dst >= 0)
    adjacency = normalized_adjacency(global_indices.size, src[keep], dst[keep], device)
    return global_indices, adjacency


def prefix_mean_graph(data: GraphData, cutoff: int, device: torch.device):
    """Return the chronological graph prefix with row-mean adjacency."""
    global_indices = np.flatnonzero(data.times <= cutoff)
    global_to_local = np.full(data.ids.size, -1, dtype=np.int64)
    global_to_local[global_indices] = np.arange(global_indices.size, dtype=np.int64)
    src = global_to_local[data.source_edges]
    dst = global_to_local[data.target_edges]
    keep = (src >= 0) & (dst >= 0)
    adjacency = mean_adjacency(global_indices.size, src[keep], dst[keep], device)
    return global_indices, adjacency
