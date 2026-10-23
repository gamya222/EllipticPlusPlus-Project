"""Read existing graph Parquet and emit deterministic, label-free analytics."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import networkx as nx


ROOT = Path(__file__).resolve().parents[2]
GRAPH_ROOT = ROOT / "data/processed/elliptic/graph"
ANALYTICS_DIR = Path(__file__).resolve().parent / "artifacts"
EXPECTED_NODES = 203_769
EXPECTED_DIRECTED_EDGES = 234_355
RISK_PATH = ROOT / "models/risk_predictor/artifacts/test_risk_scores.csv"


def read_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read only IDs, timesteps, and endpoints; class labels are never loaded."""
    train = ds.dataset(str(ROOT / "data/processed/elliptic/graph/train_nodes"), format="parquet")
    test = ds.dataset(str(ROOT / "data/processed/elliptic/graph/test_nodes"), format="parquet")
    train_table = train.to_table(columns=["txId", "Time step"])
    test_table = test.to_table(columns=["txId", "Time step"])
    ids = np.concatenate((train_table["txId"].to_numpy(), test_table["txId"].to_numpy())).astype(np.int64)
    times = np.concatenate((train_table["Time step"].to_numpy(), test_table["Time step"].to_numpy())).astype(np.int16)
    order = np.argsort(ids, kind="stable")
    ids, times = ids[order], times[order]
    if ids.size != EXPECTED_NODES or np.any(ids[1:] <= ids[:-1]):
        raise ValueError(f"Unexpected graph node count or duplicate transaction IDs: {ids.size}")
    if np.any((times < 1) | (times > 49)):
        raise ValueError("Graph node timestep outside the documented 1–49 range")

    edge_table = ds.dataset(str(GRAPH_ROOT / "edges"), format="parquet").to_table(
        columns=["txId1", "txId2"]
    )
    source_ids = edge_table["txId1"].to_numpy().astype(np.int64)
    target_ids = edge_table["txId2"].to_numpy().astype(np.int64)
    if source_ids.size != EXPECTED_DIRECTED_EDGES:
        raise ValueError(f"Expected {EXPECTED_DIRECTED_EDGES} directed edges, found {source_ids.size}")
    src = np.searchsorted(ids, source_ids)
    dst = np.searchsorted(ids, target_ids)
    if (src >= ids.size).any() or (dst >= ids.size).any():
        raise ValueError("Edge endpoint is not present in the graph node data")
    if not np.array_equal(ids[src], source_ids) or not np.array_equal(ids[dst], target_ids):
        raise ValueError("Edge endpoint is not present in the graph node data")
    if np.any(src == dst):
        raise ValueError("Self-loop found in graph edge artifact")
    pairs = np.column_stack((src, dst))
    if np.unique(pairs, axis=0).shape[0] != source_ids.size:
        raise ValueError("Duplicate directed edge found in graph edge artifact")
    return ids, times, src, dst, source_ids


def degree_statistics(
    num_nodes: int, src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Directed in/out degrees and undirected degree/neighborhood mean."""
    out_degree = np.bincount(src, minlength=num_nodes).astype(np.int64)
    in_degree = np.bincount(dst, minlength=num_nodes).astype(np.int64)
    low, high = np.minimum(src, dst), np.maximum(src, dst)
    undirected_pairs = np.unique(np.column_stack((low, high)), axis=0)
    u, v = undirected_pairs.T
    undirected_degree = np.bincount(np.concatenate((u, v)), minlength=num_nodes).astype(np.int64)
    neighbor_degree_sum = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(neighbor_degree_sum, u, undirected_degree[v])
    np.add.at(neighbor_degree_sum, v, undirected_degree[u])
    mean_neighbor_degree = np.divide(
        neighbor_degree_sum,
        undirected_degree,
        out=np.zeros(num_nodes, dtype=np.float64),
        where=undirected_degree > 0,
    )
    return in_degree, out_degree, undirected_degree, mean_neighbor_degree, undirected_pairs


def validate_degree_statistics(
    in_degree: np.ndarray,
    out_degree: np.ndarray,
    undirected_degree: np.ndarray,
    mean_neighbor_degree: np.ndarray,
    directed_edges: int,
    undirected_edges: int,
) -> None:
    assert int(in_degree.sum()) == directed_edges
    assert int(out_degree.sum()) == directed_edges
    assert int(undirected_degree.sum()) == 2 * undirected_edges
    assert np.isfinite(mean_neighbor_degree).all()
    assert np.all(mean_neighbor_degree >= 0)


def weak_components(
    ids: np.ndarray, undirected_pairs: np.ndarray, src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find weak components; component IDs are the minimum txId in each set."""
    n = ids.size
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for left, right in undirected_pairs:
        a, b = find(int(left)), find(int(right))
        if a == b:
            continue
        if size[a] < size[b] or (size[a] == size[b] and ids[a] > ids[b]):
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    canonical_ids = np.empty(unique_roots.size, dtype=np.int64)
    component_sizes = np.bincount(inverse, minlength=unique_roots.size).astype(np.int64)
    for index, root in enumerate(unique_roots):
        canonical_ids[index] = ids[roots == root].min()
    node_component_ids = canonical_ids[inverse]
    node_component_sizes = component_sizes[inverse]
    assert np.all(node_component_ids[src] == node_component_ids[dst])
    edge_component_ids = node_component_ids[src]
    edge_counts = np.bincount(inverse[src], minlength=unique_roots.size).astype(np.int64)
    assert int(component_sizes.sum()) == n
    assert int(edge_counts.sum()) == src.size
    return node_component_ids, node_component_sizes, edge_component_ids


def louvain_communities(
    ids: np.ndarray, undirected_pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run seeded Louvain on the unweighted, undirected simple projection."""
    graph = nx.Graph()
    graph.add_nodes_from(range(ids.size))
    graph.add_edges_from((int(left), int(right)) for left, right in undirected_pairs)
    partitions = nx.community.louvain_communities(
        graph, weight=None, resolution=1.0, threshold=1e-7, seed=2026
    )
    partitions.sort(key=lambda group: int(ids[min(group)]))
    node_community = np.full(ids.size, -1, dtype=np.int64)
    community_ids = np.empty(len(partitions), dtype=np.int64)
    community_sizes = np.empty(len(partitions), dtype=np.int64)
    for index, members in enumerate(partitions):
        member_indices = np.fromiter(members, dtype=np.int64)
        community_ids[index] = int(ids[member_indices].min())
        community_sizes[index] = member_indices.size
        node_community[member_indices] = community_ids[index]
    if np.any(node_community < 0) or int(community_sizes.sum()) != ids.size:
        raise ValueError("Louvain did not assign every node exactly once")
    modularity = float(nx.community.modularity(graph, partitions, weight=None, resolution=1.0))
    return node_community, community_ids, community_sizes, modularity


def validate_community_determinism() -> None:
    """Small deterministic regression check for the fixed-seed Louvain call."""
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3), (6, 7)])
    first = nx.community.louvain_communities(graph, weight=None, seed=2026)
    second = nx.community.louvain_communities(graph, weight=None, seed=2026)
    canonical = lambda groups: sorted(tuple(sorted(group)) for group in groups)
    assert canonical(first) == canonical(second)


def write_statistics(
    ids: np.ndarray,
    times: np.ndarray,
    in_degree: np.ndarray,
    out_degree: np.ndarray,
    undirected_degree: np.ndarray,
    mean_neighbor_degree: np.ndarray,
    undirected_pairs: np.ndarray,
    component_ids: np.ndarray,
    component_sizes: np.ndarray,
    edge_component_ids: np.ndarray,
    community_ids: np.ndarray,
    community_sizes: np.ndarray,
    modularity: float,
    cross_timestep_edges: int,
    runtime: float,
) -> None:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    node_path = ANALYTICS_DIR / "node_graph_statistics.csv"
    with node_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("txId", "Time step", "in_degree", "out_degree", "directed_total_degree", "undirected_degree", "mean_neighbor_undirected_degree", "weak_component_id", "weak_component_size", "community_id", "community_size"))
        for i in range(ids.size):
            writer.writerow((int(ids[i]), int(times[i]), int(in_degree[i]), int(out_degree[i]),
                             int(in_degree[i] + out_degree[i]), int(undirected_degree[i]),
                             f"{mean_neighbor_degree[i]:.8f}", int(component_ids[i]), int(component_sizes[i]),
                             int(community_ids[i]), int(community_sizes[i])))

    component_path = ANALYTICS_DIR / "component_summaries.csv"
    component_values, component_node_counts = np.unique(component_ids, return_counts=True)
    with component_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("weak_component_id", "node_count", "directed_edge_count"))
        for component_id, node_count in zip(component_values, component_node_counts, strict=True):
            edge_count = int(np.sum(edge_component_ids == component_id))
            writer.writerow((int(component_id), int(node_count), edge_count))

    community_path = ANALYTICS_DIR / "community_summaries.csv"
    community_values, community_node_counts = np.unique(community_ids, return_counts=True)
    community_index = np.searchsorted(community_values, community_ids)
    edge_community = community_index[undirected_pairs[:, 0]]
    internal_edges = edge_community == community_index[undirected_pairs[:, 1]]
    internal_edge_counts = np.bincount(
        edge_community[internal_edges], minlength=community_values.size
    )
    degree_sums = np.bincount(
        community_index, weights=undirected_degree, minlength=community_values.size
    )
    with community_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("community_id", "node_count", "mean_undirected_degree", "internal_undirected_edges"))
        for index, (community_id, node_count) in enumerate(zip(community_values, community_node_counts, strict=True)):
            writer.writerow((int(community_id), int(node_count),
                             f"{degree_sums[index] / node_count:.8f}", int(internal_edge_counts[index])))

    summary = {
        "nodes": int(ids.size),
        "directed_edges": int(in_degree.sum()),
        "undirected_simple_edges": int(undirected_pairs.shape[0]),
        "cross_timestep_edges": cross_timestep_edges,
        "in_degree_sum": int(in_degree.sum()),
        "out_degree_sum": int(out_degree.sum()),
        "undirected_degree_sum": int(undirected_degree.sum()),
        "directed_isolates": int(np.sum((in_degree + out_degree) == 0)),
        "undirected_isolates": int(np.sum(undirected_degree == 0)),
        "in_degree_max": int(in_degree.max(initial=0)),
        "out_degree_max": int(out_degree.max(initial=0)),
        "undirected_degree_max": int(undirected_degree.max(initial=0)),
        "mean_neighbor_undirected_degree_max": float(mean_neighbor_degree.max(initial=0)),
        "weak_connected_components": int(component_values.size),
        "largest_component_nodes": int(component_node_counts.max(initial=0)),
        "component_size_median": float(np.median(component_node_counts)) if component_node_counts.size else 0,
        "community_detection": {
            "algorithm": "NetworkX Louvain",
            "projection": "unweighted undirected simple graph",
            "seed": 2026,
            "resolution": 1.0,
            "communities": int(community_values.size),
            "largest_community_nodes": int(community_node_counts.max(initial=0)),
            "modularity": modularity,
        },
        "runtime_seconds": runtime,
        "labels_loaded_or_used": False,
    }
    (ANALYTICS_DIR / "graph_statistics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    # Round-trip schema, row count, missingness, and deterministic ID order checks.
    with node_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["txId", "Time step", "in_degree", "out_degree", "directed_total_degree", "undirected_degree", "mean_neighbor_undirected_degree", "weak_component_id", "weak_component_size", "community_id", "community_size"]
        rows = list(reader)
    assert len(rows) == ids.size
    assert len({int(row["txId"]) for row in rows}) == ids.size
    assert all(all(value != "" for value in row.values()) for row in rows)
    assert np.array_equal(np.fromiter((int(row["txId"]) for row in rows), dtype=np.int64), ids)
    assert int(component_node_counts.sum()) == ids.size
    assert int(community_node_counts.sum()) == ids.size
    with component_path.open(newline="", encoding="utf-8") as stream:
        component_reader = csv.DictReader(stream)
        assert component_reader.fieldnames == ["weak_component_id", "node_count", "directed_edge_count"]
        component_rows = list(component_reader)
    with community_path.open(newline="", encoding="utf-8") as stream:
        community_reader = csv.DictReader(stream)
        assert community_reader.fieldnames == ["community_id", "node_count", "mean_undirected_degree", "internal_undirected_edges"]
        community_rows = list(community_reader)
    assert len(component_rows) == component_values.size
    assert sum(int(row["node_count"]) for row in component_rows) == ids.size
    assert sum(int(row["directed_edge_count"]) for row in component_rows) == EXPECTED_DIRECTED_EDGES
    assert len(community_rows) == community_values.size
    assert sum(int(row["node_count"]) for row in community_rows) == ids.size
    assert all(all(value != "" for value in row.values()) for row in component_rows + community_rows)
    print("DEGREE_VALIDATION=PASS node_rows=203769 missing=0 directed_degree_sums_match_edges=True")
    print(f"COMPONENT_VALIDATION=PASS components={component_values.size} component_nodes={int(component_node_counts.sum())} directed_edges={int(edge_component_ids.size)}")
    print(f"COMMUNITY_VALIDATION=PASS communities={community_values.size} assigned_nodes={int(community_node_counts.sum())} modularity={modularity:.6f} seed=2026")
    print(f"SAVED {node_path}")
    print(f"SAVED {component_path}")
    print(f"SAVED {community_path}")
    print(f"SAVED {ANALYTICS_DIR / 'graph_statistics.json'}")
    print("GRAPH_STATISTICS " + json.dumps(summary, sort_keys=True))


def read_risk_scores(ids: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Load model scores only; never read ground-truth labels for rankings."""
    expected_rows = np.flatnonzero(times >= 35)
    expected_ids = ids[expected_rows]
    expected_times = times[expected_rows]
    risk_by_node = np.full(ids.size, np.nan, dtype=np.float64)
    node_index = {int(tx_id): index for index, tx_id in enumerate(ids)}
    seen: set[int] = set()
    with RISK_PATH.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["txId", "Time step", "calibrated_illicit_risk"]:
            raise ValueError(f"Unexpected risk-score CSV schema: {reader.fieldnames}")
        for row in reader:
            tx_id, timestep = int(row["txId"]), int(row["Time step"])
            score = float(row["calibrated_illicit_risk"])
            if tx_id not in node_index:
                raise ValueError(f"Risk score references unknown transaction ID {tx_id}")
            index = node_index[tx_id]
            if tx_id in seen or timestep != int(times[index]):
                raise ValueError(f"Duplicate ID or timestep mismatch in risk CSV for {tx_id}")
            if timestep < 35 or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Invalid test timestep or risk score for transaction {tx_id}")
            seen.add(tx_id)
            risk_by_node[index] = score
    if len(seen) != expected_ids.size or not np.array_equal(np.sort(np.fromiter(seen, dtype=np.int64)), expected_ids):
        raise ValueError("Risk-score IDs do not exactly match the test-period graph nodes")
    if not np.isfinite(risk_by_node[expected_rows]).all() or np.isfinite(risk_by_node[times < 35]).any():
        raise ValueError("Risk scores are missing for test nodes or unexpectedly present in train nodes")
    return risk_by_node


def rank_suspicious_transactions(
    ids: np.ndarray,
    times: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    risk_by_node: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Rank scored nodes using their risk and scored-neighbor context available by t."""
    n = ids.size
    out_degree_asof = np.zeros(n, dtype=np.int64)
    in_degree_asof = np.zeros(n, dtype=np.int64)
    undirected_degree_asof = np.zeros(n, dtype=np.int64)
    scored_neighbor_count = np.zeros(n, dtype=np.int64)
    neighbor_sum = np.zeros(n, dtype=np.float64)
    neighbor_max = np.full(n, np.nan, dtype=np.float64)

    for timestep in range(35, 50):
        scored_at_timestep = np.flatnonzero((times == timestep) & np.isfinite(risk_by_node))
        eligible_edges = (times[src] <= timestep) & (times[dst] <= timestep)
        edge_src, edge_dst = src[eligible_edges], dst[eligible_edges]
        current_out_degree = np.bincount(edge_src, minlength=n)
        current_in_degree = np.bincount(edge_dst, minlength=n)
        low, high = np.minimum(edge_src, edge_dst), np.maximum(edge_src, edge_dst)
        undirected_pairs = np.unique(np.column_stack((low, high)), axis=0)
        left, right = undirected_pairs.T
        current_undirected_degree = np.bincount(np.concatenate((left, right)), minlength=n)

        eligible_scores = np.isfinite(risk_by_node) & (times <= timestep)
        current_neighbor_count = np.zeros(n, dtype=np.int64)
        current_neighbor_sum = np.zeros(n, dtype=np.float64)
        current_neighbor_max = np.full(n, -np.inf, dtype=np.float64)
        for node_indices, neighbor_indices in ((left, right), (right, left)):
            has_score = eligible_scores[neighbor_indices]
            scored_nodes = node_indices[has_score]
            neighbor_scores = risk_by_node[neighbor_indices[has_score]]
            np.add.at(current_neighbor_sum, scored_nodes, neighbor_scores)
            np.add.at(current_neighbor_count, scored_nodes, 1)
            np.maximum.at(current_neighbor_max, scored_nodes, neighbor_scores)
        current_neighbor_max[current_neighbor_count == 0] = np.nan
        out_degree_asof[scored_at_timestep] = current_out_degree[scored_at_timestep]
        in_degree_asof[scored_at_timestep] = current_in_degree[scored_at_timestep]
        undirected_degree_asof[scored_at_timestep] = current_undirected_degree[scored_at_timestep]
        scored_neighbor_count[scored_at_timestep] = current_neighbor_count[scored_at_timestep]
        neighbor_sum[scored_at_timestep] = current_neighbor_sum[scored_at_timestep]
        neighbor_max[scored_at_timestep] = current_neighbor_max[scored_at_timestep]

    scored_rows = np.flatnonzero(np.isfinite(risk_by_node))
    mean_neighbor_risk = np.divide(
        neighbor_sum,
        scored_neighbor_count,
        out=np.full(n, np.nan, dtype=np.float64),
        where=scored_neighbor_count > 0,
    )
    combined = risk_by_node[scored_rows].copy()
    has_neighbors = scored_neighbor_count[scored_rows] > 0
    combined[has_neighbors] = (
        0.5 * risk_by_node[scored_rows[has_neighbors]]
        + 0.5 * mean_neighbor_risk[scored_rows[has_neighbors]]
    )
    # Stable deterministic tie breaks: combined score, own model score, degree, ID.
    order = np.lexsort((ids[scored_rows], -undirected_degree_asof[scored_rows],
                        -risk_by_node[scored_rows], -combined))
    rank = np.empty(scored_rows.size, dtype=np.int64)
    rank[order] = np.arange(1, scored_rows.size + 1)
    return {
        "rows": scored_rows,
        "in_degree": in_degree_asof[scored_rows],
        "out_degree": out_degree_asof[scored_rows],
        "undirected_degree": undirected_degree_asof[scored_rows],
        "neighbor_count": scored_neighbor_count[scored_rows],
        "neighbor_mean": mean_neighbor_risk[scored_rows],
        "neighbor_max": neighbor_max[scored_rows],
        "combined_score": combined,
        "rank": rank,
    }, order


def write_suspicious_outputs(
    ids: np.ndarray,
    times: np.ndarray,
    risk_by_node: np.ndarray,
    ranked: dict[str, np.ndarray],
    order: np.ndarray,
) -> None:
    output = ANALYTICS_DIR / "suspicious_transactions.csv"
    rows = ranked["rows"][order]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("rank", "txId", "Time step", "calibrated_illicit_risk",
                         "in_degree_asof", "out_degree_asof", "directed_total_degree_asof",
                         "undirected_degree_asof", "scored_neighbor_count_asof",
                         "mean_scored_neighbor_risk_asof", "max_scored_neighbor_risk_asof",
                         "structural_suspicion_rank_score"))
        for index in order:
            row = int(ranked["rows"][index])
            mean_value = ranked["neighbor_mean"][index]
            max_value = ranked["neighbor_max"][index]
            writer.writerow((int(ranked["rank"][index]), int(ids[row]), int(times[row]),
                             f"{risk_by_node[row]:.8f}", int(ranked["in_degree"][index]),
                             int(ranked["out_degree"][index]),
                             int(ranked["in_degree"][index] + ranked["out_degree"][index]),
                             int(ranked["undirected_degree"][index]),
                             int(ranked["neighbor_count"][index]),
                             "" if not np.isfinite(mean_value) else f"{mean_value:.8f}",
                             "" if not np.isfinite(max_value) else f"{max_value:.8f}",
                             f"{ranked['combined_score'][index]:.8f}"))
    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected_header = ["rank", "txId", "Time step", "calibrated_illicit_risk",
                           "in_degree_asof", "out_degree_asof", "directed_total_degree_asof",
                           "undirected_degree_asof", "scored_neighbor_count_asof",
                           "mean_scored_neighbor_risk_asof", "max_scored_neighbor_risk_asof",
                           "structural_suspicion_rank_score"]
        assert reader.fieldnames == expected_header
        output_rows = list(reader)
    assert len(output_rows) == 67_504
    assert [int(row["rank"]) for row in output_rows] == list(range(1, len(output_rows) + 1))
    assert len({int(row["txId"]) for row in output_rows}) == len(output_rows)
    assert all(35 <= int(row["Time step"]) <= 49 for row in output_rows)
    assert all(0 <= float(row["calibrated_illicit_risk"]) <= 1 for row in output_rows)
    assert all(0 <= float(row["structural_suspicion_rank_score"]) <= 1 for row in output_rows)
    assert all((int(row["scored_neighbor_count_asof"]) > 0) == bool(row["mean_scored_neighbor_risk_asof"]) for row in output_rows)
    assert all((int(row["scored_neighbor_count_asof"]) > 0) == bool(row["max_scored_neighbor_risk_asof"]) for row in output_rows)
    report_path = ANALYTICS_DIR / "graph_statistics.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["suspicious_connection_analysis"] = {
        "output_rows": len(output_rows),
        "score_source": "existing calibrated GRU risk scores; ground-truth labels not read",
        "temporal_context": "directed/undirected edges and scored neighbors with timestep <= the scored transaction timestep",
        "combined_score": "0.5 * own calibrated illicit score + 0.5 * mean available scored-neighbor risk; own score alone if no scored neighbor exists",
        "rank_ties": "own model score descending, as-of undirected degree descending, txId ascending",
        "interpretation": "ranking heuristic, not a calibrated probability or causal explanation",
        "threshold_tuned": False,
    }
    report["total_analytics_runtime_seconds"] = time.perf_counter() - ANALYTICS_STARTED
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SUSPICIOUS_VALIDATION=PASS rows=67504 duplicate_ids=0 missing_scores=0 labels_loaded=False deterministic_tie_break=True")
    print(f"SAVED {output}")


def main() -> None:
    global ANALYTICS_STARTED
    ANALYTICS_STARTED = time.perf_counter()
    started = time.perf_counter()
    ids, times, src, dst, _ = read_inputs()
    cross_timestep_edges = int(np.sum(times[src] != times[dst]))
    in_degree, out_degree, undirected_degree, mean_neighbor_degree, undirected_pairs = degree_statistics(
        ids.size, src, dst
    )
    component_ids, component_sizes, edge_component_ids = weak_components(ids, undirected_pairs, src, dst)
    validate_community_determinism()
    community_ids, unique_community_ids, unique_community_sizes, modularity = louvain_communities(ids, undirected_pairs)
    community_sizes = unique_community_sizes[
        np.searchsorted(unique_community_ids, community_ids)
    ]
    validate_degree_statistics(
        in_degree, out_degree, undirected_degree, mean_neighbor_degree,
        src.size, undirected_pairs.shape[0],
    )
    write_statistics(
        ids, times, in_degree, out_degree, undirected_degree, mean_neighbor_degree,
        undirected_pairs, component_ids, component_sizes, edge_component_ids,
        community_ids, community_sizes, modularity,
        cross_timestep_edges,
        time.perf_counter() - started,
    )
    risk_by_node = read_risk_scores(ids, times)
    ranked, order = rank_suspicious_transactions(ids, times, src, dst, risk_by_node)
    ranked_repeat, order_repeat = rank_suspicious_transactions(ids, times, src, dst, risk_by_node)
    assert np.array_equal(order, order_repeat)
    for name in ranked:
        assert np.allclose(ranked[name], ranked_repeat[name], equal_nan=True)
    write_suspicious_outputs(ids, times, risk_by_node, ranked, order)


if __name__ == "__main__":
    main()
