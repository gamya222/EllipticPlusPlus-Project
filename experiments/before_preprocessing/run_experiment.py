"""Run the isolated original-scale Elliptic++ model comparison."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.gcn.model import GCN
from models.graphsage.model import GraphSAGE
from models.gru.model import TemporalGRU


RAW_DIR = ROOT / "data" / "raw" / "elliptic"
OUTPUT_DIR = ROOT / "evaluation_metrics" / "before_preprocessing"
TRAIN_END, VALIDATION_START, FINAL_TRAIN_END, TEST_START, TEST_END = 30, 31, 34, 35, 49
EXPECTED_NODES, EXPECTED_EDGES, FEATURE_COUNT = 203_769, 234_355, 182
EXPECTED_TEST_UNKNOWN = 50_834


@dataclass
class RawData:
    ids: np.ndarray
    times: np.ndarray
    labels: np.ndarray
    raw_features: np.ndarray
    feature_names: list[str]
    source_edges: np.ndarray
    target_edges: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_raw_data() -> RawData:
    """Load immutable source CSVs and validate an exact ID-based label join."""
    feature_path = RAW_DIR / "txs_features.csv"
    class_path = RAW_DIR / "txs_classes.csv"
    edge_path = RAW_DIR / "txs_edgelist.csv"
    features = pd.read_csv(feature_path, dtype={"txId": "int64", "Time step": "int16"}, low_memory=False)
    features = features.copy()  # compact the wide CSV frame before the validated label join
    classes = pd.read_csv(class_path, dtype={"txId": "int64", "class": "int8"})
    if not features.txId.is_unique or not classes.txId.is_unique:
        raise ValueError("Raw transaction or class input contains duplicate IDs")
    joined = features.merge(classes, on="txId", how="left", validate="one_to_one", indicator=True, sort=False)
    if len(joined) != len(features) or not (joined["_merge"] == "both").all():
        missing = int((joined["_merge"] != "both").sum())
        raise ValueError(f"Feature/class join changed row count or missed labels: unmatched={missing}")
    joined = joined.drop(columns="_merge").sort_values("txId", kind="stable").reset_index(drop=True)
    feature_names = [c for c in features.columns if c not in {"txId", "Time step"}]
    if len(feature_names) != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} numeric input features, found {len(feature_names)}")
    ids = joined.txId.to_numpy(dtype=np.int64)
    times = joined["Time step"].to_numpy(dtype=np.int16)
    source_classes = joined["class"].to_numpy(dtype=np.int8)
    if not np.isin(source_classes, [1, 2, 3]).all():
        raise ValueError("Raw classes must be 1 (illicit), 2 (licit), or 3 (unknown)")
    labels = np.full(source_classes.shape, -1, dtype=np.int8)
    labels[source_classes == 1] = 1
    labels[source_classes == 2] = 0
    raw = joined[feature_names].to_numpy(dtype=np.float32, na_value=np.nan)

    edge_frame = pd.read_csv(edge_path, dtype="int64")
    edge_ids = edge_frame.to_numpy(dtype=np.int64)
    src = np.searchsorted(ids, edge_ids[:, 0])
    dst = np.searchsorted(ids, edge_ids[:, 1])
    if (src >= ids.size).any() or (dst >= ids.size).any():
        raise ValueError("An edge endpoint does not exist in the raw transaction table")
    if not np.array_equal(ids[src], edge_ids[:, 0]) or not np.array_equal(ids[dst], edge_ids[:, 1]):
        raise ValueError("An edge endpoint does not exist in the raw transaction table")
    if np.any(src == dst) or np.unique(np.column_stack((src, dst)), axis=0).shape[0] != edge_ids.shape[0]:
        raise ValueError("Raw edge file contains self-loops or duplicate directed edges")
    if ids.size != EXPECTED_NODES or edge_ids.shape[0] != EXPECTED_EDGES:
        raise ValueError(f"Raw totals mismatch: nodes={ids.size}, edges={edge_ids.shape[0]}")
    if not np.isin(times, np.arange(1, 50)).all():
        raise ValueError("Unexpected timestep in source feature matrix")
    if np.intersect1d(ids[times <= FINAL_TRAIN_END], ids[times >= TEST_START]).size:
        raise ValueError("Transaction ID overlap between train and test periods")
    return RawData(ids, times, labels, raw, feature_names, src, dst)


def fit_train_medians(raw: np.ndarray, times: np.ndarray, through: int) -> np.ndarray:
    fit_rows = times <= through
    if not fit_rows.any():
        raise ValueError(f"No observations available to fit medians through timestep {through}")
    values = raw[fit_rows].astype(np.float64, copy=False)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
    if not np.isfinite(medians).all():
        bad = np.flatnonzero(~np.isfinite(medians)).tolist()
        raise ValueError(f"Features lack finite training observations: {bad}")
    return medians


def impute_only(raw: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Apply train-only median imputation; intentionally no scaling/log transform."""
    result = np.where(np.isfinite(raw), raw, medians).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("Median-imputed original-scale features contain non-finite values")
    return result


def graph_prefix(data: RawData, cutoff: int, device: torch.device, sage: bool = False):
    nodes = np.flatnonzero(data.times <= cutoff)
    local = np.full(data.ids.size, -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size, dtype=np.int64)
    src, dst = local[data.source_edges], local[data.target_edges]
    keep = (src >= 0) & (dst >= 0)
    src, dst = src[keep], dst[keep]
    if sage:
        rows = np.concatenate((src, dst))
        cols = np.concatenate((dst, src))
    else:
        loop = np.arange(nodes.size, dtype=np.int64)
        rows = np.concatenate((src, dst, loop))
        cols = np.concatenate((dst, src, loop))
    idx = torch.from_numpy(np.stack((rows, cols))).to(torch.long)
    adj = torch.sparse_coo_tensor(idx, torch.ones(rows.size), (nodes.size, nodes.size)).coalesce()
    idx = adj.indices()
    degree = torch.bincount(idx[0], minlength=nodes.size).to(torch.float32)
    if sage:
        weights = degree[idx[0]].clamp_min(1).reciprocal()
    else:
        inv = degree.clamp_min(1).rsqrt()
        weights = inv[idx[0]] * inv[idx[1]]
    adj = torch.sparse_coo_tensor(idx, weights, (nodes.size, nodes.size)).coalesce().to(device)
    return nodes, adj


def make_summaries(features: np.ndarray, times: np.ndarray) -> np.ndarray:
    summaries = np.empty((49, features.shape[1]), dtype=np.float32)
    for step in range(1, 50):
        selected = times == step
        if not selected.any():
            raise ValueError(f"No transactions at timestep {step}")
        summaries[step - 1] = features[selected].mean(axis=0)
    if not np.isfinite(summaries).all():
        raise ValueError("Timestep summaries contain non-finite values")
    return summaries


def class_weights(labels: np.ndarray, rows: np.ndarray, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    y = labels[rows].astype(np.int64)
    counts = np.bincount(y, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Training period must contain both classes, got {counts.tolist()}")
    weights = torch.as_tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    return weights, counts.astype(int).tolist()


def calculate_metrics(y_true: np.ndarray, probability: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    y_pred = (probability >= 0.5).astype(np.int8)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    def div(n: float, d: float) -> float:
        return float(n / d) if d else 0.0
    values = {
        "accuracy": div(tn + tp, tn + fp + fn + tp),
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "fpr": div(fp, fp + tn),
        "fnr": div(fn, fn + tp),
        "error_rate": div(fp + fn, tn + fp + fn + tp),
        "support": {"licit": tn + fp, "illicit": fn + tp},
        "count": int(y_true.size),
        "threshold": 0.5,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    for key, value in values.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"Metric {key} is non-finite")
    return values, cm, y_pred


def macro_f1(y_true: np.ndarray, probability: np.ndarray) -> float:
    pred = (probability >= 0.5).astype(np.int8)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return float((f1_pos + f1_neg) / 2)


def evaluate_graph(model, data, features, graphs, first: int, last: int, device):
    model.eval()
    ids, steps, truths, scores = [], [], [], []
    with torch.no_grad():
        for step in range(first, last + 1):
            nodes, adj = graphs[step]
            target_rows = np.flatnonzero((data.times == step) & (data.labels >= 0))
            if not target_rows.size:
                continue
            local = np.full(data.ids.size, -1, dtype=np.int64)
            local[nodes] = np.arange(nodes.size, dtype=np.int64)
            logits = model(features[nodes], adj)
            local_rows = torch.as_tensor(local[target_rows], dtype=torch.long, device=device)
            prob = torch.softmax(logits[local_rows], dim=1)[:, 1].cpu().numpy().astype(np.float32)
            if not np.isfinite(prob).all():
                raise FloatingPointError(f"Non-finite scores at timestep {step}")
            ids.append(data.ids[target_rows]); steps.append(data.times[target_rows])
            truths.append(data.labels[target_rows]); scores.append(prob)
            del logits, local_rows
    if not truths:
        raise ValueError(f"No labeled transactions in timesteps {first}-{last}")
    return tuple(np.concatenate(parts) for parts in (ids, steps, truths, scores))


def evaluate_node(model, data, features, rows: np.ndarray):
    model.eval()
    with torch.no_grad():
        score = torch.softmax(model(features[rows]), dim=1)[:, 1].cpu().numpy().astype(np.float32)
    if not np.isfinite(score).all():
        raise FloatingPointError("Node classifier produced non-finite probabilities")
    return data.ids[rows], data.times[rows], data.labels[rows], score


def assert_finite_training(model, loss: torch.Tensor) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError("Training loss became non-finite on original-scale features")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite gradient in {name}; raw-scale run cannot continue safely")


def fit_graph_model(name: str, model_type, sage: bool, data: RawData, args, device):
    started = time.perf_counter()
    med30 = fit_train_medians(data.raw_features, data.times, TRAIN_END)
    x30 = torch.from_numpy(impute_only(data.raw_features, med30)).to(device)
    train_nodes, train_adj = graph_prefix(data, TRAIN_END, device, sage)
    train_rows = train_nodes[data.labels[train_nodes] >= 0]
    train_map = np.full(data.ids.size, -1, dtype=np.int64); train_map[train_nodes] = np.arange(train_nodes.size)
    train_local = torch.as_tensor(train_map[train_rows], dtype=torch.long, device=device)
    train_y = torch.as_tensor(data.labels[train_rows].astype(np.int64), dtype=torch.long, device=device)
    weights, train_counts = class_weights(data.labels, train_rows, device)
    val_graphs = {s: graph_prefix(data, s, device, sage) for s in range(VALIDATION_START, 35)}
    set_seed(args.seed)
    model = model_type(FEATURE_COUNT, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    epoch_logs = []
    for epoch in range(1, args.max_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(x30[train_nodes], train_adj)
        loss = F.cross_entropy(logits[train_local], train_y, weight=weights)
        assert_finite_training(model, loss); optimizer.step()
        train_loss = float(loss.detach().cpu())
        del logits, loss
        _, _, y_val, p_val = evaluate_graph(model, data, x30, val_graphs, 31, 34, device)
        score = macro_f1(y_val, p_val)
        epoch_logs.append({"epoch": epoch, "train_loss": train_loss, "validation_macro_f1": score})
        print(f"{name} epoch={epoch:03d} train_loss={train_loss:.6f} val_macro_f1={score:.6f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"{name} early_stop epoch={epoch} selected={best_epoch}", flush=True); break
    if best_state is None:
        raise RuntimeError(f"{name} validation selected no checkpoint")

    med34 = fit_train_medians(data.raw_features, data.times, FINAL_TRAIN_END)
    x34 = torch.from_numpy(impute_only(data.raw_features, med34)).to(device)
    final_nodes, final_adj = graph_prefix(data, FINAL_TRAIN_END, device, sage)
    final_rows = final_nodes[data.labels[final_nodes] >= 0]
    final_map = np.full(data.ids.size, -1, dtype=np.int64); final_map[final_nodes] = np.arange(final_nodes.size)
    final_local = torch.as_tensor(final_map[final_rows], dtype=torch.long, device=device)
    final_y = torch.as_tensor(data.labels[final_rows].astype(np.int64), dtype=torch.long, device=device)
    final_weights, final_counts = class_weights(data.labels, final_rows, device)
    set_seed(args.seed)
    final_model = model_type(FEATURE_COUNT, args.hidden_dim, args.dropout).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=0.01, weight_decay=5e-4)
    final_logs = []
    for epoch in range(1, best_epoch + 1):
        final_model.train(); final_optimizer.zero_grad(set_to_none=True)
        logits = final_model(x34[final_nodes], final_adj)
        loss = F.cross_entropy(logits[final_local], final_y, weight=final_weights)
        assert_finite_training(final_model, loss); final_optimizer.step()
        final_loss = float(loss.detach().cpu()); del logits, loss
        final_logs.append({"epoch": epoch, "train_loss": final_loss})
        print(f"{name} final_train={epoch:03d}/{best_epoch:03d} loss={final_loss:.6f}", flush=True)
    test_graphs = {s: graph_prefix(data, s, device, sage) for s in range(TEST_START, TEST_END + 1)}
    ids, steps, y, p = evaluate_graph(final_model, data, x34, test_graphs, TEST_START, TEST_END, device)
    result = make_result(name, ids, steps, y, p, best_epoch, best_score, epoch_logs, final_logs, train_counts, final_counts, med30, med34, args, device, time.perf_counter() - started)
    return result, {"state_dict": final_model.state_dict(), "feature_names": data.feature_names, "feature_medians": med34, "config": result[0]["config"], "best_epoch": best_epoch}


def make_result(name, ids, steps, y, p, best_epoch, val_score, logs, final_logs, train_counts, final_counts, med30, med34, args, device, runtime):
    values, cm, pred = calculate_metrics(y, p)
    return ({
        "schema_version": "elliptic-before-preprocessing-v1",
        "model": name,
        "protocol": "Before Preprocessing - Original-Scale Features with Minimal Missing-Value Imputation",
        "dataset": {"source": "data/raw/elliptic/txs_features.csv + txs_classes.csv + txs_edgelist.csv", "transactions": EXPECTED_NODES, "edges": EXPECTED_EDGES, "feature_count": FEATURE_COUNT, "missing_cells": 16405, "missing_feature_count": 17, "missing_per_feature": 965},
        "preprocessing": {"features": "original scale; no median/IQR scaling; no signed log1p", "missing_values": "feature-wise median imputation; fitted using timesteps 1-30 during selection and timesteps 1-34 during final fit", "imputation_fit_timesteps": {"selection": [1, TRAIN_END], "final_fit": [1, FINAL_TRAIN_END]}},
        "config": {"seed": args.seed, "hidden_dim": args.hidden_dim, "dropout": args.dropout, "max_epochs": args.max_epochs, "patience": args.patience, "learning_rate": 0.01, "weight_decay": 5e-4, "loss": "class-weighted cross entropy", "optimizer": "Adam", "selection_metric": "validation macro F1 at probability threshold 0.5", "train_timesteps": [1, TRAIN_END], "validation_timesteps": [VALIDATION_START, FINAL_TRAIN_END], "final_train_timesteps": [1, FINAL_TRAIN_END], "test_timesteps": [TEST_START, TEST_END]},
        "validation": {"selected_epoch": best_epoch, "selected_macro_f1": val_score, "epoch_logs": logs},
        "final_training": {"epoch_logs": final_logs, "train_class_counts_licit_illicit": final_counts},
        "test": values,
        "test_population": {"labeled_count": int(y.size), "unknown_excluded": EXPECTED_TEST_UNKNOWN, "timesteps": [int(steps.min()), int(steps.max())], "positive_class": "source class 1 (illicit); encoded 1", "class_1_illicit": int(np.sum(y == 1)), "class_2_licit": int(np.sum(y == 0)), "transaction_ids_unique": bool(np.unique(ids).size == ids.size)},
        "device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "runtime_seconds": runtime,
        "confusion_matrix_licit_illicit": cm.tolist(),
        "feature_medians_selection": med30.tolist(), "feature_medians_final": med34.tolist(),
    }, {"txId": ids, "timestep": steps, "label": y, "probability": p, "prediction": pred})


def fit_node_model(name: str, data: RawData, args):
    started = time.perf_counter()
    med30 = fit_train_medians(data.raw_features, data.times, TRAIN_END)
    x30 = torch.from_numpy(impute_only(data.raw_features, med30))
    train_rows = np.flatnonzero((data.times <= TRAIN_END) & (data.labels >= 0))
    val_rows = np.flatnonzero((data.times >= VALIDATION_START) & (data.times <= FINAL_TRAIN_END) & (data.labels >= 0))
    train_y = torch.as_tensor(data.labels[train_rows].astype(np.int64), dtype=torch.long)
    weights, train_counts = class_weights(data.labels, train_rows, torch.device("cpu"))
    set_seed(args.seed)
    model = nn.Linear(FEATURE_COUNT, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best, best_epoch, best_state, stale, logs = -1.0, 0, None, 0, []
    for epoch in range(1, args.max_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(x30[train_rows]); loss = F.cross_entropy(logits, train_y, weight=weights)
        assert_finite_training(model, loss); optimizer.step()
        pval = torch.softmax(model(x30[val_rows]), dim=1)[:, 1].detach().numpy()
        score = macro_f1(data.labels[val_rows], pval)
        logs.append({"epoch": epoch, "train_loss": float(loss.detach()), "validation_macro_f1": score})
        print(f"{name} epoch={epoch:03d} train_loss={loss.item():.6f} val_macro_f1={score:.6f}", flush=True)
        if score > best:
            best, best_epoch, best_state, stale = score, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"{name} early_stop epoch={epoch} selected={best_epoch}", flush=True); break
    if best_state is None:
        raise RuntimeError("Logistic validation selected no checkpoint")
    med34 = fit_train_medians(data.raw_features, data.times, FINAL_TRAIN_END)
    x34 = torch.from_numpy(impute_only(data.raw_features, med34))
    final_rows = np.flatnonzero((data.times <= FINAL_TRAIN_END) & (data.labels >= 0))
    weights_final, final_counts = class_weights(data.labels, final_rows, torch.device("cpu"))
    set_seed(args.seed); final_model = nn.Linear(FEATURE_COUNT, 2)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=0.01, weight_decay=5e-4)
    final_logs = []
    final_y = torch.as_tensor(data.labels[final_rows].astype(np.int64), dtype=torch.long)
    for epoch in range(1, best_epoch + 1):
        final_model.train(); final_optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(final_model(x34[final_rows]), final_y, weight=weights_final)
        assert_finite_training(final_model, loss); final_optimizer.step()
        final_logs.append({"epoch": epoch, "train_loss": float(loss.detach())})
    test_rows = np.flatnonzero((data.times >= TEST_START) & (data.times <= TEST_END) & (data.labels >= 0))
    ids, steps, y, p = evaluate_node(final_model, data, x34, test_rows)
    result = make_result(name, ids, steps, y, p, best_epoch, best, logs, final_logs, train_counts, final_counts, med30, med34, args, torch.device("cpu"), time.perf_counter() - started)
    result[0]["config"]["model"] = "single linear layer (binary Logistic Regression)"
    result[0]["config"]["hidden_dim"] = None; result[0]["config"]["dropout"] = None
    return result, {"state_dict": final_model.state_dict(), "feature_names": data.feature_names, "feature_medians": med34, "config": result[0]["config"], "best_epoch": best_epoch}


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    design = np.column_stack((logits.astype(np.float64), np.ones(logits.size)))
    target = labels.astype(np.float64); coefficients = np.array([1.0, 0.0]); penalty = np.diag([1e-6, 1e-6])
    for _ in range(100):
        raw = np.clip(design @ coefficients, -40, 40); probs = 1 / (1 + np.exp(-raw))
        grad = design.T @ (probs - target) + penalty @ coefficients
        curvature = probs * (1 - probs); hessian = (design.T * curvature) @ design + penalty
        delta = np.linalg.solve(hessian, grad); coefficients -= delta
        if np.linalg.norm(delta) < 1e-8: break
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("Validation-only Platt calibration diverged on the raw-scale GRU")
    return float(coefficients[0]), float(coefficients[1])


def validation_gru_and_risk(data: RawData, args, device: torch.device):
    """Train selection GRU once; fit calibration on validation labels and score test."""
    started = time.perf_counter(); med30 = fit_train_medians(data.raw_features, data.times, TRAIN_END)
    x_np = impute_only(data.raw_features, med30); summaries_np = make_summaries(x_np, data.times)
    x = torch.from_numpy(x_np).to(device); summaries = torch.from_numpy(summaries_np).to(device)
    times_t = torch.from_numpy(data.times.astype(np.int64) - 1).to(device)
    train_rows = np.flatnonzero((data.times <= TRAIN_END) & (data.labels >= 0))
    val_rows = np.flatnonzero((data.times >= VALIDATION_START) & (data.times <= FINAL_TRAIN_END) & (data.labels >= 0))
    weights, train_counts = class_weights(data.labels, train_rows, device)
    set_seed(args.seed); model = TemporalGRU(FEATURE_COUNT, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best, best_epoch, best_state, stale, logs = -1.0, 0, None, 0, []
    for epoch in range(1, args.max_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(x, times_t, summaries)
        loss = F.cross_entropy(logits[train_rows], torch.as_tensor(data.labels[train_rows].astype(np.int64), device=device), weight=weights)
        assert_finite_training(model, loss); optimizer.step()
        with torch.no_grad():
            vp = torch.softmax(model(x, times_t, summaries)[val_rows], dim=1)[:, 1].cpu().numpy()
        score = macro_f1(data.labels[val_rows], vp)
        logs.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()), "validation_macro_f1": score})
        print(f"GRU epoch={epoch:03d} train_loss={loss.item():.6f} val_macro_f1={score:.6f}", flush=True)
        if score > best:
            best, best_epoch, best_state, stale = score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"GRU early_stop epoch={epoch} selected={best_epoch}", flush=True); break
    if best_state is None:
        raise RuntimeError("GRU validation selected no checkpoint")
    validation_model = TemporalGRU(FEATURE_COUNT, args.hidden_dim, args.dropout).to(device)
    validation_model.load_state_dict(best_state); validation_model.eval()
    with torch.no_grad():
        logits = validation_model(x, times_t, summaries)
        raw_logits = (logits[:, 1] - logits[:, 0]).cpu().numpy().astype(np.float64)
    slope, intercept = fit_platt(raw_logits[val_rows], data.labels[val_rows])
    test_rows = np.flatnonzero((data.times >= TEST_START) & (data.times <= TEST_END) & (data.labels >= 0))
    risk = 1 / (1 + np.exp(-np.clip(slope * raw_logits[test_rows] + intercept, -40, 40)))
    test = (data.ids[test_rows], data.times[test_rows], data.labels[test_rows], risk.astype(np.float32))
    result, arrays = make_result("calibrated_risk_scorer", *test, best_epoch, best, logs, [], train_counts, train_counts, med30, med30, args, device, time.perf_counter() - started)
    result["config"].update({"model": "validation-selected Temporal GRU plus one-dimensional Platt logistic calibration", "calibration_fit_timesteps": [VALIDATION_START, FINAL_TRAIN_END], "calibration_fit_label_count": int(val_rows.size), "calibration_uses_test_labels": False})
    result["config"]["train_timesteps"] = [1, TRAIN_END]
    result["config"]["final_train_timesteps"] = [1, TRAIN_END]
    result["preprocessing"]["missing_values"] = "feature-wise median imputation fitted on timesteps 1-30 for the validation-selected GRU, calibration, and test scoring"
    result["preprocessing"]["imputation_fit_timesteps"] = {"gru_fit": [1, TRAIN_END], "calibration_and_test_scoring": [1, TRAIN_END]}
    result["feature_medians_scoring"] = result.pop("feature_medians_final")
    result["final_training"] = {"model_fit_timesteps": [1, TRAIN_END], "train_class_counts_licit_illicit": train_counts, "epoch_logs": []}
    result["calibration"] = {"method": "regularized one-dimensional logistic/Platt scaling", "slope": slope, "intercept": intercept, "fit_timesteps": [VALIDATION_START, FINAL_TRAIN_END], "fit_rows": int(val_rows.size), "source_model": "the raw-scale GRU selected using validation macro F1; trained on timesteps 1-30"}
    result["test"] = calculate_metrics(test[2], test[3])[0]
    selection = {"state_dict": best_state, "feature_names": data.feature_names, "feature_medians": med30, "config": result["config"], "best_epoch": best_epoch, "train_counts": train_counts, "selection_score": best, "selection_logs": logs, "calibration": result["calibration"]}
    return (result, arrays), selection


def save_result(name: str, pair, checkpoint: dict, output: Path):
    report, arrays = pair
    report_path = output / f"{name}.json"
    checkpoint_dir = output / "artifacts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}.pt"
    predictions_path = checkpoint_dir / f"{name}_test_predictions.csv"
    if any(p.exists() for p in (report_path, checkpoint_path, predictions_path)):
        raise FileExistsError(f"Refusing to overwrite existing before-preprocessing artifact for {name}")
    torch.save(checkpoint, checkpoint_path)
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(("txId", "timestep", "label", "probability", "prediction"))
        writer.writerows(zip(arrays["txId"].tolist(), arrays["timestep"].tolist(), arrays["label"].tolist(), arrays["probability"].tolist(), arrays["prediction"].tolist(), strict=True))
    def display_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path.resolve())
    report["artifacts"] = {"checkpoint": display_path(checkpoint_path), "test_predictions": display_path(predictions_path)}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"SAVED {report_path.relative_to(ROOT)}", flush=True)


def smoke_test(data: RawData, args, device: torch.device) -> None:
    """Two real-data optimizer steps per architecture; writes no outputs."""
    medians = fit_train_medians(data.raw_features, data.times, TRAIN_END)
    x_np = impute_only(data.raw_features, medians)
    train_nodes = np.flatnonzero(data.times <= TRAIN_END)
    labeled = train_nodes[data.labels[train_nodes] >= 0]
    if labeled.size < 8 or np.unique(data.labels[labeled]).size != 2:
        raise ValueError("Training-period data do not contain both labeled classes for smoke test")
    # Select a deterministic class-stratified set of only pre-validation nodes.
    rng = np.random.default_rng(args.seed)
    chosen = []
    for klass, count in ((0, 512), (1, 512), (-1, 1024)):
        pool = train_nodes[data.labels[train_nodes] == klass]
        if pool.size < count:
            count = pool.size
        chosen.extend(rng.choice(pool, size=count, replace=False).tolist())
    nodes = np.array(sorted(chosen), dtype=np.int64)
    keep_map = np.full(data.ids.size, -1, dtype=np.int64); keep_map[nodes] = np.arange(nodes.size)
    src, dst = keep_map[data.source_edges], keep_map[data.target_edges]
    keep = (src >= 0) & (dst >= 0)
    smoke_data = RawData(data.ids, data.times, data.labels, data.raw_features, data.feature_names, data.source_edges[keep], data.target_edges[keep])
    # Graph functions use global indices; retain full-sized arrays but compact edges are remapped.
    smoke_data.source_edges = nodes[src[keep]]; smoke_data.target_edges = nodes[dst[keep]]
    # restrict graph models to a local view for a fast check
    local_data = RawData(data.ids[nodes], data.times[nodes], data.labels[nodes], data.raw_features[nodes], data.feature_names,
                         np.searchsorted(nodes, smoke_data.source_edges), np.searchsorted(nodes, smoke_data.target_edges))
    local_x = torch.from_numpy(x_np[nodes]).to(device)
    train_rows = np.flatnonzero((local_data.times <= TRAIN_END) & (local_data.labels >= 0))
    if np.unique(local_data.labels[train_rows]).size != 2:
        raise ValueError("Smoke subset lacks both classes; refusing to report a valid smoke test")
    for name, cls, sage in (("GCN", GCN, False), ("GraphSAGE", GraphSAGE, True)):
        set_seed(args.seed); model = cls(FEATURE_COUNT, args.hidden_dim, args.dropout).to(device)
        nodes_local, adj = graph_prefix(local_data, TRAIN_END, device, sage)
        rows_local = np.flatnonzero((local_data.times <= TRAIN_END) & (local_data.labels >= 0))
        weights, _ = class_weights(local_data.labels, rows_local, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True); z = model(local_x[nodes_local], adj)
            loss = F.cross_entropy(z[torch.as_tensor(rows_local, device=device)], torch.as_tensor(local_data.labels[rows_local].astype(np.int64), device=device), weight=weights)
            assert_finite_training(model, loss); optimizer.step()
        print(f"SMOKE {name}: 2 actual-data optimizer steps, finite loss/gradients", flush=True)
    summaries = torch.from_numpy(make_summaries(x_np, data.times)).to(device)
    times_t = torch.from_numpy(data.times.astype(np.int64) - 1).to(device)
    for name, model in (("Temporal GRU", TemporalGRU(FEATURE_COUNT, args.hidden_dim, args.dropout)), ("Logistic Regression", nn.Linear(FEATURE_COUNT, 2))):
        model = model.to(device); rows = np.flatnonzero((data.times <= TRAIN_END) & (data.labels >= 0))[:4096]
        if np.unique(data.labels[rows]).size != 2: rows = np.flatnonzero((data.times <= TRAIN_END) & (data.labels >= 0))
        weights, _ = class_weights(data.labels, rows, device); opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            if name == "Temporal GRU": z = model(torch.from_numpy(x_np).to(device), times_t, summaries)[rows]
            else: z = model(torch.from_numpy(x_np[rows]).to(device))
            loss = F.cross_entropy(z, torch.as_tensor(data.labels[rows].astype(np.int64), device=device), weight=weights)
            assert_finite_training(model, loss); opt.step()
        print(f"SMOKE {name}: 2 actual-data optimizer steps, finite loss/gradients", flush=True)
    print("SMOKE PASSED: original-scale raw CSV inputs; no artifacts written", flush=True)


def validate_test_population(name: str, pair, data: RawData) -> None:
    arrays = pair[1]
    expected_rows = np.flatnonzero((data.times >= TEST_START) & (data.times <= TEST_END) & (data.labels >= 0))
    ids = np.asarray(arrays["txId"], dtype=np.int64)
    order = np.argsort(ids)
    expected_ids = data.ids[expected_rows]
    expected_order = np.argsort(expected_ids)
    if ids.size != 16_670 or np.unique(ids).size != ids.size or not np.array_equal(ids[order], expected_ids[expected_order]):
        raise ValueError(f"{name} predictions do not cover the exact labeled held-out test ID set")
    expected_labels = data.labels[expected_rows][expected_order]
    if not np.array_equal(np.asarray(arrays["label"])[order], expected_labels):
        raise ValueError(f"{name} labels are misaligned with held-out transaction IDs")


def load_existing_result(name: str, output: Path, data: RawData):
    """Validate an existing successful result before allowing a non-destructive resume."""
    report_path = output / f"{name}.json"
    if not report_path.exists():
        return None
    artifact_dir = output / "artifacts"
    checkpoint = artifact_dir / f"{name}.pt"
    predictions = artifact_dir / f"{name}_test_predictions.csv"
    if not checkpoint.is_file() or not predictions.is_file():
        raise FileNotFoundError(f"Cannot safely resume {name}: checkpoint or prediction CSV is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "elliptic-before-preprocessing-v1":
        raise ValueError(f"Cannot safely resume {name}: unsupported report schema")
    if report.get("test", {}).get("count") != 16_670:
        raise ValueError(f"Cannot safely resume {name}: expected 16,670 labeled test predictions")
    frame = pd.read_csv(predictions, dtype={"txId": "int64", "timestep": "int16", "label": "int8"})
    expected_rows = np.flatnonzero((data.times >= TEST_START) & (data.times <= TEST_END) & (data.labels >= 0))
    expected_ids = data.ids[expected_rows]
    order = np.argsort(expected_ids)
    got_order = np.argsort(frame.txId.to_numpy())
    if len(frame) != expected_ids.size or not frame.txId.is_unique or not np.array_equal(frame.txId.to_numpy()[got_order], expected_ids[order]):
        raise ValueError(f"Cannot safely resume {name}: prediction IDs differ from the current test set")
    if not np.array_equal(frame.label.to_numpy()[got_order], data.labels[expected_rows][order]):
        raise ValueError(f"Cannot safely resume {name}: prediction labels are misaligned")
    probabilities = frame.probability.to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"Cannot safely resume {name}: probabilities are invalid")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-model outputs; never overwrite them")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    print(f"Python={__import__('sys').version.split()[0]} torch={torch.__version__} CUDA={torch.version.cuda} device={device}", flush=True)
    print(f"GPU={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'not used'}", flush=True)
    data = load_raw_data()
    missing = int((~np.isfinite(data.raw_features)).sum())
    print(f"Loaded raw CSV nodes={len(data.ids)} edges={len(data.source_edges)} features={len(data.feature_names)} missing_cells={missing}", flush=True)
    if missing != 16405:
        raise ValueError(f"Expected 16,405 documented missing feature cells, found {missing}")
    labeled_test = int(np.sum((data.times >= TEST_START) & (data.times <= TEST_END) & (data.labels >= 0)))
    unknown_test = int(np.sum((data.times >= TEST_START) & (data.times <= TEST_END) & (data.labels < 0)))
    print(f"Temporal nodes train={int(np.sum(data.times <= 34))} test={int(np.sum(data.times >= 35))}; labeled test={labeled_test}; unknown test={unknown_test}", flush=True)
    if labeled_test != 16_670 or unknown_test != EXPECTED_TEST_UNKNOWN:
        raise ValueError(f"Unexpected held-out population: labeled={labeled_test}, unknown={unknown_test}")
    if args.smoke_only:
        smoke_test(data, args, device); return
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "experiment_summary.json"
    if args.resume and summary_path.is_file():
        prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior_summary.get("completed"):
            for name in ("gcn", "graphsage", "gru", "logistic_regression", "calibrated_risk_scorer"):
                if load_existing_result(name, args.output_dir, data) is None:
                    raise ValueError(f"Completed summary is missing its {name} result")
            print("RESUME: all five completed model outputs validated; no retraining or files overwritten", flush=True)
            return
    run_started = time.perf_counter()
    results = {}
    force_cpu = args.cpu
    for name, cls, sage in (("gcn", GCN, False), ("graphsage", GraphSAGE, True)):
        existing = load_existing_result(name, args.output_dir, data) if args.resume else None
        if existing is not None:
            results[name] = existing
            print(f"RESUME {name}: validated existing result; no retraining or overwrite", flush=True)
            continue
        model_device = torch.device("cpu") if force_cpu else device
        try:
            pair, checkpoint = fit_graph_model(name, cls, sage, data, args, model_device)
        except (torch.OutOfMemoryError, torch.AcceleratorError) as exc:
            if model_device.type != "cuda" or "out of memory" not in str(exc).lower():
                raise
            print(f"FALLBACK {name}: CUDA out of memory; rerunning this model on CPU with the same settings", flush=True)
            torch.cuda.empty_cache(); force_cpu = True; model_device = torch.device("cpu")
            pair, checkpoint = fit_graph_model(name, cls, sage, data, args, model_device)
        validate_test_population(name, pair, data)
        save_result(name, pair, checkpoint, args.output_dir); results[name] = pair[0]
    gru_device = torch.device("cpu") if force_cpu else device
    try:
        risk_pair, selection_checkpoint = validation_gru_and_risk(data, args, gru_device)
    except (torch.OutOfMemoryError, torch.AcceleratorError) as exc:
        if gru_device.type != "cuda" or "out of memory" not in str(exc).lower(): raise
        print("FALLBACK GRU/risk selection: CUDA out of memory; rerunning on CPU with the same settings", flush=True)
        torch.cuda.empty_cache(); force_cpu = True; gru_device = torch.device("cpu")
        risk_pair, selection_checkpoint = validation_gru_and_risk(data, args, gru_device)
    # Keep the validation-selected raw-scale GRU separately for its validation-fit risk artifact.
    val_path = args.output_dir / "artifacts" / "gru_validation_selected.pt"
    torch.save(selection_checkpoint, val_path)
    try:
        gru_pair, gru_checkpoint = fit_final_gru(data, args, gru_device, selection_checkpoint)
    except (torch.OutOfMemoryError, torch.AcceleratorError) as exc:
        if gru_device.type != "cuda" or "out of memory" not in str(exc).lower(): raise
        print("FALLBACK final GRU: CUDA out of memory; rerunning final fit on CPU", flush=True)
        torch.cuda.empty_cache(); gru_device = torch.device("cpu")
        gru_pair, gru_checkpoint = fit_final_gru(data, args, gru_device, selection_checkpoint)
    validate_test_population("gru", gru_pair, data)
    save_result("gru", gru_pair, gru_checkpoint, args.output_dir); results["gru"] = gru_pair[0]
    validate_test_population("calibrated_risk_scorer", risk_pair, data)
    save_result("calibrated_risk_scorer", risk_pair, selection_checkpoint, args.output_dir); results["calibrated_risk_scorer"] = risk_pair[0]
    logistic_pair, logistic_checkpoint = fit_node_model("logistic_regression", data, args)
    validate_test_population("logistic_regression", logistic_pair, data)
    save_result("logistic_regression", logistic_pair, logistic_checkpoint, args.output_dir); results["logistic_regression"] = logistic_pair[0]
    summary = {"schema_version": "elliptic-before-preprocessing-run-v1", "completed": len(results) == 5, "models": list(results), "test_rows": [r["test"]["count"] for r in results.values()], "model_devices": {name: result["device"] for name, result in results.items()}, "train_timesteps": [1, FINAL_TRAIN_END], "selection_train_timesteps": [1, TRAIN_END], "validation_timesteps": [VALIDATION_START, FINAL_TRAIN_END], "test_timesteps": [TEST_START, TEST_END], "resume_process_runtime_seconds": time.perf_counter() - run_started, "cumulative_model_runtime_seconds": sum(float(r.get("runtime_seconds", 0)) for r in results.values()), "device": str(device)}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("EXPERIMENT_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


def fit_final_gru(data, args, device, validation_checkpoint):
    """Train the separate final GRU, retaining the selection GRU used by the calibrator."""
    started = time.perf_counter()
    med34 = fit_train_medians(data.raw_features, data.times, FINAL_TRAIN_END)
    x_np = impute_only(data.raw_features, med34); summaries_np = make_summaries(x_np, data.times)
    x = torch.from_numpy(x_np).to(device); summaries = torch.from_numpy(summaries_np).to(device)
    times_t = torch.from_numpy(data.times.astype(np.int64) - 1).to(device)
    train_rows = np.flatnonzero((data.times <= FINAL_TRAIN_END) & (data.labels >= 0))
    weights, counts = class_weights(data.labels, train_rows, device)
    set_seed(args.seed); model = TemporalGRU(FEATURE_COUNT, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4); logs=[]
    epochs = int(validation_checkpoint["best_epoch"])
    y = torch.as_tensor(data.labels[train_rows].astype(np.int64), device=device)
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x,times_t,summaries)[train_rows],y,weight=weights)
        assert_finite_training(model,loss); optimizer.step(); logs.append({"epoch":epoch,"train_loss":float(loss.detach().cpu())})
        print(f"gru final_train={epoch:03d}/{epochs:03d} loss={loss.item():.6f}",flush=True)
    test_rows=np.flatnonzero((data.times>=TEST_START)&(data.times<=TEST_END)&(data.labels>=0))
    with torch.no_grad(): p=torch.softmax(model(x,times_t,summaries)[test_rows],dim=1)[:,1].cpu().numpy().astype(np.float32)
    ids,steps,y_np,pred_score=data.ids[test_rows],data.times[test_rows],data.labels[test_rows],p
    report,arrays=make_result("gru",ids,steps,y_np,pred_score,epochs,validation_checkpoint["selection_score"],validation_checkpoint["selection_logs"],logs,validation_checkpoint["train_counts"],counts,validation_checkpoint["feature_medians"],med34,args,device,time.perf_counter()-started)
    report["validation"]={"selected_epoch":epochs,"selected_macro_f1":validation_checkpoint["selection_score"],"epoch_logs":validation_checkpoint["selection_logs"]}
    return (report,arrays),{"state_dict":model.state_dict(),"feature_names":data.feature_names,"feature_medians":med34,"config":report["config"],"best_epoch":epochs}


if __name__ == "__main__":
    main()
