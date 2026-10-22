"""Train and evaluate a reproducible temporal GCN on generated Elliptic Parquet."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import FeatureTransform, PROCESSED_ROOT, load_graph_data, prefix_graph
from .model import GCN


VALIDATION_START = 31
TRAIN_START = 1
TEST_START = 35
TEST_END = 49
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute non-interpolated average precision (step-wise PR area)."""
    if y_true.size == 0 or not np.any(y_true == 1):
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_positive = y_true[order] == 1
    true_positive = np.cumsum(sorted_positive)
    false_positive = np.cumsum(~sorted_positive)
    threshold_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), y_true.size - 1]
    precision = true_positive[threshold_ends] / (
        true_positive[threshold_ends] + false_positive[threshold_ends]
    )
    recall = true_positive[threshold_ends] / np.sum(y_true == 1)
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, object]:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_pos = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    precision_neg = tn / (tn + fn) if tn + fn else 0.0
    recall_neg = tn / (tn + fp) if tn + fp else 0.0
    f1_neg = (
        2 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if precision_neg + recall_neg
        else 0.0
    )
    return {
        "count": int(y_true.size),
        "accuracy": (tn + tp) / y_true.size if y_true.size else 0.0,
        "illicit_precision": precision,
        "illicit_recall": recall,
        "illicit_f1": f1_pos,
        "illicit_pr_auc": average_precision(y_true, y_score),
        "macro_f1": (f1_neg + f1_pos) / 2,
        "confusion_matrix_licit_illicit": [[tn, fp], [fn, tp]],
        "support": {"licit": tn + fp, "illicit": fn + tp},
    }


def evaluate_steps(
    model: GCN,
    data,
    features: torch.Tensor,
    device: torch.device,
    first_step: int,
    last_step: int,
    log_each: bool = False,
    graph_cache: dict[int, tuple[np.ndarray, torch.Tensor]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    with torch.no_grad():
        for step in range(first_step, last_step + 1):
            global_idx, adjacency = (
                graph_cache[step]
                if graph_cache is not None
                else prefix_graph(data, step, device)
            )
            step_rows = global_idx[data.times[global_idx] == step]
            known = data.labels[step_rows] >= 0
            rows = step_rows[known]
            if not rows.size:
                continue
            local_map = np.full(data.ids.size, -1, dtype=np.int64)
            local_map[global_idx] = np.arange(global_idx.size, dtype=np.int64)
            local_target = torch.from_numpy(local_map[rows]).to(device=device, dtype=torch.long)
            logits = model(features[global_idx], adjacency)
            scores = torch.softmax(logits[local_target], dim=1)[:, 1]
            score_np = scores.cpu().numpy().astype(np.float32)
            preds = (score_np >= 0.5).astype(np.int8)
            truths = data.labels[rows].astype(np.int8)
            true_parts.append(truths)
            pred_parts.append(preds)
            score_parts.append(score_np)
            if log_each:
                result = metrics(truths, preds, score_np)
                print(
                    f"TEST timestep={step} labeled={result['count']} "
                    f"illicit_precision={result['illicit_precision']:.4f} "
                    f"illicit_recall={result['illicit_recall']:.4f} "
                    f"illicit_f1={result['illicit_f1']:.4f} "
                    f"illicit_pr_auc={result['illicit_pr_auc']:.4f} "
                    f"macro_f1={result['macro_f1']:.4f}"
                )
    if not true_parts:
        raise ValueError(f"No labeled examples in timestep range {first_step}..{last_step}")
    return (
        np.concatenate(true_parts),
        np.concatenate(pred_parts),
        np.concatenate(score_parts),
    )


def fit_model(
    data,
    feature_transform: FeatureTransform,
    train_through: int,
    epochs: int,
    seed: int,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
) -> tuple[GCN, dict[str, object], np.ndarray, FeatureTransform]:
    set_seed(seed)
    x_np = feature_transform.transform(data.raw_features)
    features = torch.from_numpy(x_np).to(device)
    global_idx, adjacency = prefix_graph(data, train_through, device)
    train_rows = global_idx[(data.times[global_idx] <= train_through) & (data.labels[global_idx] >= 0)]
    local_map = np.full(data.ids.size, -1, dtype=np.int64)
    local_map[global_idx] = np.arange(global_idx.size, dtype=np.int64)
    local_train = torch.from_numpy(local_map[train_rows]).to(device=device, dtype=torch.long)
    y_train = torch.from_numpy(data.labels[train_rows].astype(np.int64)).to(device)
    class_counts = torch.bincount(y_train, minlength=2).to(torch.float32)
    if (class_counts == 0).any():
        raise ValueError(f"Both known classes must appear in training; counts={class_counts.tolist()}")
    class_weights = (class_counts.sum() / (2 * class_counts)).to(device)

    model = GCN(len(data.feature_names), hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    validation_graphs = {
        step: prefix_graph(data, step, device)
        for step in range(VALIDATION_START, 35)
    }
    logs: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features[global_idx], adjacency)
        loss = F.cross_entropy(logits[local_train], y_train, weight=class_weights)
        loss.backward()
        optimizer.step()
        logs.append({"epoch": float(epoch), "train_loss": float(loss.detach().cpu())})
    return model, {"class_counts": class_counts.cpu().tolist(), "train_log": logs}, x_np, feature_transform


def main() -> None:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"PyTorch={torch.__version__} CUDA={torch.version.cuda} device={device}")
    print(f"GPU={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'not used'}")
    print(f"Processed input root: {PROCESSED_ROOT}")

    data = load_graph_data()
    n_train = int(np.sum(data.times <= 34))
    n_test = int(np.sum(data.times >= TEST_START))
    print(
        f"Loaded nodes={data.ids.size} edges={data.source_edges.size} features={len(data.feature_names)} "
        f"train_nodes={n_train} test_nodes={n_test}"
    )
    print(
        "Classes (whole split): "
        f"train illicit={int(np.sum((data.times <= 34) & (data.labels == 1)))} "
        f"licit={int(np.sum((data.times <= 34) & (data.labels == 0)))} "
        f"unknown={int(np.sum((data.times <= 34) & (data.labels < 0)))}; "
        f"test illicit={int(np.sum((data.times >= 35) & (data.labels == 1)))} "
        f"licit={int(np.sum((data.times >= 35) & (data.labels == 0)))} "
        f"unknown={int(np.sum((data.times >= 35) & (data.labels < 0)))}"
    )
    null_counts = np.isnan(data.raw_features).sum(axis=0)
    print(
        f"Feature NULL/NaN cells={int(null_counts.sum())}; "
        f"affected columns={int(np.sum(null_counts > 0))}; "
        f"per-column missing counts={sorted(set(null_counts[null_counts > 0].tolist()))}"
    )

    # Select epochs on timesteps 31-34; transformations are fit only through 30.
    validation_transform = FeatureTransform.fit(data.raw_features, data.times, VALIDATION_START - 1)
    val_features = torch.from_numpy(validation_transform.transform(data.raw_features)).to(device)
    train_idx, train_adj = prefix_graph(data, VALIDATION_START - 1, device)
    train_rows = train_idx[data.labels[train_idx] >= 0]
    train_local_map = np.full(data.ids.size, -1, dtype=np.int64)
    train_local_map[train_idx] = np.arange(train_idx.size, dtype=np.int64)
    train_local = torch.from_numpy(train_local_map[train_rows]).to(device=device, dtype=torch.long)
    train_y = torch.from_numpy(data.labels[train_rows].astype(np.int64)).to(device)
    counts = torch.bincount(train_y, minlength=2).to(torch.float32)
    if (counts == 0).any():
        raise ValueError(f"Both known classes must appear before validation; counts={counts.tolist()}")
    weights = (counts.sum() / (2 * counts)).to(device)
    validation_graphs = {
        step: prefix_graph(data, step, device)
        for step in range(VALIDATION_START, 35)
    }
    model = GCN(len(data.feature_names), args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    validation_epoch_logs: list[dict[str, float]] = []
    print(
        f"VALIDATION fit_cutoff=30 train_labeled={len(train_rows)} "
        f"class_counts_licit_illicit={counts.to(torch.int64).tolist()} "
        f"max_epochs={args.max_epochs} patience={args.patience}"
    )
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(val_features[train_idx], train_adj)
        loss = F.cross_entropy(logits[train_local], train_y, weight=weights)
        loss.backward()
        optimizer.step()

        val_y, val_pred, val_score = evaluate_steps(
            model, data, val_features, device, VALIDATION_START, 34,
            graph_cache=validation_graphs,
        )
        val_result = metrics(val_y, val_pred, val_score)
        score = float(val_result["macro_f1"])
        validation_epoch_logs.append(
            {
                "epoch": float(epoch),
                "train_loss": float(loss.detach().cpu()),
                "validation_macro_f1": score,
                "validation_illicit_precision": float(val_result["illicit_precision"]),
                "validation_illicit_recall": float(val_result["illicit_recall"]),
                "validation_illicit_f1": float(val_result["illicit_f1"]),
                "validation_illicit_pr_auc": float(val_result["illicit_pr_auc"]),
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={loss.item():.6f} "
            f"val_macro_f1={score:.4f} "
            f"val_illicit_precision={val_result['illicit_precision']:.4f} "
            f"val_illicit_recall={val_result['illicit_recall']:.4f} "
            f"val_illicit_f1={val_result['illicit_f1']:.4f} "
            f"val_illicit_pr_auc={val_result['illicit_pr_auc']:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
                break
    if best_state is None:
        raise RuntimeError("Validation did not produce a model checkpoint")
    print(f"SELECTED best_epoch={best_epoch} val_macro_f1={best_score:.4f}")

    # Final fit uses the full documented training period only; test features do not fit transforms.
    final_transform = FeatureTransform.fit(data.raw_features, data.times, 34)
    set_seed(args.seed)
    final_x_np = final_transform.transform(data.raw_features)
    final_features = torch.from_numpy(final_x_np).to(device)
    final_idx, final_adj = prefix_graph(data, 34, device)
    final_rows = final_idx[data.labels[final_idx] >= 0]
    final_local_map = np.full(data.ids.size, -1, dtype=np.int64)
    final_local_map[final_idx] = np.arange(final_idx.size, dtype=np.int64)
    final_local = torch.from_numpy(final_local_map[final_rows]).to(device=device, dtype=torch.long)
    final_y = torch.from_numpy(data.labels[final_rows].astype(np.int64)).to(device)
    final_counts = torch.bincount(final_y, minlength=2).to(torch.float32)
    final_weights = (final_counts.sum() / (2 * final_counts)).to(device)
    print(f"FINAL_TRAIN class_counts_licit_illicit={final_counts.to(torch.int64).tolist()}")
    final_model = GCN(len(data.feature_names), args.hidden_dim, args.dropout).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=0.01, weight_decay=5e-4)
    final_fit_epoch_logs: list[dict[str, float]] = []
    for epoch in range(1, best_epoch + 1):
        final_model.train()
        final_optimizer.zero_grad(set_to_none=True)
        logits = final_model(final_features[final_idx], final_adj)
        loss = F.cross_entropy(logits[final_local], final_y, weight=final_weights)
        loss.backward()
        final_optimizer.step()
        final_fit_epoch_logs.append(
            {"epoch": float(epoch), "train_loss": float(loss.detach().cpu())}
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == best_epoch:
            print(f"FINAL_TRAIN epoch={epoch:03d}/{best_epoch:03d} loss={loss.item():.6f}")

    test_graphs = {
        step: prefix_graph(data, step, device)
        for step in range(TEST_START, TEST_END + 1)
    }
    test_y, test_pred, test_score = evaluate_steps(
        final_model, data, final_features, device, TEST_START, TEST_END,
        log_each=True, graph_cache=test_graphs,
    )
    test_metrics = metrics(test_y, test_pred, test_score)
    print("TEST_METRICS " + json.dumps(test_metrics, sort_keys=True))
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    if peak_gpu_memory_bytes is not None:
        print(f"PEAK_GPU_MEMORY_BYTES={peak_gpu_memory_bytes}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ARTIFACT_DIR / "gcn_baseline.pt"
    metrics_path = ARTIFACT_DIR / "gcn_baseline_metrics.json"
    config = {
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "validation_timesteps": [VALIDATION_START, 34],
        "train_through_timestep": 34,
        "test_timesteps": [TEST_START, TEST_END],
        "selection_metric": "validation macro F1 at fixed probability threshold 0.5",
        "feature_transform": "train-only median imputation, median/IQR scaling, then signed log1p",
    }
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "feature_names": data.feature_names,
            "feature_medians": final_transform.medians,
            "feature_centers": final_transform.centers,
            "feature_scales": final_transform.scales,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "train_through_timestep": 34,
            "config": config,
        },
        checkpoint_path,
    )
    elapsed_seconds = time.perf_counter() - run_started
    metrics_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "runtime_seconds": elapsed_seconds,
                "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
                "config": config,
                "epochs_selected": best_epoch,
                "validation": {
                    "timesteps": [VALIDATION_START, 34],
                    "selected_macro_f1": best_score,
                    "epoch_logs": validation_epoch_logs,
                },
                "final_fit_epoch_logs": final_fit_epoch_logs,
                "test_timesteps": [TEST_START, TEST_END],
                "test": test_metrics,
                "model": {"type": "2-layer GCN", "hidden_dim": args.hidden_dim, "dropout": args.dropout},
                "feature_transform": config["feature_transform"],
                "pr_auc_definition": "non-interpolated average precision",
                "temporal_graph_protocol": (
                    "At each inference timestep, only nodes and edges through that timestep are used."
                ),
                "limitations": [
                    "Aggregate feature provenance is external to this repository; same-timestep features and graph context are transductive.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"SAVED checkpoint={checkpoint_path}")
    print(f"SAVED metrics={metrics_path}")


if __name__ == "__main__":
    main()
