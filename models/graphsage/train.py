"""Train and evaluate chronological GraphSAGE on generated Elliptic Parquet."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.gcn.data import FeatureTransform, PROCESSED_ROOT, load_graph_data, prefix_mean_graph
from models.gcn.train import TEST_END, TEST_START, VALIDATION_START, evaluate_steps, metrics
from models.graphsage.model import GraphSAGE


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"PyTorch={torch.__version__} CUDA={torch.version.cuda} device={device}")
    print(f"GPU={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'not used'}")
    print(f"Processed input root: {PROCESSED_ROOT}")

    data = load_graph_data()
    print(
        f"Loaded nodes={data.ids.size} edges={data.source_edges.size} "
        f"features={len(data.feature_names)}"
    )
    transform = FeatureTransform.fit(data.raw_features, data.times, VALIDATION_START - 1)
    features = torch.from_numpy(transform.transform(data.raw_features)).to(device)
    train_idx, train_adj = prefix_mean_graph(data, VALIDATION_START - 1, device)
    train_rows = train_idx[data.labels[train_idx] >= 0]
    local_map = np.full(data.ids.size, -1, dtype=np.int64)
    local_map[train_idx] = np.arange(train_idx.size, dtype=np.int64)
    train_local = torch.from_numpy(local_map[train_rows]).to(device=device, dtype=torch.long)
    train_y = torch.from_numpy(data.labels[train_rows].astype(np.int64)).to(device)
    counts = torch.bincount(train_y, minlength=2).to(torch.float32)
    if (counts == 0).any():
        raise ValueError(f"Both known classes must appear before validation; counts={counts.tolist()}")
    weights = counts.sum() / (2 * counts)
    validation_graphs = {
        step: prefix_mean_graph(data, step, device)
        for step in range(VALIDATION_START, 35)
    }

    model = GraphSAGE(len(data.feature_names), args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    logs: list[dict[str, float]] = []
    print(
        f"VALIDATION fit_cutoff=30 train_labeled={len(train_rows)} "
        f"class_counts_licit_illicit={counts.to(torch.int64).tolist()} "
        f"max_epochs={args.max_epochs} patience={args.patience}"
    )
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features[train_idx], train_adj)
        loss = F.cross_entropy(logits[train_local], train_y, weight=weights)
        loss.backward()
        optimizer.step()

        val_y, val_pred, val_score = evaluate_steps(
            model,
            data,
            features,
            device,
            VALIDATION_START,
            34,
            graph_cache=validation_graphs,
        )
        result = metrics(val_y, val_pred, val_score)
        score = float(result["macro_f1"])
        logs.append(
            {
                "epoch": float(epoch),
                "train_loss": float(loss.detach().cpu()),
                "validation_macro_f1": score,
                "validation_illicit_precision": float(result["illicit_precision"]),
                "validation_illicit_recall": float(result["illicit_recall"]),
                "validation_illicit_f1": float(result["illicit_f1"]),
                "validation_illicit_pr_auc": float(result["illicit_pr_auc"]),
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={loss.item():.6f} "
            f"val_macro_f1={score:.4f} "
            f"val_illicit_precision={result['illicit_precision']:.4f} "
            f"val_illicit_recall={result['illicit_recall']:.4f} "
            f"val_illicit_f1={result['illicit_f1']:.4f} "
            f"val_illicit_pr_auc={result['illicit_pr_auc']:.4f}"
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
        raise RuntimeError("Validation did not produce a GraphSAGE checkpoint")
    print(f"SELECTED best_epoch={best_epoch} val_macro_f1={best_score:.4f}")

    final_transform = FeatureTransform.fit(data.raw_features, data.times, 34)
    final_features = torch.from_numpy(final_transform.transform(data.raw_features)).to(device)
    final_idx, final_adj = prefix_mean_graph(data, 34, device)
    final_rows = final_idx[data.labels[final_idx] >= 0]
    final_local_map = np.full(data.ids.size, -1, dtype=np.int64)
    final_local_map[final_idx] = np.arange(final_idx.size, dtype=np.int64)
    final_local = torch.from_numpy(final_local_map[final_rows]).to(device=device, dtype=torch.long)
    final_y = torch.from_numpy(data.labels[final_rows].astype(np.int64)).to(device)
    final_counts = torch.bincount(final_y, minlength=2).to(torch.float32)
    final_weights = final_counts.sum() / (2 * final_counts)
    set_seed(args.seed)
    final_model = GraphSAGE(len(data.feature_names), args.hidden_dim, args.dropout).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=0.01, weight_decay=5e-4)
    final_logs: list[dict[str, float]] = []
    for epoch in range(1, best_epoch + 1):
        final_model.train()
        final_optimizer.zero_grad(set_to_none=True)
        logits = final_model(final_features[final_idx], final_adj)
        loss = F.cross_entropy(logits[final_local], final_y, weight=final_weights)
        loss.backward()
        final_optimizer.step()
        final_logs.append({"epoch": float(epoch), "train_loss": float(loss.detach().cpu())})
        print(f"FINAL_TRAIN epoch={epoch:03d}/{best_epoch:03d} loss={loss.item():.6f}")

    test_graphs = {step: prefix_mean_graph(data, step, device) for step in range(TEST_START, TEST_END + 1)}
    test_y, test_pred, test_score = evaluate_steps(
        final_model,
        data,
        final_features,
        device,
        TEST_START,
        TEST_END,
        log_each=True,
        graph_cache=test_graphs,
    )
    test_result = metrics(test_y, test_pred, test_score)
    print("TEST_METRICS " + json.dumps(test_result, sort_keys=True))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
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
        "feature_transform": "train-only median imputation, median/IQR scaling, then signed log1p",
        "selection_metric": "validation macro F1 at fixed probability threshold 0.5",
    }
    checkpoint_path = ARTIFACT_DIR / "graphsage_baseline.pt"
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "feature_names": data.feature_names,
            "feature_medians": final_transform.medians,
            "feature_centers": final_transform.centers,
            "feature_scales": final_transform.scales,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "config": config,
        },
        checkpoint_path,
    )
    runtime = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    metrics_path = ARTIFACT_DIR / "graphsage_baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "runtime_seconds": runtime,
                "peak_gpu_memory_bytes": peak_memory,
                "config": config,
                "epochs_selected": best_epoch,
                "validation": {
                    "timesteps": [VALIDATION_START, 34],
                    "selected_macro_f1": best_score,
                    "epoch_logs": logs,
                },
                "final_fit_epoch_logs": final_logs,
                "test_timesteps": [TEST_START, TEST_END],
                "test": test_result,
                "pr_auc_definition": "non-interpolated average precision",
                "temporal_graph_protocol": "At each inference timestep, only nodes and edges through that timestep are used.",
                "limitations": ["Aggregate-feature provenance is external; same-timestep features and graph context are transductive."],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PEAK_GPU_MEMORY_BYTES={peak_memory}")
    print(f"RUNTIME_SECONDS={runtime:.3f}")
    print(f"SAVED checkpoint={checkpoint_path}")
    print(f"SAVED metrics={metrics_path}")


if __name__ == "__main__":
    main()
