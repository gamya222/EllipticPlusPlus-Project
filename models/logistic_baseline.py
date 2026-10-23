"""Small PyTorch Logistic Regression reference on the Elliptic features."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.gcn.data import FeatureTransform, load_graph_data
from models.gcn.train import metrics


ARTIFACT_DIR = Path(__file__).resolve().parent / "logistic_baseline" / "artifacts"
VAL_START, FIT_END, TEST_START, TEST_END = 31, 34, 35, 49


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_epochs(x: np.ndarray, labels: np.ndarray, times: np.ndarray, seed: int):
    """Fit through 30 and select a fixed-threshold epoch on timesteps 31–34."""
    train_rows = np.flatnonzero((times <= 30) & (labels >= 0))
    val_rows = np.flatnonzero((times >= VAL_START) & (times <= FIT_END) & (labels >= 0))
    y_train = torch.as_tensor(labels[train_rows], dtype=torch.long)
    counts = torch.bincount(y_train, minlength=2).to(torch.float32)
    weights = counts.sum() / (2 * counts)
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_val = labels[val_rows].astype(np.int8)
    set_seed(seed)
    model = nn.Linear(x.shape[1], 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    logs = []
    for epoch in range(1, 61):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor[train_rows])
        loss = F.cross_entropy(logits, y_train, weight=weights)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_scores = torch.softmax(model(x_tensor[val_rows]), dim=1)[:, 1].numpy()
        val_pred = (val_scores >= 0.5).astype(np.int8)
        val_result = metrics(y_val, val_pred, val_scores)
        score = float(val_result["macro_f1"])
        logs.append({"epoch": epoch, "train_loss": float(loss.detach()), "validation_macro_f1": score})
        print(f"epoch={epoch:03d} loss={loss.item():.6f} val_macro_f1={score:.6f}")
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 10:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
                break
    if best_state is None:
        raise RuntimeError("Validation did not select a logistic baseline epoch")
    return best_epoch, best_score, best_state, logs, train_rows, val_rows


def main() -> None:
    started = time.perf_counter()
    seed = 2026
    set_seed(seed)
    data = load_graph_data()
    transform = FeatureTransform.fit(data.raw_features, data.times, VAL_START - 1)
    x = transform.transform(data.raw_features)
    best_epoch, best_score, _, val_logs, train_rows, val_rows = select_epochs(
        x, data.labels, data.times, seed
    )

    # Final fit uses only labeled training-period rows; the feature transform
    # is independently fitted on timesteps 1–34, never on test statistics.
    final_transform = FeatureTransform.fit(data.raw_features, data.times, FIT_END)
    final_x = torch.as_tensor(final_transform.transform(data.raw_features), dtype=torch.float32)
    final_rows = np.flatnonzero((data.times <= FIT_END) & (data.labels >= 0))
    final_y = torch.as_tensor(data.labels[final_rows], dtype=torch.long)
    counts = torch.bincount(final_y, minlength=2).to(torch.float32)
    weights = counts.sum() / (2 * counts)
    set_seed(seed)
    model = nn.Linear(final_x.shape[1], 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    fit_logs = []
    for epoch in range(1, best_epoch + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(final_x[final_rows]), final_y, weight=weights)
        loss.backward()
        optimizer.step()
        fit_logs.append({"epoch": epoch, "train_loss": float(loss)})
        print(f"FINAL_TRAIN epoch={epoch:03d}/{best_epoch:03d} loss={loss.item():.6f}")

    test_rows = np.flatnonzero((data.times >= TEST_START) & (data.times <= TEST_END) & (data.labels >= 0))
    with torch.no_grad():
        test_scores = torch.softmax(model(final_x[test_rows]), dim=1)[:, 1].numpy()
    test_pred = (test_scores >= 0.5).astype(np.int8)
    result = metrics(data.labels[test_rows].astype(np.int8), test_pred, test_scores)
    print("TEST_METRICS " + json.dumps(result, sort_keys=True))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": seed,
        "model": "single linear layer (binary Logistic Regression)",
        "fit_timesteps_for_selection": [1, 30],
        "validation_timesteps": [31, 34],
        "final_fit_timesteps": [1, 34],
        "test_timesteps": [35, 49],
        "selection_metric": "validation macro F1 at fixed probability threshold 0.5",
        "loss": "class-weighted cross entropy",
        "optimizer": "Adam lr=0.01 weight_decay=0.0005",
        "feature_transform": "training-only median imputation, median/IQR scaling, then signed log1p",
    }
    torch.save({
        "state_dict": model.state_dict(), "feature_names": data.feature_names,
        "feature_medians": final_transform.medians, "feature_centers": final_transform.centers,
        "feature_scales": final_transform.scales, "best_epoch": best_epoch,
        "seed": seed, "config": config,
    }, ARTIFACT_DIR / "logistic_baseline.pt")
    report = {
        "config": config, "epochs_selected": best_epoch,
        "validation": {"best_macro_f1": best_score, "epoch_logs": val_logs},
        "final_fit_epoch_logs": fit_logs, "test_timesteps": [TEST_START, TEST_END],
        "test": result, "runtime_seconds": time.perf_counter() - started,
        "device": "cpu", "validation_rows": int(val_rows.size),
        "selection_train_rows": int(train_rows.size),
    }
    (ARTIFACT_DIR / "logistic_baseline_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"RUNTIME_SECONDS={report['runtime_seconds']:.3f}")
    print(f"SAVED {ARTIFACT_DIR / 'logistic_baseline.pt'}")
    print(f"SAVED {ARTIFACT_DIR / 'logistic_baseline_metrics.json'}")


if __name__ == "__main__":
    main()
