"""Fit validation-only Platt calibration and score the held-out test period."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from models.gcn.data import FeatureTransform, PROCESSED_ROOT, load_graph_data
from models.gcn.train import TEST_END, TEST_START, VALIDATION_START, metrics
from models.gru.model import TemporalGRU
from models.gru.train import make_timestep_summaries


GRU_CHECKPOINT = Path("models/gru/artifacts/temporal_gru_baseline.pt")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
RISK_ARTIFACT = ARTIFACT_DIR / "risk_scorer.pt"


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit a regularized 1-D logistic calibration by Newton updates."""
    design = np.column_stack((logits.astype(np.float64), np.ones(logits.size)))
    target = labels.astype(np.float64)
    coefficients = np.array([1.0, 0.0], dtype=np.float64)
    penalty = np.diag([1e-6, 1e-6])
    for _ in range(100):
        raw = design @ coefficients
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))
        gradient = design.T @ (probabilities - target) + penalty @ coefficients
        curvature = probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.linalg.norm(step) < 1e-8:
            break
    if not np.isfinite(coefficients).all():
        raise ValueError("Platt calibration produced non-finite coefficients")
    return float(coefficients[0]), float(coefficients[1])


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if selected.any():
            result += float(selected.mean()) * abs(
                float(probabilities[selected].mean()) - float(y_true[selected].mean())
            )
    return result


def main() -> None:
    started = time.perf_counter()
    if not GRU_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Required validation/final GRU checkpoint not found: {GRU_CHECKPOINT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = torch.load(GRU_CHECKPOINT, map_location="cpu", weights_only=False)
    if "validation_state_dict" not in checkpoint:
        raise ValueError(
            "GRU checkpoint lacks validation-selected weights required for leakage-safe calibration; "
            "rerun the GRU trainer version that saves validation_state_dict."
        )
    config = checkpoint["config"]
    if RISK_ARTIFACT.is_file():
        scorer = torch.load(RISK_ARTIFACT, map_location="cpu", weights_only=False)
        if scorer.get("gru_checkpoint") != str(GRU_CHECKPOINT):
            raise ValueError("Risk artifact references a different GRU checkpoint")
        checkpoint_sha256 = hashlib.sha256(GRU_CHECKPOINT.read_bytes()).hexdigest()
        if scorer.get("gru_checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("Risk artifact GRU checkpoint checksum does not match")
        if scorer.get("feature_names") != checkpoint["feature_names"]:
            raise ValueError("Risk artifact feature order differs from GRU checkpoint")
        calibrator = scorer["calibrator"]
        slope = float(calibrator["slope"])
        intercept = float(calibrator["intercept"])
        saved_calibrator = json.loads(
            (ARTIFACT_DIR / "risk_predictor_metrics.json").read_text(encoding="utf-8")
        )["calibrator"]
        if not np.isclose(slope, saved_calibrator["slope"], rtol=0.0, atol=1e-12):
            raise ValueError("Risk artifact slope differs from the existing metrics report")
        if not np.isclose(intercept, saved_calibrator["intercept"], rtol=0.0, atol=1e-12):
            raise ValueError("Risk artifact intercept differs from the existing metrics report")
        transform = FeatureTransform(
            np.asarray(scorer["feature_medians"]),
            np.asarray(scorer["feature_centers"]),
            np.asarray(scorer["feature_scales"]),
        )
        if scorer["gru_config"] != checkpoint["config"]:
            raise ValueError("Risk artifact GRU configuration differs from checkpoint")
    else:
        # Compatibility path for repositories created before risk_scorer.pt.
        transform = FeatureTransform(
            np.asarray(checkpoint["validation_feature_medians"]),
            np.asarray(checkpoint["validation_feature_centers"]),
            np.asarray(checkpoint["validation_feature_scales"]),
        )
        slope = intercept = None
    model = TemporalGRU(
        len(checkpoint["feature_names"]),
        config["hidden_dim"],
        config["dropout"],
    )
    model.load_state_dict(checkpoint["validation_state_dict"])
    model.to(device).eval()

    data = load_graph_data()
    if checkpoint["feature_names"] != data.feature_names:
        raise ValueError("GRU checkpoint feature order differs from the current Parquet schema")
    features_np = transform.transform(data.raw_features)
    summaries_np = make_timestep_summaries(features_np, data.times)
    features = torch.from_numpy(features_np).to(device)
    summaries = torch.from_numpy(summaries_np).to(device)
    times = torch.from_numpy(data.times.astype(np.int64) - 1).to(device)
    with torch.no_grad():
        logits = model(features, times, summaries)
    raw_logits = (logits[:, 1] - logits[:, 0]).cpu().numpy().astype(np.float64)

    validation_rows = np.flatnonzero(
        (data.times >= VALIDATION_START) & (data.times <= 34) & (data.labels >= 0)
    )
    test_rows = np.flatnonzero(
        (data.times >= TEST_START) & (data.times <= TEST_END) & (data.labels >= 0)
    )
    all_test_rows = np.flatnonzero((data.times >= TEST_START) & (data.times <= TEST_END))
    validation_y = data.labels[validation_rows].astype(np.int8)
    test_y = data.labels[test_rows].astype(np.int8)
    if slope is None or intercept is None:
        slope, intercept = fit_platt(raw_logits[validation_rows], validation_y)
    test_risk = sigmoid(slope * raw_logits[test_rows] + intercept).astype(np.float32)
    all_test_risk = sigmoid(slope * raw_logits[all_test_rows] + intercept).astype(np.float32)
    test_pred = (test_risk >= 0.5).astype(np.int8)
    test_result = metrics(test_y, test_pred, test_risk)
    brier = float(np.mean((test_risk - test_y) ** 2))
    ece = calibration_error(test_y, test_risk)
    print(
        f"CALIBRATOR validation_timesteps={VALIDATION_START}-34 "
        f"labeled={validation_y.size} licit={int(np.sum(validation_y == 0))} "
        f"illicit={int(np.sum(validation_y == 1))} slope={slope:.6f} intercept={intercept:.6f}"
    )
    print(
        f"TEST labeled={test_y.size} licit={int(np.sum(test_y == 0))} "
        f"illicit={int(np.sum(test_y == 1))}"
    )
    print("TEST_RISK_METRICS " + json.dumps(test_result, sort_keys=True))
    print(f"TEST_BRIER_SCORE={brier:.6f} TEST_ECE_10_BINS={ece:.6f}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = ARTIFACT_DIR / "test_risk_scores.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("txId", "Time step", "calibrated_illicit_risk"))
        for row, risk in zip(all_test_rows, all_test_risk, strict=True):
            writer.writerow((int(data.ids[row]), int(data.times[row]), f"{risk:.8f}"))

    runtime = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    report = {
        "risk_score_definition": "Platt-calibrated probability of class 1 (illicit) from the validation-selected temporal GRU.",
        "difference_from_classifier": "The GRU classifier emits class logits; this component calibrates its continuous illicit probability. It adds no separate future-risk target.",
        "calibrator": {"type": "one-dimensional logistic/Platt scaling", "slope": slope, "intercept": intercept},
        "calibration_fit_timesteps": [VALIDATION_START, 34],
        "calibration_fit_counts": {
            "labeled": int(validation_y.size),
            "licit": int(np.sum(validation_y == 0)),
            "illicit": int(np.sum(validation_y == 1)),
        },
        "test_timesteps": [TEST_START, TEST_END],
        "test": test_result,
        "test_brier_score": brier,
        "test_expected_calibration_error_10_bins": ece,
        "threshold": 0.5,
        "test_labels_used_for_fit_or_selection": False,
        "base_model": "temporal GRU selected on validation timesteps 31–34 and fitted only through timestep 30",
        "runtime_seconds": runtime,
        "device": str(device),
        "peak_gpu_memory_bytes": peak_memory,
        "limitations": [
            "Calibration is assessed only on labeled transactions; class-3 unknown transactions have no ground-truth risk label.",
            "The validation period is used for model selection and calibration, so a separate calibration split would provide a stronger calibration assessment.",
            "Same-timestep GRU summaries are transductive; aggregate-feature provenance remains external to this repository.",
        ],
    }
    report_path = ARTIFACT_DIR / "risk_predictor_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    artifact = {
        "artifact_version": 1,
        "risk_score_definition": report["risk_score_definition"],
        "calibrator": {"type": "platt_logistic", "slope": slope, "intercept": intercept},
        "gru_checkpoint": str(GRU_CHECKPOINT),
        "gru_checkpoint_sha256": hashlib.sha256(GRU_CHECKPOINT.read_bytes()).hexdigest(),
        "gru_config": config,
        "feature_names": checkpoint["feature_names"],
        "feature_transform": {
            "method": "training-period median imputation, median/IQR scaling, then signed log1p",
            "fit_through_timestep": 30,
            "fit_split": "validation-selected GRU fitting period",
        },
        "feature_medians": transform.medians,
        "feature_centers": transform.centers,
        "feature_scales": transform.scales,
        "calibration_fit_timesteps": [VALIDATION_START, 34],
        "threshold": 0.5,
    }
    torch.save(artifact, RISK_ARTIFACT)
    print(f"RUNTIME_SECONDS={runtime:.3f} PEAK_GPU_MEMORY_BYTES={peak_memory}")
    print(f"SAVED predictions={prediction_path}")
    print(f"SAVED metrics={report_path}")
    print(f"SAVED risk_artifact={RISK_ARTIFACT}")


if __name__ == "__main__":
    main()
