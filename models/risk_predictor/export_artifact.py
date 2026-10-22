"""Persist the already-fitted risk calibrator without rerunning scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
GRU_CHECKPOINT = ROOT / "models/gru/artifacts/temporal_gru_baseline.pt"
METRICS_PATH = ROOT / "models/risk_predictor/artifacts/risk_predictor_metrics.json"
OUTPUT_PATH = ROOT / "models/risk_predictor/artifacts/risk_scorer.pt"


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing risk artifact: {OUTPUT_PATH}")
    if not GRU_CHECKPOINT.is_file() or not METRICS_PATH.is_file():
        raise FileNotFoundError("Existing GRU checkpoint and risk metrics are required")

    checkpoint = torch.load(GRU_CHECKPOINT, map_location="cpu", weights_only=False)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if "validation_state_dict" not in checkpoint:
        raise ValueError("GRU checkpoint has no validation-selected model state")
    calibrator = metrics["calibrator"]
    if calibrator.get("type") != "one-dimensional logistic/Platt scaling":
        raise ValueError("Unexpected calibrator type in existing metrics")

    artifact = {
        "artifact_version": 1,
        "risk_score_definition": metrics["risk_score_definition"],
        "calibrator": {
            "type": "platt_logistic",
            "slope": float(calibrator["slope"]),
            "intercept": float(calibrator["intercept"]),
        },
        "gru_checkpoint": "models/gru/artifacts/temporal_gru_baseline.pt",
        "gru_checkpoint_sha256": hashlib.sha256(GRU_CHECKPOINT.read_bytes()).hexdigest(),
        "gru_config": checkpoint["config"],
        "feature_names": checkpoint["feature_names"],
        "feature_transform": {
            "method": "training-period median imputation, median/IQR scaling, then signed log1p",
            "fit_through_timestep": 30,
            "fit_split": "validation-selected GRU fitting period",
        },
        "feature_medians": checkpoint["validation_feature_medians"],
        "feature_centers": checkpoint["validation_feature_centers"],
        "feature_scales": checkpoint["validation_feature_scales"],
        "calibration_fit_timesteps": metrics["calibration_fit_timesteps"],
        "threshold": metrics["threshold"],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, OUTPUT_PATH)
    print(f"SAVED {OUTPUT_PATH}")
    print(f"BYTES {OUTPUT_PATH.stat().st_size}")
    print(f"SLOPE {artifact['calibrator']['slope']:.16g}")
    print(f"INTERCEPT {artifact['calibrator']['intercept']:.16g}")


if __name__ == "__main__":
    main()
