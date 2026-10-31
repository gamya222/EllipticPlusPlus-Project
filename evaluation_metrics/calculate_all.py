"""Evaluate saved Elliptic++ model predictions on the existing test split.

Run from the repository root in WSL:
    TORCH_USE_RTLD_GLOBAL=1 python -m evaluation_metrics.calculate_all

No model is fitted here. Ground-truth labels are read only by this offline
evaluation module; the inference implementation itself remains label-blind.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.dataset as ds

from evaluation_metrics._validation import MetricInputError, UndefinedMetricError
from evaluation_metrics._validation import confusion_counts
from evaluation_metrics.classification import (
    calculate_accuracy, calculate_auc_roc, calculate_balanced_accuracy,
    calculate_cohens_kappa, calculate_f1_score, calculate_log_loss,
    calculate_mcc, calculate_precision, calculate_recall, calculate_specificity,
)
from evaluation_metrics.definitions import METRIC_DEFINITIONS
from evaluation_metrics.regression import (
    calculate_bias, calculate_mae, calculate_mape, calculate_medae,
    calculate_mse, calculate_nrmse, calculate_r2, calculate_rmse,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_PARQUET = ROOT / "data/processed/elliptic/transactions_test"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS = ("gcn", "graphsage", "gru", "logistic", "risk_scorer")
RISK_ARTIFACT_MODEL_KEY = "risk_predictor"
METRIC_PATHS = {
    "gcn": ROOT / "models/gcn/artifacts/gcn_baseline_metrics.json",
    "graphsage": ROOT / "models/graphsage/artifacts/graphsage_baseline_metrics.json",
    "gru": ROOT / "models/gru/artifacts/temporal_gru_baseline_metrics.json",
    "logistic": ROOT / "models/logistic_baseline/artifacts/logistic_baseline_metrics.json",
    "risk_scorer": ROOT / "models/risk_predictor/artifacts/risk_predictor_metrics.json",
}
POSITIVE_CLASS = "illicit (source class 1; encoded 1)"
DECISION_THRESHOLD = 0.5

REGRESSION_METRICS: dict[str, Callable] = {
    "rmse": calculate_rmse,
    "mse": calculate_mse,
    "mae": calculate_mae,
    "mape": calculate_mape,
    "r2": calculate_r2,
    "nrmse": calculate_nrmse,
    "medae": calculate_medae,
    "bias": calculate_bias,
}
CLASSIFICATION_CLASS_METRICS: dict[str, Callable] = {
    "accuracy": calculate_accuracy,
    "precision": calculate_precision,
    "recall": calculate_recall,
    "f1_score": calculate_f1_score,
    "specificity": calculate_specificity,
    "mcc": calculate_mcc,
    "cohens_kappa": calculate_cohens_kappa,
    "balanced_accuracy": calculate_balanced_accuracy,
}
CLASSIFICATION_PROBABILITY_METRICS: dict[str, Callable] = {
    "auc_roc": calculate_auc_roc,
    "log_loss": calculate_log_loss,
}


def load_test_truth(parquet_path: Path = TEST_PARQUET) -> list[dict[str, int]]:
    """Read test IDs, timesteps, and original classes for offline evaluation."""
    if not parquet_path.is_dir():
        raise FileNotFoundError(f"Test transaction Parquet directory is missing: {parquet_path}")
    table = ds.dataset(str(parquet_path), format="parquet").to_table(
        columns=["txId", "Time step", "class"]
    )
    if table.num_rows == 0:
        raise ValueError("Test transaction Parquet is empty")
    truth = []
    seen: set[int] = set()
    for tx_id, timestep, original_class in zip(
        table["txId"].to_pylist(),
        table["Time step"].to_pylist(),
        table["class"].to_pylist(),
        strict=True,
    ):
        if tx_id is None or timestep is None or original_class is None:
            raise ValueError("Test IDs, timesteps, and labels must not be NULL")
        tx_id, timestep, original_class = int(tx_id), int(timestep), int(original_class)
        if tx_id in seen:
            raise ValueError(f"Duplicate test transaction ID {tx_id}")
        if original_class not in (1, 2, 3):
            raise ValueError(f"Unexpected source class {original_class} for transaction {tx_id}")
        if not 35 <= timestep <= 49:
            raise ValueError(f"Unexpected test timestep {timestep} for transaction {tx_id}")
        seen.add(tx_id)
        truth.append({"txId": tx_id, "Time step": timestep, "class": original_class})
    return truth


def align_prediction_records(
    predictions: list[dict[str, Any]], truth_rows: list[dict[str, int]]
) -> list[tuple[dict[str, Any], int | None]]:
    """Align predictions by ID and timestep; map class 1/2 to 1/0, class 3 to None."""
    prediction_by_id: dict[int, dict[str, Any]] = {}
    for record in predictions:
        tx_id = int(record["txId"])
        if tx_id in prediction_by_id:
            raise ValueError(f"Duplicate prediction transaction ID {tx_id}")
        prediction_by_id[tx_id] = record
    truth_by_id: dict[int, dict[str, int]] = {}
    for row in truth_rows:
        tx_id = int(row["txId"])
        if tx_id in truth_by_id:
            raise ValueError(f"Duplicate truth transaction ID {tx_id}")
        truth_by_id[tx_id] = row
    if set(prediction_by_id) != set(truth_by_id):
        missing = sorted(set(truth_by_id) - set(prediction_by_id))[:5]
        extra = sorted(set(prediction_by_id) - set(truth_by_id))[:5]
        raise ValueError(f"Prediction/truth IDs differ; missing={missing}, extra={extra}")

    aligned = []
    for truth in truth_rows:
        tx_id = int(truth["txId"])
        prediction = prediction_by_id[tx_id]
        if prediction.get("Time step") is None or int(prediction["Time step"]) != int(truth["Time step"]):
            raise ValueError(f"Timestep mismatch for transaction {tx_id}")
        if prediction.get("status") != "scored":
            raise ValueError(f"Transaction {tx_id} was not scored: {prediction.get('status')}")
        label = {1: 1, 2: 0, 3: None}[int(truth["class"])]
        aligned.append((prediction, label))
    return aligned


def _metric_entry(function: Callable, *args) -> dict[str, Any]:
    try:
        value = float(function(*args))
        if not np.isfinite(value):
            return {"value": None, "status": "error", "explanation": "Metric returned a non-finite value"}
        return {"value": value, "status": "calculated", "explanation": None}
    except UndefinedMetricError as exc:
        return {"value": None, "status": "undefined", "explanation": str(exc)}
    except MetricInputError as exc:
        return {"value": None, "status": "error", "explanation": str(exc)}


def evaluate_arrays(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    y_predicted: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Calculate every requested regression-style and classification metric."""
    regression = {name: _metric_entry(fn, y_true, y_probability) for name, fn in REGRESSION_METRICS.items()}
    classification = {
        name: _metric_entry(fn, y_true, y_predicted)
        for name, fn in CLASSIFICATION_CLASS_METRICS.items()
    }
    classification.update({
        name: _metric_entry(fn, y_true, y_probability)
        for name, fn in CLASSIFICATION_PROBABILITY_METRICS.items()
    })
    return regression, classification


def _unavailable_result(reason: str, population: dict[str, Any]) -> dict[str, Any]:
    def entries(metric_names: list[str]) -> dict[str, dict[str, Any]]:
        return {name: {"value": None, "status": "unavailable", "explanation": reason} for name in metric_names}
    return {
        "population": population,
        "regression_metrics": entries(list(REGRESSION_METRICS)),
        "classification_metrics": entries(list(CLASSIFICATION_CLASS_METRICS) + list(CLASSIFICATION_PROBABILITY_METRICS)),
        "error": reason,
    }


def _collect_scores(model: str, aligned: list[tuple[dict[str, Any], int | None]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    labeled = [(record, label) for record, label in aligned if label is not None]
    if not labeled:
        raise ValueError("No labeled test transactions are available")
    y_true = np.asarray([label for _, label in labeled], dtype=np.int8)
    if model == "risk_scorer":
        y_probability = np.asarray([record["risk"]["score"] for record, _ in labeled], dtype=np.float64)
        thresholds = {float(record["risk"]["threshold"]) for record, _ in labeled}
        if len(thresholds) != 1:
            raise ValueError("Risk artifact returned inconsistent classification thresholds")
        threshold = thresholds.pop()
        y_predicted = (y_probability >= threshold).astype(np.int8)
    else:
        y_probability = np.asarray([record["model_output"]["score"] for record, _ in labeled], dtype=np.float64)
        predictions = []
        for record, _ in labeled:
            predicted = record["model_output"].get("prediction")
            if predicted not in ("illicit", "licit"):
                raise ValueError(f"Invalid predicted class {predicted!r} for transaction {record['txId']}")
            predictions.append(1 if predicted == "illicit" else 0)
        y_predicted = np.asarray(predictions, dtype=np.int8)
        threshold = DECISION_THRESHOLD
    support = {
        "labeled": int(y_true.size),
        "illicit": int(np.sum(y_true == 1)),
        "licit": int(np.sum(y_true == 0)),
        "unknown_excluded": int(sum(label is None for _, label in aligned)),
    }
    if not np.isfinite(y_probability).all():
        raise ValueError("Model probability output contains NaN or infinity")
    return y_true, y_probability, y_predicted, {**support, "decision_threshold": threshold}


def _read_saved_metrics(model: str) -> dict[str, Any] | None:
    path = METRIC_PATHS[model]
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_check(model: str, classification: dict, y_true, y_predicted) -> dict[str, Any]:
    reference = _read_saved_metrics(model)
    if reference is None or not isinstance(reference.get("test"), dict):
        return {"status": "unavailable", "explanation": "Saved reference metrics JSON is missing"}
    test = reference["test"]
    checks = {
        "accuracy": (classification.get("accuracy", {}).get("value"), test.get("accuracy")),
        "precision": (classification.get("precision", {}).get("value"), test.get("illicit_precision")),
        "recall": (classification.get("recall", {}).get("value"), test.get("illicit_recall")),
        "f1_score": (classification.get("f1_score", {}).get("value"), test.get("illicit_f1")),
    }
    mismatches = {
        key: {"calculated": actual, "saved": expected}
        for key, (actual, expected) in checks.items()
        if actual is None or expected is None or not np.isclose(actual, expected, atol=5e-5, rtol=0.0)
    }
    saved_cm = test.get("confusion_matrix_licit_illicit")
    calculated_cm = None
    if classification.get("accuracy", {}).get("status") == "calculated":
        tn, fp, fn, tp = confusion_counts(y_true, y_predicted)
        calculated_cm = [[tn, fp], [fn, tp]]
        if saved_cm != calculated_cm:
            mismatches["confusion_matrix_licit_illicit"] = {
                "calculated": calculated_cm,
                "saved": saved_cm,
            }
    return {
        "status": "match" if not mismatches else "mismatch",
        "compared_metrics": list(checks),
        "saved_confusion_matrix_licit_illicit": saved_cm,
        "calculated_confusion_matrix_licit_illicit": calculated_cm,
        "mismatches": mismatches,
        "source": str(METRIC_PATHS[model].relative_to(ROOT)).replace("\\", "/"),
    }


def _result_rows(records: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    rows = []
    task_key = "regression_metrics" if task == "regression" else "classification_metrics"
    for model_record in records:
        sample_count = model_record.get("population", {}).get("labeled", 0)
        for name, result in model_record.get(task_key, {}).items():
            definition = METRIC_DEFINITIONS[name]
            rows.append({
                "model": model_record["model"],
                "task": task,
                "metric": name,
                "abbreviation": definition["abbreviation"],
                "value": result["value"],
                "status": result["status"],
                "explanation": result["explanation"] or "",
                "sample_count": sample_count,
                "positive_class": POSITIVE_CLASS,
            })
    return rows


def _comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    direction = {name: definition["desirable"] for name, definition in METRIC_DEFINITIONS.items()}
    comparisons = {}
    for task in ("regression_metrics", "classification_metrics"):
        metric_names = list(REGRESSION_METRICS) if task == "regression_metrics" else list(CLASSIFICATION_CLASS_METRICS) + list(CLASSIFICATION_PROBABILITY_METRICS)
        for name in metric_names:
            values = {}
            excluded = {}
            for model_record in records:
                result = model_record.get(task, {}).get(name)
                model = model_record["model"]
                if result is not None and result["status"] == "calculated":
                    values[model] = result["value"]
                else:
                    excluded[model] = result["status"] if result else "unavailable"
            desirable = direction[name]
            ranking = []
            if desirable in ("higher", "lower"):
                reverse = desirable == "higher"
                ranking = sorted(values, key=values.get, reverse=reverse)
            comparisons[name] = {
                "desirable": desirable,
                "values": values,
                "ranking": ranking,
                "excluded_models": excluded,
            }
    return {
        "models": [record["model"] for record in records],
        "positive_class": POSITIVE_CLASS,
        "threshold_protocol": "Existing inference class decisions; 0.5 for classifier models, saved risk-artifact threshold for calibrated risk.",
        "regression_interpretation": "Continuous class probabilities are compared with binary labels for descriptive score-error metrics; the project is a classifier, not a separate regression task.",
        "bias_interpretation": "Read signed bias values and their magnitudes; bias is not ranked as simply higher or lower.",
        "comparisons": comparisons,
        "model_results": records,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["model", "task", "metric", "abbreviation", "value", "status", "explanation", "sample_count", "positive_class"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation() -> dict[str, Any]:
    """Run actual inference for each supported model and export metric results."""
    truth_rows = load_test_truth()
    truth_by_id = {row["txId"]: row for row in truth_rows}
    class_counts = {str(label): sum(row["class"] == label for row in truth_rows) for label in (1, 2, 3)}
    population = {
        "test_rows": len(truth_rows),
        "class_1_illicit": class_counts["1"],
        "class_2_licit": class_counts["2"],
        "class_3_unknown_excluded": class_counts["3"],
        "labeled_rows": class_counts["1"] + class_counts["2"],
        "test_timesteps": [35, 49],
        "positive_class": POSITIVE_CLASS,
    }
    records: list[dict[str, Any]] = []
    try:
        from models.inference import EllipticInference

        scorer = EllipticInference()
        mask = (scorer.data.times >= 35) & (scorer.data.times <= 49)
        tx_ids = [int(tx_id) for tx_id in scorer.data.ids[mask]]
        if set(tx_ids) != set(truth_by_id):
            raise ValueError("Inference transaction IDs do not match held-out test Parquet IDs")
    except Exception as exc:
        reason = f"Inference engine unavailable: {type(exc).__name__}: {exc}"
        for model in MODELS:
            records.append({"model": model, "population": population, **_unavailable_result(reason, population)})
        return export_results(records, population)

    for model in MODELS:
        try:
            requested_model = "logistic" if model == "risk_scorer" else model
            predicted_records = scorer.score_batch(tx_ids, model=requested_model)
            aligned = align_prediction_records(predicted_records, truth_rows)
            y_true, y_probability, y_predicted, support = _collect_scores(model, aligned)
            regression, classification = evaluate_arrays(y_true, y_probability, y_predicted)
            selected_population = {
                **population,
                "labeled": support["labeled"],
                "illicit": support["illicit"],
                "licit": support["licit"],
                "unknown_excluded": support["unknown_excluded"],
                "decision_threshold": support["decision_threshold"],
            }
            records.append({
                "model": model,
                "task": "binary transaction classification",
                "prediction_source": "models.inference.EllipticInference.score_batch; risk_scorer uses its saved calibrated GRU score.",
                "population": selected_population,
                "regression_metrics": regression,
                "classification_metrics": classification,
                "saved_metrics_reference_check": _reference_check(model, classification, y_true, y_predicted),
                "confusion_matrix_licit_illicit": confusion_counts(y_true, y_predicted),
            })
            print(f"EVALUATED {model}: labeled={support['labeled']} excluded_unknown={support['unknown_excluded']} auc_roc={classification['auc_roc']['value']} status={classification['auc_roc']['status']}")
        except Exception as exc:
            reason = f"Model evaluation failed: {type(exc).__name__}: {exc}"
            records.append({"model": model, "population": population, **_unavailable_result(reason, population)})
            print(f"UNAVAILABLE {model}: {reason}")
        finally:
            # Each checkpoint has its own full-graph feature tensor on CUDA.
            # Release it between models to avoid retaining five such tensors.
            scorer.feature_cache.clear()
            if scorer.device.type == "cuda":
                import torch

                torch.cuda.empty_cache()
    return export_results(records, population)


def export_results(records: list[dict[str, Any]], population: dict[str, Any]) -> dict[str, Any]:
    """Write required CSV and JSON result files under evaluation_metrics/results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    regression_rows = _result_rows(records, "regression")
    classification_rows = _result_rows(records, "classification")
    _write_csv(RESULTS_DIR / "regression_results.csv", regression_rows)
    _write_csv(RESULTS_DIR / "classification_results.csv", classification_rows)
    _write_csv(RESULTS_DIR / "combined_results.csv", regression_rows + classification_rows)
    common = {
        "protocol": "held-out timesteps 35-49; class 1=illicit positive, class 2=licit negative, class 3 excluded from supervised metrics",
        "population": population,
        "model_results": records,
    }
    (RESULTS_DIR / "regression_results.json").write_text(
        json.dumps({**common, "metrics": "continuous model probability vs binary target"}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "classification_results.json").write_text(
        json.dumps(common, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    comparison = _comparison(records)
    (RESULTS_DIR / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"EXPORTED {RESULTS_DIR}")
    print(f"CSV rows: regression={len(regression_rows)}, classification={len(classification_rows)}, combined={len(regression_rows) + len(classification_rows)}")
    return comparison


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
