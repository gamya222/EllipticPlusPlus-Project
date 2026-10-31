"""Shared, validated plotting functions for independent metric scripts."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from evaluation_metrics.definitions import METRIC_DEFINITIONS

ROOT = Path(__file__).resolve().parents[2]
GRAPH_ROOT = Path(__file__).resolve().parent
MODEL_ORDER = ("gcn", "graphsage", "gru", "logistic", "risk_scorer")
MODEL_LABELS = {
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gru": "GRU",
    "logistic": "Logistic Regression",
    "risk_scorer": "Calibrated Risk Scorer",
}
MODEL_COLORS = {
    "gcn": "#2F5D8A",
    "graphsage": "#4F8A8B",
    "gru": "#6F7FA8",
    "logistic": "#9A7653",
    "risk_scorer": "#A64B4B",
}
RATE_METRICS = {
    "accuracy", "precision", "recall", "f1_score", "specificity",
    "auc_roc", "balanced_accuracy",
}
SIGNED_METRICS = {"r2", "bias", "mcc", "cohens_kappa"}
Y_LABELS = {
    "rmse": "RMSE (probability vs. binary-label scale)",
    "mse": "MSE (squared binary-label scale)",
    "mae": "MAE (probability vs. binary-label scale)",
    "mape": "MAPE (%)",
    "r2": "R²",
    "nrmse": "Range-normalized RMSE",
    "medae": "Median absolute error (binary-label scale)",
    "bias": "Mean error (probability − binary label)",
    "accuracy": "Accuracy (0–1)",
    "precision": "Precision (0–1)",
    "recall": "Recall (0–1)",
    "f1_score": "F1-score (0–1)",
    "specificity": "Specificity (0–1)",
    "auc_roc": "AUC-ROC (0–1)",
    "log_loss": "Log loss",
    "mcc": "Matthews correlation coefficient",
    "cohens_kappa": "Cohen's kappa",
    "balanced_accuracy": "Balanced accuracy (0–1)",
}


def results_csv(task: str) -> Path:
    """Return the source CSV for a regression or classification metric."""
    if task not in {"regression", "classification"}:
        raise ValueError(f"Unsupported metric task: {task}")
    return ROOT / "evaluation_metrics" / "results" / f"{task}_results.csv"


def output_png(task: str, metric: str) -> Path:
    """Return the generated PNG path for the selected metric."""
    return GRAPH_ROOT / task / f"{metric}_comparison.png"


def collect_metric_data(metric: str, task: str) -> dict[str, Any]:
    """Load only calculated finite values, retaining exclusions for reporting."""
    if metric not in METRIC_DEFINITIONS:
        raise ValueError(f"Unknown metric: {metric}")
    source = results_csv(task)
    if not source.is_file():
        raise FileNotFoundError(
            f"Evaluation results are missing: {source}. Run "
            "`python -m evaluation_metrics.calculate_all` first."
        )

    valid: dict[str, dict[str, Any]] = {}
    excluded: dict[str, str] = {}
    matched = 0
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"model", "metric", "value", "status", "sample_count"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{source.name} is missing required columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            if row["metric"] != metric:
                continue
            matched += 1
            model = row["model"]
            label = MODEL_LABELS.get(model, model)
            if model not in MODEL_LABELS:
                excluded[label] = "model is not in the supported evaluation set"
                continue
            if row["status"] != "calculated":
                excluded[label] = row["explanation"] or f"status is {row['status']}"
                continue
            try:
                value = float(row["value"])
                sample_count = int(row["sample_count"])
            except (TypeError, ValueError):
                excluded[label] = "calculated row has a missing or invalid numeric value"
                continue
            if not math.isfinite(value):
                excluded[label] = "calculated row contains a non-finite numeric value"
                continue
            if sample_count <= 0:
                excluded[label] = "calculated row has an invalid sample count"
                continue
            if model in valid:
                raise ValueError(f"Duplicate calculated result for metric={metric}, model={model}")
            valid[model] = {"label": label, "value": value, "sample_count": sample_count}

    if matched == 0:
        excluded["results"] = f"metric {metric!r} has no rows in {source.name}"
    for model in MODEL_ORDER:
        label = MODEL_LABELS[model]
        if model not in valid and label not in excluded:
            excluded[label] = "no evaluation result row exists for this model and metric"
    ordered = [model for model in MODEL_ORDER if model in valid]
    return {
        "metric": metric,
        "task": task,
        "source": source,
        "values": valid,
        "ordered_models": ordered,
        "excluded": excluded,
    }


def _style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8792A2")
    ax.spines["bottom"].set_color("#8792A2")
    ax.tick_params(axis="both", colors="#253247", labelsize=9)


def _axis_limits(ax, metric: str, values: list[float]) -> None:
    if metric in RATE_METRICS:
        ax.set_ylim(0.0, 1.05)
        return
    if metric in SIGNED_METRICS:
        low = min(0.0, min(values))
        high = max(0.0, max(values))
        span = high - low
        padding = max(span * 0.14, 0.04)
        ax.set_ylim(low - padding, high + padding)
        ax.axhline(0.0, color="#374151", linewidth=1.0, zorder=2)
        return
    high = max(values)
    ax.set_ylim(0.0, max(high * 1.18, 0.05))


def generate_metric_plot(metric: str, task: str) -> dict[str, Any]:
    """Generate one 300-DPI bar chart using values from the evaluation CSV."""
    data = collect_metric_data(metric, task)
    if not data["ordered_models"]:
        reasons = "; ".join(f"{model}: {reason}" for model, reason in data["excluded"].items())
        reason = reasons or "no calculated values are available"
        print(f"SKIPPED {metric}: {reason}")
        return {"metric": metric, "task": task, "status": "skipped", "reason": reason, "source": str(data["source"])}

    models = data["ordered_models"]
    entries = data["values"]
    values = [entries[model]["value"] for model in models]
    labels = [entries[model]["label"] for model in models]
    counts = {entries[model]["sample_count"] for model in models}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    positions = np.arange(len(models))
    colors = [MODEL_COLORS[model] for model in models]
    bars = ax.bar(positions, values, width=0.68, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(positions, labels, rotation=16 if len(labels) > 3 else 0, ha="right" if len(labels) > 3 else "center")
    ax.set_xlabel("Models", labelpad=8)
    ax.set_ylabel(Y_LABELS[metric], labelpad=8)
    definition = METRIC_DEFINITIONS[metric]
    abbreviation = definition["abbreviation"]
    ax.set_title(f"Model Comparison — {definition['full_form']} ({abbreviation})", loc="left", pad=12, color="#17243A", weight="semibold")
    _style_axis(ax)
    _axis_limits(ax, metric, values)

    span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-9)
    offset = span * 0.018
    for bar, value in zip(bars, values, strict=True):
        if metric in SIGNED_METRICS and value < 0:
            y = value - offset
            va = "top"
        else:
            y = value + offset
            va = "bottom"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.4g}", ha="center", va=va, fontsize=8.5, color="#17243A", clip_on=False)

    if len(counts) == 1:
        provenance = f"Evaluation results: {data['source'].relative_to(ROOT)}  •  labeled n = {next(iter(counts)):,}"
    else:
        supports = ", ".join(f"{entries[model]['label']} n={entries[model]['sample_count']:,}" for model in models)
        provenance = f"Evaluation results: {data['source'].relative_to(ROOT)}  •  {supports}"
    if data["excluded"]:
        absent = ", ".join(f"{model} ({reason})" for model, reason in data["excluded"].items())
        provenance += f"  •  Excluded: {absent}"
    fig.text(0.01, -0.015, provenance, ha="left", va="top", fontsize=7.2, color="#5D6878", wrap=True)

    target = output_png(task, metric)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=320, bbox_inches="tight", metadata={"Title": f"{definition['full_form']} model comparison", "Source": str(data["source"])})
    plt.close(fig)
    print(f"GENERATED {target} models={len(models)} values={dict(zip(models, values, strict=True))}")
    return {
        "metric": metric,
        "task": task,
        "status": "generated",
        "path": str(target),
        "source": str(data["source"]),
        "dpi": 320,
        "plotted_models": [entries[model]["label"] for model in models],
        "plotted_values": {entries[model]["label"]: entries[model]["value"] for model in models},
        "sample_counts": {entries[model]["label"]: entries[model]["sample_count"] for model in models},
        "excluded": data["excluded"],
        "y_axis": Y_LABELS[metric],
        "title": f"Model Comparison — {definition['full_form']} ({abbreviation})",
    }
