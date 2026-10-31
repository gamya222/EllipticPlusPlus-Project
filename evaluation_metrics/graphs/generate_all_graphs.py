"""Execute all standalone metric plotters and summarize their outcomes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluation_metrics.graphs._plotting import GRAPH_ROOT, ROOT, collect_metric_data, output_png, results_csv

PLOTTERS = (
    ("regression", "rmse", "evaluation_metrics.graphs.regression.plot_rmse"),
    ("regression", "mse", "evaluation_metrics.graphs.regression.plot_mse"),
    ("regression", "mae", "evaluation_metrics.graphs.regression.plot_mae"),
    ("regression", "mape", "evaluation_metrics.graphs.regression.plot_mape"),
    ("regression", "r2", "evaluation_metrics.graphs.regression.plot_r2"),
    ("regression", "nrmse", "evaluation_metrics.graphs.regression.plot_nrmse"),
    ("regression", "medae", "evaluation_metrics.graphs.regression.plot_medae"),
    ("regression", "bias", "evaluation_metrics.graphs.regression.plot_bias"),
    ("classification", "accuracy", "evaluation_metrics.graphs.classification.plot_accuracy"),
    ("classification", "precision", "evaluation_metrics.graphs.classification.plot_precision"),
    ("classification", "recall", "evaluation_metrics.graphs.classification.plot_recall"),
    ("classification", "f1_score", "evaluation_metrics.graphs.classification.plot_f1_score"),
    ("classification", "specificity", "evaluation_metrics.graphs.classification.plot_specificity"),
    ("classification", "auc_roc", "evaluation_metrics.graphs.classification.plot_auc_roc"),
    ("classification", "log_loss", "evaluation_metrics.graphs.classification.plot_log_loss"),
    ("classification", "mcc", "evaluation_metrics.graphs.classification.plot_mcc"),
    ("classification", "cohens_kappa", "evaluation_metrics.graphs.classification.plot_cohens_kappa"),
    ("classification", "balanced_accuracy", "evaluation_metrics.graphs.classification.plot_balanced_accuracy"),
)


def _write_summary(summary: dict[str, Any]) -> Path:
    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    path = GRAPH_ROOT / "generation_summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def generate_all_graphs() -> dict[str, Any]:
    """Run each independent plot module, continuing after individual failures."""
    missing = [str(results_csv(task).relative_to(ROOT)) for task in ("regression", "classification") if not results_csv(task).is_file()]
    records = []
    if missing:
        reason = (
            f"Evaluation result file(s) missing: {', '.join(missing)}. "
            "Run `python -m evaluation_metrics.calculate_all` before generating graphs."
        )
        records = [
            {"metric": metric, "task": task, "status": "skipped", "reason": reason}
            for task, metric, _ in PLOTTERS
        ]
        print(f"SKIPPED ALL GRAPHS: {reason}")
    else:
        for task, metric, module in PLOTTERS:
            target = output_png(task, metric)
            try:
                data = collect_metric_data(metric, task)
                command = [sys.executable, "-m", module]
                env = os.environ.copy()
                env["MPLBACKEND"] = "Agg"
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = (result.stdout or "").strip()
                if result.returncode != 0:
                    reason = (result.stderr or output or f"plotter exited with code {result.returncode}").strip()
                    record = {
                        "metric": metric,
                        "task": task,
                        "status": "failed",
                        "reason": reason[-2000:],
                        "source": str(data["source"]),
                    }
                elif not data["ordered_models"] or output.startswith("SKIPPED"):
                    record = {
                        "metric": metric,
                        "task": task,
                        "status": "skipped",
                        "reason": output or "No rows with status=calculated and finite values",
                        "source": str(data["source"]),
                        "existing_png_retained": target.is_file(),
                    }
                elif target.is_file() and output.startswith("GENERATED"):
                    record = {
                        "metric": metric,
                        "task": task,
                        "status": "generated",
                        "path": str(target),
                        "source": str(data["source"]),
                        "dpi": 320,
                        "plotted_models": [data["values"][name]["label"] for name in data["ordered_models"]],
                        "plotted_values": {data["values"][name]["label"]: data["values"][name]["value"] for name in data["ordered_models"]},
                        "sample_counts": {data["values"][name]["label"]: data["values"][name]["sample_count"] for name in data["ordered_models"]},
                        "excluded": data["excluded"],
                    }
                else:
                    record = {
                        "metric": metric,
                        "task": task,
                        "status": "failed",
                        "reason": f"Plotter exited {result.returncode} without producing the expected PNG",
                        "source": str(data["source"]),
                    }
                if output:
                    print(output)
                if result.stderr and result.returncode == 0:
                    record["warnings"] = result.stderr.strip()[-2000:]
                records.append(record)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                records.append({"metric": metric, "task": task, "status": "failed", "reason": reason})
                print(f"FAILED {metric}: {reason}")

    summary = {
        "schema_version": "elliptic-metric-graphs-v1",
        "provenance": "Only calculated finite values from evaluation_metrics/results CSV exports are plotted.",
        "graph_directory": str(GRAPH_ROOT),
        "requested_graph_count": len(PLOTTERS),
        "generated_count": sum(row["status"] == "generated" for row in records),
        "skipped_count": sum(row["status"] == "skipped" for row in records),
        "failed_count": sum(row["status"] == "failed" for row in records),
        "graphs": records,
    }
    path = _write_summary(summary)
    print(
        f"GRAPH SUMMARY generated={summary['generated_count']} "
        f"skipped={summary['skipped_count']} failed={summary['failed_count']} report={path}"
    )
    return summary


def main() -> None:
    generate_all_graphs()


if __name__ == "__main__":
    main()
