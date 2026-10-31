# Evaluation Metric Graphs

This module renders academic-style PNG comparisons from the calculated CSV
exports created by `evaluation_metrics.calculate_all`. It does not calculate
metrics, load models, infer predictions, or synthesize values. Only rows whose
status is `calculated` and whose values are finite are plotted. Undefined,
unavailable, inapplicable, erroneous, malformed, or missing model results are
excluded and reported; unavailable values are never converted to zero.

## Structure

```text
graphs/
├── _plotting.py
├── regression/                 # 8 standalone plot modules and their PNGs
├── classification/             # 10 standalone plot modules and their PNGs
├── generate_all_graphs.py
├── generation_summary.json     # written by the central runner
└── README.md
```

## Metrics and figures

Regression-style plots compare continuous classifier probabilities to the
binary illicit/licit target. They do not imply a separate regression task.

| Plot module | Output PNG | Display |
| --- | --- | --- |
| `regression.plot_rmse` | `regression/rmse_comparison.png` | Root mean square error |
| `regression.plot_mse` | `regression/mse_comparison.png` | Mean squared error |
| `regression.plot_mae` | `regression/mae_comparison.png` | Mean absolute error |
| `regression.plot_mape` | `regression/mape_comparison.png` | Percentage error; skipped when every valid result is undefined |
| `regression.plot_r2` | `regression/r2_comparison.png` | Coefficient of determination, retaining negative values |
| `regression.plot_nrmse` | `regression/nrmse_comparison.png` | Range-normalized RMSE |
| `regression.plot_medae` | `regression/medae_comparison.png` | Median absolute error |
| `regression.plot_bias` | `regression/bias_comparison.png` | Signed mean error with zero reference |
| `classification.plot_accuracy` | `classification/accuracy_comparison.png` | Classification accuracy |
| `classification.plot_precision` | `classification/precision_comparison.png` | Positive-class precision |
| `classification.plot_recall` | `classification/recall_comparison.png` | Positive-class recall |
| `classification.plot_f1_score` | `classification/f1_score_comparison.png` | Positive-class F1 |
| `classification.plot_specificity` | `classification/specificity_comparison.png` | True-negative rate |
| `classification.plot_auc_roc` | `classification/auc_roc_comparison.png` | ROC AUC (distinct from PR-AUC) |
| `classification.plot_log_loss` | `classification/log_loss_comparison.png` | Binary log loss |
| `classification.plot_mcc` | `classification/mcc_comparison.png` | Matthews correlation coefficient |
| `classification.plot_cohens_kappa` | `classification/cohens_kappa_comparison.png` | Cohen's kappa |
| `classification.plot_balanced_accuracy` | `classification/balanced_accuracy_comparison.png` | Average class recall |

The five-model axis order follows the evaluation protocol: GCN, GraphSAGE, GRU,
Logistic Regression, Calibrated Risk Scorer. A model is shown only if that
metric has a calculated finite value in the source CSV. If fewer models have
valid values, the figure notes the exclusions. If none are valid, the script
prints a skip reason and writes no empty PNG.

## Styling and units

Matplotlib is used with a white background, restrained model palette, readable
labels, subtle horizontal gridlines, and 320 DPI output. Signed metrics (R²,
MCC, Cohen's kappa, and bias) include a zero reference and retain negative
values. Bounded classification rates and AUC-ROC use their original 0–1 scale;
log loss and regression error values are not rescaled. MAPE uses percent units
if calculated, but the current evaluation's zero-valued labels make it
undefined for every model, so its graph is correctly skipped.

Each PNG includes a title, axis labels, sample support, and the source results
CSV in its provenance note. The underlying `plotted_values` in
`generation_summary.json` are parsed directly from that CSV.

## Run an individual plot

From the repository root in the existing WSL venv:

```bash
cd /mnt/d/BDE_Project/EllipticPlusPlus-Project
source ~/.venvs/elliptic-plus/bin/activate
python -m evaluation_metrics.graphs.regression.plot_rmse
python -m evaluation_metrics.graphs.classification.plot_auc_roc
```

The other scripts follow the module/output table above. The required source
CSV must exist under `evaluation_metrics/results/`; no model inference is
triggered by plotting.

## Generate all graphs

```bash
cd /mnt/d/BDE_Project/EllipticPlusPlus-Project
source ~/.venvs/elliptic-plus/bin/activate
python -m evaluation_metrics.graphs.generate_all_graphs
```

If the result CSVs are missing, run `python -m evaluation_metrics.calculate_all`
first. The runner invokes all 18 scripts independently, continues after
individual failures, and writes `generation_summary.json` with generated,
skipped, and failed counts, reasons, plotted model values, source CSVs, and PNG
paths. It does not run evaluation or replace evaluation results.

To regenerate after updating metrics, first run the evaluation command, then
run the graph-generation command. The PNGs and summary are refreshed from the
new exported CSV values.
