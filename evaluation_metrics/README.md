# Evaluation Metrics

This module independently evaluates actual held-out predictions from the five
existing classifiers. It does not train or modify any model. The only positive
class is illicit source label **1**; licit source label **2** maps to negative
class **0**. Source class **3** is unknown and receives predictions, but is
excluded from supervised metrics.

## Data and interpretation

Evaluation uses `data/processed/elliptic/transactions_test` (timesteps 35–49)
for ground truth and calls `models.inference.EllipticInference.score_batch`
for predictions. Labels are read only by this offline evaluation module, after
inference records are produced; inference itself continues to exclude labels.
Prediction IDs and timesteps must align exactly with the test Parquet rows.
The expected population is 67,504 rows: 1,083 illicit, 15,587 licit, and
50,834 unknown; supervised support is 16,670.

All eight regression-style metrics compare each model's continuous illicit
probability with the binary target. They are descriptive score-error measures;
the project is a classification task, not a separate regression task. MAPE is
explicitly `undefined` because licit targets are zero. NRMSE uses target range
`max(y_true)-min(y_true)`. Bias is `mean(prediction-target)`; positive means
overprediction and negative means underprediction. Bias is not ranked.

Classifier hard decisions use the prediction returned by the existing
inference implementation (fixed 0.5 threshold). The calibrated risk scorer
uses the threshold saved in its artifact. AUC-ROC and log loss use continuous
probabilities. The positive class is illicit. Unknown-class rows are never
treated as negative examples. Because the test set is imbalanced (6.4967%
illicit among labeled rows), read recall, specificity, balanced accuracy,
MCC, and probability metrics alongside accuracy. AUC-ROC here is distinct from
average precision / PR-AUC in the existing model comparison report; this module
does not relabel or replace the saved PR-AUC results.

## Metrics and formulas

### Regression-style metrics

| Name | Formula | Interpretation |
| --- | --- | --- |
| RMSE | `sqrt(mean((y_true-y_pred)^2))` | Lower; square-error magnitude, target units. |
| MSE | `mean((y_true-y_pred)^2)` | Lower; average squared error. |
| MAE | `mean(abs(y_true-y_pred))` | Lower; average absolute error. |
| MAPE | `mean(abs((y_true-y_pred)/y_true))*100` | Lower; undefined if any target is zero. |
| R² | `1 - SSE/SST` | Higher; may be negative; undefined for constant targets. |
| NRMSE | `RMSE/(max(y_true)-min(y_true))` | Lower; target-range normalization; undefined for zero range. |
| MedAE | `median(abs(y_true-y_pred))` | Lower; robust central absolute error. |
| Bias / ME | `mean(y_pred-y_true)` | Near zero; sign indicates over/underprediction. |

### Classification metrics

| Name | Formula | Interpretation |
| --- | --- | --- |
| Accuracy | `(TP+TN)/N` | Higher; can conceal minority errors. |
| Precision | `TP/(TP+FP)` | Higher; illicit-prediction reliability. |
| Recall | `TP/(TP+FN)` | Higher; illicit detection rate. |
| F1-score | `2TP/(2TP+FP+FN)` | Higher; positive-class precision/recall balance. |
| Specificity | `TN/(TN+FP)` | Higher; licit detection rate. |
| AUC-ROC | Positive-vs-negative probability ranking; ties receive half credit | Higher; threshold independent, undefined for one class. |
| Log Loss | `-mean(y log(p)+(1-y)log(1-p))` | Lower; probabilities clipped to `[1e-15,1-1e-15]`. |
| MCC | `(TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | Higher; balanced correlation summary. |
| Cohen's Kappa | `(observed agreement-expected agreement)/(1-expected agreement)` | Higher; chance-adjusted agreement. |
| Balanced Accuracy | `(sensitivity+specificity)/2` | Higher; equal weight to both true classes. |

Each metric function validates dimensions, matching nonempty lengths, finite
values, binary encodings, and probability ranges as applicable. Undefined
denominators raise `UndefinedMetricError`; the pipeline records `value: null`
and `status: undefined` with the reason. Input contract failures are errors;
model loading or scoring failures are recorded as unavailable for that model.

## Structure

```text
evaluation_metrics/
├── regression/                 # 8 independent formulas and functions
├── classification/             # 10 independent formulas and functions
├── definitions/metric_definitions.py
├── tests/                       # mathematical and alignment tests
├── results/                     # generated CSV and JSON exports
├── calculate_all.py             # actual model evaluation entry point
└── README.md
```

The structured definitions are available in
`evaluation_metrics.definitions.METRIC_DEFINITIONS` and can be serialized with
`definitions_as_json()` or written to a chosen path with `export_definitions()`.

## Run evaluation

Use the existing WSL environment. This computes predictions over the held-out
test IDs for GCN, GraphSAGE, GRU, Logistic Regression, and the saved calibrated
risk scorer. It does not fit models, transforms, calibration, or thresholds.

```bash
cd /mnt/d/BDE_Project/EllipticPlusPlus-Project
source ~/.venvs/elliptic-plus/bin/activate
TORCH_USE_RTLD_GLOBAL=1 python -m evaluation_metrics.calculate_all
```

The script evaluates models independently, writes per-metric statuses, and
compares calculated classification metrics and confusion matrices with each
model's saved metrics JSON. A mismatch is reported in
`saved_metrics_reference_check`; it does not replace the independently
calculated result. Comparison rankings include only metrics with status
`calculated`; undefined and unavailable values are excluded. Bias is listed
with its direction and magnitude, not ranked.

## Outputs

The command creates or refreshes these module-owned files under `results/`:

- `regression_results.csv` and `.json`
- `classification_results.csv` and `.json`
- `combined_results.csv`
- `model_comparison.json`

Every result row contains model, task, metric, value (or null), status,
explanation, sample count, and positive-class definition. The comparison JSON
includes all five model names, per-model results, metric values, valid rankings,
and models excluded from each ranking. No result is generated from synthetic
data; synthetic arrays are used only by unit tests.

## Tests

```bash
cd /mnt/d/BDE_Project/EllipticPlusPlus-Project
source ~/.venvs/elliptic-plus/bin/activate
TORCH_USE_RTLD_GLOBAL=1 python -m unittest \
  evaluation_metrics.tests.test_regression_metrics \
  evaluation_metrics.tests.test_classification_metrics \
  evaluation_metrics.tests.test_model_integration -v
```

## Fair-comparison limitations

All models are evaluated on the same labeled test transactions and the same
chronological split. Unknown labels are excluded consistently. Hard-label
metrics follow the saved inference decisions; probability metrics use
continuous scores. The calibrated risk scorer is the Platt-calibrated GRU
probability, not a distinct future-risk target. Its calibration was fit on the
GRU validation period, which was also used for model selection. Existing
aggregate-feature provenance and transductive graph-context limitations still
apply; this module does not establish causal or leakage-free performance.

## Visualization workflow

The companion `evaluation_metrics.graphs` package creates 300+ DPI metric
comparison PNGs only from the CSVs described above. It does not rerun model
inference. Matplotlib is required in the active environment. Generate all
figures with:

```bash
python -m evaluation_metrics.graphs.generate_all_graphs
```

Individual scripts are available as
`python -m evaluation_metrics.graphs.regression.plot_rmse` and
`python -m evaluation_metrics.graphs.classification.plot_auc_roc`; see
[`graphs/README.md`](graphs/README.md) for all 18 modules and output paths.
Undefined MAPE results are omitted, not drawn as zero. The runner writes
`graphs/generation_summary.json` with generated, skipped, and failed plots and
the exact values taken from each source CSV. Run
`python -m evaluation_metrics.calculate_all` first if the result CSVs are
missing or have been regenerated.
