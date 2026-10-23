# Model comparison

## Protocol

The labeled test period is timesteps 35–49: 16,670 transactions, including
15,587 licit and 1,083 illicit. The illicit prevalence is **6.4967%**. Class 3
is excluded from supervised metrics. Precision, recall, F1, and confusion
matrices use a fixed 0.5 threshold. PR-AUC is non-interpolated average
precision; its difference from prevalence is shown as a reference, not as a
separate metric guarantee.

GCN, GraphSAGE, and GRU use validation timesteps 31–34 to select epoch count
after fitting through timestep 30, then refit on timesteps 1–34. Test labels
are used only for final test metrics. The calibrated risk scorer uses a
validation-selected GRU trained through timestep 30 and fits Platt calibration
on labeled timesteps 31–34. This reuses the model-selection period for
calibration and may make calibration quality optimistic. Its test labels are
used only for final metrics; threshold remains fixed at 0.5.

The Logistic Regression reference is a single linear PyTorch classifier. It
uses the same 1–30 fitting / 31–34 epoch-selection / 1–34 final-fit / 35–49
test sequence, training-only median imputation and median/IQR scaling, signed
log1p compression, class-weighted cross-entropy, and fixed 0.5 threshold. No
scikit-learn or other package was installed.

Reproduce this baseline from the repository root in WSL with:

```bash
TORCH_USE_RTLD_GLOBAL=1 python -u -m models.logistic_baseline
```

## Held-out test comparison

| Model | Precision | Recall | F1 | Average precision | AP minus prevalence | Confusion matrix (licit, illicit rows) | Runtime | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| GCN | 0.3662 | 0.5420 | 0.4371 | 0.4461 | +37.11 pp | `[[14571, 1016], [496, 587]]` | 80.0 s | 1,143,282,688 B |
| GraphSAGE | 0.3166 | 0.5402 | 0.3992 | 0.5055 | +44.05 pp | `[[14324, 1263], [498, 585]]` | 90.3 s | 1,123,240,448 B |
| Temporal GRU | 0.3555 | 0.7655 | 0.4855 | 0.7266 | +66.16 pp | `[[14084, 1503], [254, 829]]` | 54.3 s | 778,695,680 B |
| GRU calibrated risk score | 0.4997 | 0.7276 | 0.5925 | 0.7022 | +63.72 pp | `[[14798, 789], [295, 788]]` | 43.0 s | 517,964,800 B |
| Logistic Regression | 0.2115 | 0.7664 | 0.3315 | 0.4739 | +40.89 pp | `[[12493, 3094], [253, 830]]` | 54.9 s | Not used (CPU) |

All listed PR-AUC values exceed test illicit prevalence. Metrics for the four
existing models were read from their saved reports and compared with the
previously reported values; each matched at four decimal places. Logistic
Regression is a new reference result, selected at epoch 3 (validation
macro-F1 0.6254; stopped after epoch 13) and refit for 3 epochs. It ran on CPU
for 54.946 seconds. Its saved logs and configuration are in
`models/logistic_baseline/artifacts/logistic_baseline_metrics.json`.

## Interpretation limits

The risk-score row reports a calibrated illicit-class probability from the
GRU, not an independently trained future-risk target. Calibration shares the
GRU selection period. Same-timestep graph and GRU summaries are transductive,
and the upstream provenance of aggregate features remains unresolved. These
results are chronological but should not be described as fully causal or
leakage-free.
