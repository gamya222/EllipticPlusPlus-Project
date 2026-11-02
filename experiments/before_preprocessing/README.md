# Before-preprocessing experiment

This isolated experiment compares the five existing classifiers on the original
182-feature CSV matrix, with **training-only median imputation and no scaling or
signed-log transform**. It reads `data/raw/elliptic/` directly and validates
the ID-based feature/class join, graph endpoints, expected counts, and temporal
population. It does not edit source CSVs, processed Parquet, existing model
artifacts, or saved After results.

## Environment

The verified Windows execution environment is Python 3.11 with a separate
`.venv_before_preprocessing`. The existing project `.venv` is not modified.
Verified packages: PyTorch 2.11.0+cu128, NumPy 2.4.6, pandas 3.1.0,
scikit-learn 1.9.1, and NetworkX 3.6.1. CUDA 12.8 is available on the RTX 3050
Laptop GPU. This runner uses the repository's custom PyTorch sparse GCN and
GraphSAGE implementations; PyTorch Geometric is not needed. It loads raw CSVs
with pandas, so Spark, PyArrow, and winutils are not involved in this run.

To recreate the isolated environment on a compatible Windows/Python 3.11 host:

```powershell
python -m venv .venv_before_preprocessing
.\.venv_before_preprocessing\Scripts\python.exe -m pip install numpy==2.4.6 pandas==3.1.0 scikit-learn==1.9.1
.\.venv_before_preprocessing\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128'
```

## Run

Run from the repository root. First run the small raw-data training smoke test;
it performs two optimizer steps for each architecture on a deterministic,
class-stratified sample of training-period transactions and writes no artifacts.

```powershell
.\.venv_before_preprocessing\Scripts\python.exe experiments\before_preprocessing\run_experiment.py --smoke-only
```

Then run all models (default seed 2026, maximum 60 epochs, patience 10,
hidden dimension 64, dropout 0.5, Adam learning rate 0.01, weight decay
0.0005, and class-weighted cross-entropy):

```powershell
.\.venv_before_preprocessing\Scripts\python.exe experiments\before_preprocessing\run_experiment.py
```

Use `--cpu` to force CPU. The runner refuses a nonempty output directory rather
than overwriting previous experimental files. After interruption, `--resume`
validates completed per-model JSON/CSV/checkpoint artifacts and skips them; it
does not replace them. Use `--output-dir PATH` to write a later experiment to a
new empty location.

## Evaluation protocol

- Epoch selection trains with labeled timesteps 1–30 and selects by validation
  macro-F1 at threshold 0.5 on labeled timesteps 31–34.
- Final classifiers are freshly initialized and fit on labeled timesteps 1–34
  for the selected epoch count. Their imputation medians are refit on feature
  rows from timesteps 1–34 only.
- Test evaluation is labeled timesteps 35–49 (16,670 transactions); class 3
  remains unlabeled and is excluded from supervised metrics. Unknown test rows
  are counted in metadata, not used for optimization, calibration, or selection.
- GCN/GraphSAGE graph inputs contain only nodes and observed edges whose two
  endpoints are at or before each prediction timestep. The graph neural models
  remain transductive over each available temporal prefix.
- The risk scorer uses the validation-selected raw-scale GRU fit on timesteps
  1–30. Platt coefficients are fitted on labeled timesteps 31–34 and then
  applied to test logits. The calibration period is also the GRU selection
  period; this caveat is retained in result metadata.
- Threshold metrics use 0.5. PR-AUC is average precision and ROC-AUC is computed
  from actual probabilities. No threshold is tuned on test labels.

The 10 reported metrics are accuracy, precision, recall/sensitivity,
specificity, F1, PR-AUC, ROC-AUC, FPR, FNR, and error rate. All metrics and
prediction exports use actual held-out labeled predictions.

## Outputs

The five JSON reports are written directly under
`evaluation_metrics/before_preprocessing/`. Checkpoints, the validation GRU
used for calibration, and labeled test probability CSVs are written under its
ignored `artifacts/` subdirectory. Each JSON includes the configuration,
selected epoch, per-epoch logs, imputation medians, split/population metadata,
confusion matrix, metrics, device, runtime, and artifact paths.

The dashboard reads these JSON files dynamically for the Before column. The
existing After values continue to come from the current Model Lab/evaluation
data. Change is `After - Before` for higher-is-better metrics and `Before -
After` for lower-is-better metrics. Results should be compared with care:
preprocessing is one experimental difference, but aggregate feature provenance
and the transductive graph context remain limitations. This is a research
benchmark, not causal or production validation.
