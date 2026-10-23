# GCN baseline

This package reads the existing files under `data/processed/elliptic` and does
not rewrite them. It excludes `txId`, `Time step`, and `class` from the feature
matrix; the remaining 182 numeric columns are node features. Class 1 is the
positive illicit label, class 2 maps to licit, and class 3 is unknown. Unknown
labels remain in the graph as context but never enter supervised loss or scored
metrics.

For model selection, supervised fitting uses timesteps 1–30 and evaluates
chronologically on 31–34. The final fit uses labeled timesteps 1–34. Evaluation
at each timestep uses only graph nodes with `Time step <= current timestep`.
The GCN uses binary, undirected edges with self-loops and symmetric degree
normalization. Feature medians and median/IQR scaling are fitted on training
timesteps only (including unlabeled training-period nodes); validation and test
rows do not affect those statistics. Scaled values then receive signed
`log1p(abs(x))` compression to control the very heavy feature tails (the
inspected scaled maximum was about 9.43e9). This transform has no fitted
parameters and uses no validation or test statistics. A feature with no finite
training value causes an explicit error instead of receiving an arbitrary
replacement.

`Aggregate_feature_*` columns are kept because they are part of the supplied
feature schema. Their upstream provenance is not documented in this repository,
so the baseline cannot independently establish whether their construction used
only information available at the corresponding timestep. Treat this as a data
provenance limitation when interpreting results.

Run from the repository root in the configured WSL environment:

```bash
TORCH_USE_RTLD_GLOBAL=1 python -m models.gcn.train
```

The process-level `TORCH_USE_RTLD_GLOBAL=1` setting is required in the current
WSL venv: the default PyTorch CUDA dependency preloader selected leftover CUDA
12 libraries from shared `nvidia/*/lib` paths even though the installed wheel
uses CUDA 13.2. With the setting, CPU and RTX 3050 tensor checks passed. It is
scoped to the training process and does not change the global environment.

The default seed is 2026; selection is capped at 60 epochs with patience 10.
Validation reports illicit precision, recall, F1, and non-interpolated average
precision (step-wise PR area) each epoch. The best validation macro-F1 selects
the epoch count; the model is then refit on timesteps 1–34 before test scoring
on 35–49. The final model, configuration, validation logs, and metrics are
written under `models/gcn/artifacts/`.

Graph inference uses nodes and edges only through the timestep being scored.
The inspected Parquet edge set has no cross-timestep edges. The supplied
aggregate features are one-hop neighborhood summaries, but their original
construction and prediction-time availability cannot be independently
verified from this repository. Therefore, describe this baseline as a
chronological, same-timestep transductive evaluation and retain that feature
provenance limitation; do not claim a fully causal or leakage-free result.
