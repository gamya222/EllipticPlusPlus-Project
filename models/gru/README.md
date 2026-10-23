# Temporal GRU baseline

Each timestep is represented by the mean of its 182 transformed transaction
feature vectors. The unidirectional GRU processes these 49 snapshot vectors in
time order. For each transaction, a classifier combines its own transformed
features with the GRU state at that transaction's timestep. The input is thus a
causal history of feature-distribution summaries plus the current transaction;
it is not a sequence of repeated node IDs (transactions are unique nodes).
Snapshot summaries use features only, never labels. A node at timestep `t` can
only receive the recurrent state through `t`.

The model uses training-only median imputation, median/IQR scaling, and signed
log1p compression. Training/validation/test periods are 1–30 / 31–34 / 35–49
for selection and 1–34 / 35–49 for the final fit and test. Class 3 is excluded
from supervised loss and labeled-class metrics. Since each timestep summary
uses all feature rows in that timestep, this is transductive batch context, not
an online within-timestep forecast.

Run in WSL from the repository root:

```bash
TORCH_USE_RTLD_GLOBAL=1 python -m models.gru.train
```

The input/output contract is `N x 182` node features plus 49 `x 182` snapshot
vectors, producing `N x 2` class logits. Checkpoint, configuration, logs, and
metrics are written under `artifacts/`.
