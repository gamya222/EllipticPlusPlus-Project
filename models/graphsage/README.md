# GraphSAGE mean-aggregator baseline

GraphSAGE uses the same 182 transaction features and labels as the GCN baseline.
Each layer concatenates a node's own representation with the mean of its
neighbors, then applies a learned linear transform. Edges are made undirected
and binary; node and edge prefixes stop at each inference timestep.

Training uses labeled timesteps 1–30, validation on 31–34, and final refitting
on 1–34 before test evaluation on 35–49. Class 3 remains context-only and is
excluded from loss and labeled-class metrics. Class-weighted loss handles the
licit/illicit imbalance. Feature preprocessing is fit only through the relevant
training cutoff: median imputation, median/IQR scaling, then signed log1p
compression. The input/output contract is an `N x 182` feature matrix plus a
sparse row-normalized neighborhood matrix, producing `N x 2` class logits.

Run from the repository root in WSL:

```bash
TORCH_USE_RTLD_GLOBAL=1 python -m models.graphsage.train
```

Checkpoint and essential metrics/logs are written under `artifacts/`.
