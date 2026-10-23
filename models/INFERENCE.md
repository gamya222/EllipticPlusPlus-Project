# Batch inference

The reusable Python interface is `models.inference.EllipticInference`. The CLI
writes a versioned JSON document suitable for a future visualization layer.
Each record combines the requested model's illicit-class score and fixed
threshold prediction, separate calibrated GRU risk, graph statistics,
community fields, suspicious-connection ranking where available, and an
explicit status. It never replaces an existing output file.

From the repository root in the existing WSL environment, score a small sample
and write a new output file outside the repository:

```bash
cd /mnt/d/BDE_Project/EllipticPlusPlus-Project
source ~/.venvs/elliptic-plus/bin/activate
TORCH_USE_RTLD_GLOBAL=1 python -m models.inference --model gcn --tx-ids 13385567 -1 --output "$HOME/elliptic-inference/sample.json"
```

Supported model names are `gcn`, `graphsage`, `gru`, and `logistic`. Every
result also includes the separate risk-scorer output. Example Python use:

```python
from models.inference import EllipticInference

scorer = EllipticInference()
results = scorer.score_batch([13385567, 13385568], model="gcn")
```

Only known transactions in the existing held-out test period (timesteps 35–49)
are scoreable. Unknown IDs return `unknown_transaction_id`; known IDs outside
that range return `unsupported_timestep`. Their scores and graph fields are
JSON `null`, not zero. This interface does not score unseen transactions:
the supplied model and graph inputs are transductive over the fixed existing
dataset, with no incremental graph-ingestion path.

Feature transforms are loaded from the selected model checkpoint. Calibrated
risk uses the saved risk artifact, its checksum-verified validation-selected
GRU state, saved transform, and fixed Platt coefficients. No transform,
calibrator, or threshold is fitted at inference. No class/label Parquet column
is read. Prediction uses the saved fixed 0.5 cutoff.

The output fields are `txId`, `Time step`, `status`, `requested_model`,
`model_output` (illicit probability, fixed-threshold prediction), `risk`
(calibrated GRU probability), `graph_statistics`, `community`, and
`suspicious_connection`. Full-graph degree/component/community fields are
descriptive and can include later-timestep graph structure; their scope is
explicitly marked. Suspicious rank/heuristic and neighbor fields come from the
existing as-of-timestep ranking artifact. The heuristic is a ranking measure,
not the calibrated probability. Missing analytics rows produce `null` fields.

For graph neural models, prediction-time adjacency is built only from existing
edges whose endpoints are at or before the requested timestep. Same-timestep
transductive context is used, matching the documented model evaluation setup.
Aggregate-feature provenance is upstream and cannot be independently
established here. Do not interpret these outputs as wallet identities, causal
explanations, or validated risk for unknown-class transactions.

Focused tests can be run with:

```bash
TORCH_USE_RTLD_GLOBAL=1 python -m unittest tests.test_inference -v
```

## Full supported test-period batch

To score all 67,504 existing test-period transactions for one selected model,
run (for example, for GCN):

```bash
TORCH_USE_RTLD_GLOBAL=1 python -m models.inference --all-test --model gcn --output "$HOME/elliptic-inference/gcn-test.json"
```

Use `--model graphsage`, `--model gru`, or `--model logistic` for another model.
Each run also includes the existing calibrated GRU risk and graph-analysis
records. `--all-test` derives IDs from transaction IDs and timesteps only and
does not read labels. The output document has schema version
`elliptic-inference-v1`, a count/status summary, scope metadata, and a `records`
array using the same fields and types as `score_batch`. Scores and structural
values are numbers or `null`; IDs, timesteps, counts, and ranks are integers or
`null`; status and scope values are strings. JSON does not contain NaN or
infinity values.

The full batch is not run by the focused tests. Based on the measured small GCN
CLI run (~21 seconds including environment, model, and data loading), a full
GCN batch is estimated at roughly 2–5 minutes, about 1–2 GB host memory, and
hundreds of MB of GPU memory when CUDA is active. The indented JSON output is
estimated at 70–100 MB. These are planning estimates, not a benchmark; actual
cost varies with model, device, and disk. Each run scores one requested
classifier; a full batch is optional.
