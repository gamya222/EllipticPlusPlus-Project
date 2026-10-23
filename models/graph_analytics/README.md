# Graph analytics

Run from the repository root in the existing WSL virtual environment:

```bash
python -m models.graph_analytics.analyze_graph
```

The command reads only the existing Parquet edge and graph-node directories,
plus the existing risk-score CSV. It does not read class labels or modify the
preprocessing/model artifacts. Generated files are written to
`models/graph_analytics/artifacts/`, which is covered by the existing
`models/*/artifacts/` ignore rule.

## Algorithms and outputs

- `node_graph_statistics.csv` reports each transaction ID and timestep,
  directed in/out degree, directed total degree, undirected degree, mean
  undirected-neighbor degree, weak-component ID/size, and community ID/size.
  Component and community IDs are the smallest transaction ID in their group.
- `component_summaries.csv` reports node and directed-edge counts for weakly
  connected components of the undirected projection.
- `community_summaries.csv` reports each community's node count, mean
  undirected degree, and internal undirected-edge count.
- `suspicious_transactions.csv` ranks all test-period nodes with available
  calibrated GRU risk scores. It includes the model score, as-of-timestep
  degrees, scored-neighbor count/mean/maximum, and a ranking heuristic.
- `graph_statistics.json` contains graph totals, component/community
  algorithm metadata, modularity, validation status fields, and runtime.

Degree statistics preserve the input edge direction (`txId1` to `txId2`):
`out_degree` counts source appearances and `in_degree` counts destination
appearances. The simple undirected projection collapses reciprocal rows if
they exist. In this dataset, all 234,355 input edges are same-timestep and
there are no duplicate undirected pairs.

Connected components use a deterministic union-find implementation on that
undirected projection. IDs use the minimum transaction ID, so they remain
stable across runs and do not depend on internal union-find roots.

Communities use NetworkX 3.7's unweighted Louvain modularity optimization on
the undirected simple graph (`resolution=1.0`, seed 2026). The run is about
linear in nodes plus edges per optimization pass and has been measured on this
graph: the complete analytics run took 44.13 seconds in the current WSL
environment (about 33.43 seconds through graph summaries and Louvain; the
remainder includes the temporal neighbor ranking and CSV validation). No extra
package was installed. The summary includes modularity so the
partition can be interpreted as a structural grouping, not as a semantic
label. Different resolution/seed choices can produce different communities.

Suspicious ranking uses the existing calibrated illicit-class score and
same/earlier-timestep scored neighbors only. Its heuristic is
`0.5 * own calibrated score + 0.5 * mean scored-neighbor risk`; if there is no
scored neighbor by that timestep, it uses the transaction's own score. Ties
break by own score, as-of undirected degree, then transaction ID. This is a
ranking score, not a calibrated probability or a causal explanation. Neighbor
means/maxima are blank when no scored neighbor exists; the explicit neighbor
count is zero in that case. These summaries only cover neighbors that have an
existing risk score (test-period nodes), not unscored training nodes. The
existing CSV also scores class-3 unknown transactions, but calibration was
assessed only on labeled transactions; treat those unknown-class values as
model scores rather than validated calibrated probabilities.

## Interpretation and limits

Full-graph degrees, components, and communities are descriptive summaries of
the available whole graph. They are not used as time-specific model features.
For suspicious rankings, edge structure and scored neighbors are recomputed
as of each transaction's timestep to avoid using later graph edges. The risk
score CSV contains all test transactions, including unknown class 3; graph
analytics never loads ground-truth labels. The fixed 50/50 score combination
is an explicit, untuned ranking heuristic, not a validated decision rule.

The transaction graph does not identify wallets or provide mixer proximity.
Feature provenance remains an upstream limitation and is not inferred by
these graph summaries. Louvain communities describe connectivity structure;
they do not establish illicit behavior.
