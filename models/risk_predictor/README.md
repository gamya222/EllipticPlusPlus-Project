# Calibrated transaction risk score

This component defines transaction risk as a **calibrated probability that a
labeled transaction is illicit (class 1)**. It is a score, not a new forward
looking target: the temporal GRU produces class logits, and one-dimensional
Platt scaling converts its illicit logit into a probability. Calibration fits
only on known labels from timesteps 31–34. The base GRU used for this component
was selected on that validation period and trained only through timestep 30;
the held-out test period 35–49 is used for final reporting only.

Unknown class 3 transactions receive scores but are excluded from supervised
calibration metrics because they have no ground truth. Report this as risk for
the labeled population, not as calibrated real-world probability for all
transactions. The validation period is used for both model selection and
calibration, so the calibration estimates may be optimistic; a separate
calibration window would strengthen that assessment.

Run in WSL from the repository root after training the GRU checkpoint:

```bash
TORCH_USE_RTLD_GLOBAL=1 python -m models.risk_predictor.train
```

The script saves calibrated test scores and metrics under `artifacts/`.

`artifacts/risk_scorer.pt` is the loadable scoring artifact. It contains the
existing fitted calibration coefficients, a checksum and relative reference
to the validation-selected GRU checkpoint, that checkpoint's model config,
feature order, and the training-only transform parameters used for calibration.
Create it from the already-saved files without fitting or scoring again with:

```bash
TORCH_USE_RTLD_GLOBAL=1 python -m models.risk_predictor.export_artifact
```

The export refuses to overwrite an existing risk artifact. The scoring command
uses this artifact when present, verifies its GRU checkpoint checksum and
calibration coefficients against the metrics JSON, and retains the legacy
metrics/GRU loading path when the artifact is absent.
