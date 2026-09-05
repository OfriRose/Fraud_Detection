# Fraud detection with chronological validation

Detect fraudulent card transactions without using future information. The
reproducible source pipeline replaces a legacy notebook experiment affected by
preprocessing and historical-feature leakage.

## Approach and result

The fixed experiment covers California transactions from 2020. Strict-past
card history and velocity features feed train-fitted preprocessing, logistic
regression, and two XGBoost candidates. Validation PR-AUC selects the model;
validation also locks a cost-based threshold under a 5% review-rate limit.

The saved **version 1.1.0** XGBoost model achieved **test PR-AUC 0.9543**,
**18.19% precision**, and **99.18% recall**, reviewing **1.77%** of transactions.
This illustrates the trade-off between detecting fraud and reviewing legitimate
transactions; it does not establish production readiness. Costs of $500 per
missed fraud and $5 per false alert are scenario assumptions, not realized losses.

The generated [evaluation report](reports/evaluation.md) is the authoritative
source for detailed results; [metrics.json](reports/metrics.json) contains the
machine-readable values. See [operational evaluation](docs/operational_evaluation.md)
for interpretation and limitations.

## Reproduce

Requirements: Python 3.11–3.13 and Poetry 2.x.

```bash
poetry install --with dev
poetry run pytest -q
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run python -m fraud_detection.training --config config/training.toml
```

The configured `prepped_data.pkl` is excluded from Git because it is large and
contains identifiers. For a clean clone, supply an authorized copy at that path
or pass a CSV/pickle override:

```bash
poetry run python -m fraud_detection.training \
  --config config/training.toml \
  --input /authorized/path/to/transactions.csv
```

Training writes the fitted pipeline and metadata to `artifacts/` and generated
evaluation outputs to `reports/`. Configuration lives in
[training.toml](config/training.toml).

## Inference and methodology

Use `load_artifact` and `score_transactions` from `fraud_detection.inference`.
The [methodology](docs/methodology.md) includes a scoring example, input-history
requirements, leakage controls, and model-selection rationale.

Source code lives in `src/fraud_detection/`; regression and smoke tests live in
`tests/`. Notebooks include historical experiments and explanatory views; the
source pipeline is authoritative. The [original audit](docs/audit.md) explains
why legacy random-split results cannot be used as evidence of performance.

## Limits

Evaluation covers one state, one year, and recurring cards. Dataset collection
or simulation methods are undocumented, category drift is substantial, and
scores are uncalibrated. Fraud-label availability is unknown, so target-derived
history is excluded. Legacy full-year consolidation also requires excluding
`city`, `job`, and `merchant`. Prospective evaluation is needed before deployment.
