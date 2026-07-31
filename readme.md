# Fraud detection with chronological validation

A portfolio/research-style fraud-detection pipeline that tests whether a model trained on earlier card activity remains useful on later transactions.

## Result at a glance

The selected XGBoost pipeline was trained through August 2020, tuned and thresholded on September–October, then evaluated once on November–December.

| Metric at the validation-locked threshold | Validation | Latest-period test |
|---|---:|---:|
| PR-AUC | 0.979889 | 0.954273 |
| Precision | 52.44% | 18.19% |
| Recall | 98.93% | 99.18% |
| Review rate | 1.1666% | 1.7720% |
| Estimated scenario cost | $18,375 | $38,700 |

On the 451,137-transaction test period, the model detected 1,454 of 1,466 fraud cases and flagged 6,540 legitimate transactions. The test review rate remained below the configured 5% capacity limit, but rose from 1.17% on validation to 1.77% on test; precision also declined materially. That is a reason to monitor drift and validate prospectively, not a production-readiness claim.

Full generated metrics: [evaluation report](reports/evaluation.md) and [machine-readable metrics](reports/metrics.json).

## The core problem

Fraud models must make decisions using only information available when a transaction occurs. An earlier notebook analysis used a random split and full-dataset transformations, allowing later behavior to influence earlier examples and making its results unsuitable as an estimate of future performance.

This rebuild invalidates those legacy results and uses a chronological experiment instead. The [original implementation audit](docs/audit.md) documents the findings and the [legacy-claim reconciliation](docs/legacy_claims.md) records which old claims must not be reused.

The data is also highly imbalanced: fraud represents 0.60% of training transactions and 0.62% of validation transactions. Accuracy would therefore be misleading. The pipeline uses training-derived class weighting, selects models by PR-AUC, and selects a cost- and capacity-constrained review threshold on validation data rather than relying on a default 0.5 cutoff.

## What changed

- **Strict-past features:** per-card history and 1-hour, 24-hour, and 7-day velocity features use earlier timestamps only. Transactions with the same timestamp cannot observe one another.
- **Chronological splits:** train, validation, and test cover consecutive time periods. Test data is not used for feature selection, preprocessing, model selection, or threshold selection.
- **Train-only preprocessing:** clipping bounds, imputers, scaling, and category vocabularies are fitted within the training pipeline.
- **Imbalance-aware modeling:** class weights are calculated from the training split; no synthetic over- or under-sampling is applied. PR-AUC, fraud-class precision/recall, and review volume—not accuracy—drive evaluation.
- **An operating point chosen on validation:** the threshold minimizes a stated false-negative/false-positive cost scenario while limiting validation review volume to 5%.
- **Operational evaluation:** final reporting includes fraud-class precision and recall, confusion counts, review workload, uncertainty intervals, monthly slices, and drift diagnostics.

The corrected fraud-label history features are intentionally excluded: the source has no label-availability timestamp, so using past labels at scoring time cannot be justified. See the [methodology notes](docs/methodology.md) for feature and implementation details.

## Operational interpretation

The configured scenario assigns $500 to a missed fraud and $5 to a legitimate transaction sent for review. These are decision assumptions, not observed losses or savings. The validation-locked threshold is `0.6991600990` and is not retuned on test data.

The 5% cap is a **validation threshold-selection constraint**, not a separate test-period capacity definition. The evaluation code applies it to validation; the same locked threshold produces a 1.7720% test review rate, which is below 5%. Workload nevertheless increased by about 52% relative to validation. Detailed monthly, drift, and limitation analysis is in the [operational evaluation notes](docs/operational_evaluation.md).

## Repository structure

```text
config/                  Fixed experiment and cost assumptions
src/fraud_detection/     Data, features, split, preprocessing, training, inference
tests/                   Leakage, boundary, inference, and smoke tests
artifacts/               Fitted pipeline and metadata
reports/                 Generated metrics and diagnostics
docs/                    Audit, methodology, operations, and legacy reconciliation
*.ipynb                  Legacy/explanatory notebooks
```

## Reproduce

Requirements: Python 3.11–3.13 and Poetry 2.x.

```bash
poetry install --with dev
poetry run pytest -q
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run python -m fraud_detection.training --config config/training.toml
```

The configured input is `prepped_data.pkl`, intentionally excluded from Git because it is large and contains PII-shaped identifiers. To start from the public source, download it from the [Kaggle dataset page](https://www.kaggle.com/datasets/karthikgangula/credit-card-fraud-mega-dataset) or run:

```bash
kaggle datasets download karthikgangula/credit-card-fraud-mega-dataset --unzip
poetry run python -m fraud_detection.training \
  --config config/training.toml \
  --input /path/to/credit_card_fraud.csv
```

CSV ingestion applies the declared California/2020 scope. For package usage and artifact contents, see [methodology notes](docs/methodology.md).

## Limitations

- This is a California, 2020, known-card temporal evaluation—not a cold-start, geographic, or multi-year benchmark.
- The Kaggle listing does not document data collection or simulation methods; this is not verified production banking data.
- Category and history-feature distributions drifted, and test precision fell despite high recall.
- Fraud-label delay is unknown, probabilities are not calibrated to production outcomes, and the cost inputs are hypothetical.

## Technical documentation

- [Leakage investigation and audit history](docs/audit.md)
- [Methodology and implementation notes](docs/methodology.md)
- [Operational evaluation, drift, and extended limitations](docs/operational_evaluation.md)
- [Legacy notebook and result reconciliation](docs/legacy_claims.md)
- [Generated evaluation report](reports/evaluation.md)
