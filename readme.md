# Fraud detection with chronological validation

This project detects fraudulent card transactions while treating temporal
validity, leakage prevention, reproducibility, and operational reporting as
first-class requirements.

## Portfolio overview

This project began as a notebook-based course assignment and was later rebuilt
as a portfolio project. During that review, I found that the original random
split and full-dataset transformations made its results look much stronger than
they would be on later transactions. I chose to invalidate those results,
redesign the experiment, and report the weaker—but more credible—outcome.

The project now demonstrates:

- the ability to audit and challenge an earlier analysis rather than defend an
  attractive but unreliable result;
- practical understanding of how time affects fraud detection and model
  evaluation;
- conversion of exploratory notebook work into tested, reusable Python code;
- translation of model scores into review workload and explicit cost
  assumptions; and
- clear communication of deployment risks, data limitations, and unsuccessful
  findings.

The main lesson is that high offline metrics are not enough. Corrected
strict-past velocity features improved both validation and latest-period test
performance, but precision still fell materially between those periods as
fraud prevalence and category mix shifted. That result remains a monitoring
and external-validation warning rather than a production-readiness claim.

The original notebook experiment used a random row split after full-data
preprocessing and future-inclusive feature engineering. Its metrics are not
comparable to production performance. The corrected source pipeline has now
been rerun successfully; the results below come from its fixed chronological
experiment.

## Business objective

Fraud review involves two competing errors:

- A false negative misses a fraudulent transaction.
- A false positive sends a legitimate transaction into review or blocks it.

The operating scenario assigns a fixed cost of `$500` to each false negative
and `$5` to each false positive. Validation threshold selection also limits the
intended review queue to at most 5% of validation transactions. These are
configurable scenario assumptions, not measured losses or realized savings.

Accuracy is not used as the primary objective. Candidate models are selected by
validation PR-AUC, and their locked operating points are reported with
fraud-class precision, recall, absolute confusion counts, review load, and
estimated scenario cost.

## Data and temporal design

The raw source is Karthik Gangula's
[Credit Card Fraud Mega Dataset on Kaggle](https://www.kaggle.com/datasets/karthikgangula/credit-card-fraud-mega-dataset).
The Kaggle listing identifies it as an 11.19 GB public dataset under the MIT
license. The listing does not document how the transactions were collected or
generated, so this project does **not** describe them as verified real bank
transactions. The fields include names, SSNs, card/account numbers, and
locations that look like personal data; they are treated as PII-shaped
identifiers even though the dataset is publicly downloadable.

The declared experiment scope is California transactions from 2020. The
corrected run used the surviving `prepped_data.pkl` artifact (2,013,945 rows),
derived from the Kaggle CSV by the legacy preparation notebook. That notebook
had already consolidated `city`, `job`, and `merchant` using the complete
year. To prevent that legacy operation from entering the model, all three
columns are explicitly excluded.

The split boundaries are fixed in `config/training.toml`:

| Split | Date range | Transactions | Fraud cases | Fraud rate |
|---|---|---:|---:|---:|
| Train | 2020-01-01 00:01:13–2020-08-31 23:59:59 | 1,260,902 | 7,561 | 0.5997% |
| Validation | 2020-09-01 00:58:20–2020-10-31 23:59:56 | 301,906 | 1,867 | 0.6184% |
| Test | 2020-11-01 00:23:09–2020-12-31 23:59:48 | 451,137 | 1,466 | 0.3250% |

Train, validation, and test indices are asserted to be pairwise disjoint.
Validation is used for candidate selection and threshold locking. Test is not
used for feature choice, preprocessing, parameter selection, model selection,
or threshold selection.

Random splitting is inappropriate here because it mixes early and late
transactions, conceals time drift, and allows future behavior from the same
card to influence earlier training examples.

## Leakage controls

The authoritative implementation is under `src/fraud_detection/`.

- Transactions are ordered by card and event time before history is built.
- Equal-timestamp transactions are processed as one bucket and cannot observe
  one another.
- Previous transaction count, cumulative amount, mean, sample standard
  deviation, fraud count/rate, and time since prior transaction use strictly
  earlier timestamps.
- Card transaction counts, maximum amounts, and mean amounts over the previous
  1 hour, 24 hours, and 7 days use `[timestamp - window, timestamp)` boundaries.
  A validation-only ablation accepted this corrected replacement for the
  legacy rolling features before version 1.1.0 evaluated the test period.
- First-card-event historical values are explicitly zero and accompanied by
  `IS_FIRST_CARD_TX`.
- `CC_BIN` extracts the first six digits per card; invalid, short, and missing
  values remain missing. Raw card numbers and `CC_BIN` are identifier fields and
  are excluded from modeling.
- The corrected `CC_PREV_FRAUD` and historical fraud rate are generated and
  regression-tested, but disabled in the model because the data has no
  `label_available_at` timestamp.
- IQR bounds, imputers, scaler statistics, and one-hot vocabularies are fit on
  training only inside one scikit-learn pipeline.
- Categorical encoding uses `handle_unknown="ignore"`.
- The target is asserted absent from every feature matrix.
- No resampling is performed.

The old global `qcut` features were removed. The corrected model does not need
quantile bins, so there are no bin edges to learn or apply.

## Model selection and validation result

Every candidate includes the complete train-fitted preprocessing pipeline.
Models are compared on the same temporal validation period.

| Candidate | Validation PR-AUC | Precision | Recall | Review rate | Estimated cost |
|---|---:|---:|---:|---:|---:|
| Prevalence baseline | 0.006184 | 0.0000 | 0.0000 | 0.0000% | $933,500 |
| Logistic regression | 0.634782 | 0.2242 | 0.9373 | 2.5849% | $88,770 |
| Conservative XGBoost | 0.967554 | 0.4237 | 0.9882 | 1.4422% | $23,545 |
| Configured XGBoost | **0.979889** | **0.5244** | **0.9893** | **1.1666%** | **$18,375** |

The configured XGBoost candidate was selected by validation PR-AUC. Validation
locked the operating threshold at `0.6991600990` by minimizing the stated cost
under the 5% review constraint.

## Final test result

The selected pipeline and locked threshold were applied to the latest period:

| Metric | Test result |
|---|---:|
| PR-AUC | 0.954273 |
| ROC-AUC (secondary) | 0.999266 |
| Precision | 0.181886 |
| Recall | 0.991814 |
| Fraud detected (TP) | 1,454 |
| Fraud missed (FN) | 12 |
| Legitimate transactions flagged (FP) | 6,540 |
| Legitimate transactions cleared (TN) | 443,131 |
| Review rate | 1.7720% |
| Estimated scenario cost | $38,700 |

Precision 95% Wilson interval: 17.36%–19.05%. Recall 95% Wilson interval:
98.57%–99.53%.

Precision is materially weaker than validation, but recall remains high and
the locked threshold stays within the intended 5% review capacity on test. The
threshold was not retuned using test data. November review load is 2.05%;
December falls to 1.63%.

The main observed drift signals include:

- Fraud prevalence drops from 0.6184% on validation to 0.3250% on test.
- Category distribution total-variation distance between train and test is
  0.9590.
- Transaction month and accumulated card-history features shift substantially
  by construction and should be monitored carefully.
- The 24-hour and 7-day transaction counts also rise over time, so their
  distributions require monitoring even though the short-window family
  generalized well in this test.

See `reports/evaluation.md`, `reports/test_monthly_metrics.csv`, and
`reports/train_test_drift.csv` for the generated details.

## Reproduce

Requirements: Python 3.11–3.13 and Poetry 2.x.

```bash
poetry install --with dev
poetry run pytest -q
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run python -m fraud_detection.training --config config/training.toml
```

The module form of the training command is intentional: generated console
script shebangs are not portable when this repository’s directory name contains
spaces.

The configured input is `prepped_data.pkl`, which exists in this workspace but
is intentionally ignored because it is large and contains PII-shaped
identifiers. A clean clone can download the public source through the Kaggle
CLI:

```bash
kaggle datasets download \
  karthikgangula/credit-card-fraud-mega-dataset \
  --unzip
```

The Kaggle download is the raw input, not the prepared pickle. Run the data
preparation notebook to recreate `prepped_data.pkl`, or pass the downloaded CSV
directly:

```bash
poetry run python -m fraud_detection.training \
  --config config/training.toml \
  --input /path/to/credit_card_fraud.csv
```

CSV ingestion applies the declared CA/2020 scope directly; it never chooses a
state from target outcomes. Keep both raw and prepared data out of Git: the
dataset can be reproduced from its public source, while committing multi-GB
data would make cloning and GitHub maintenance impractical.

## Inference

The versioned artifact bundles the fitted IQR transform, imputation, scaling,
one-hot vocabulary, estimator, feature schema, threshold, and metadata.

```python
import pandas as pd

from fraud_detection.inference import load_artifact, score_transactions

artifact = load_artifact("artifacts/fraud_pipeline_v1.1.0.joblib")
transaction = pd.DataFrame(
    [
        {
            "cc_num": "4111111111111111",
            "trans_timestamp": "2020-12-31 12:00:00",
            "amt": 125.00,
            "category": "shopping_net",
            "profile": "adults_2550_female_urban.json",
            "city_pop": 100_000,
            "is_male": 0,
            "dob": "1990-06-15",
            "lat": 34.05,
            "long": -118.24,
            "merch_lat": 34.06,
            "merch_long": -118.25,
        }
    ]
)

result = score_transactions(transaction, artifact)
print(result[["fraud_probability", "fraud_decision", "model_version"]])
```

For an existing card, pass only history known before every current row via the
`history=` argument. The inference function rejects equal-time or future
history. Model outputs are class-weighted risk probabilities and have not been
calibrated against observed production outcomes.

## Repository layout

```text
config/                  Fixed experiment and cost assumptions
src/fraud_detection/     Data, features, split, preprocessing, models,
                         evaluation, training, and inference
tests/                   Leakage, boundary, inference, and smoke tests
artifacts/               Versioned fitted pipeline and metadata
reports/                 Generated metrics, diagnostics, and figures
docs/                    Original audit and legacy-claim reconciliation
*.ipynb                  Lightweight explanation/EDA notebooks
```

Generated, present artifacts:

- `artifacts/fraud_pipeline_v1.1.0.joblib`
- `artifacts/model_metadata.json`
- `reports/metrics.json`
- `reports/model_comparison.csv`
- `reports/split_summary.csv`
- `reports/validation_thresholds.csv`
- `reports/test_monthly_metrics.csv`
- `reports/test_category_metrics.csv`
- `reports/train_test_drift.csv`
- `reports/feature_importance.csv`
- `reports/figures/precision_recall_curve.png`
- `reports/figures/confusion_matrix.png`

The local workspace may also contain `final_xgb_model_production.json`,
`summary.xlsx`, and the original PowerPoint. They are coursework-era artifacts
from the invalid random-split experiment, are intentionally excluded from the
portfolio version, and must not be presented as current results.

## Limitations

- The evaluation covers one state and one calendar year; it is not evidence of
  geographic or multi-year generalization.
- The Kaggle listing does not document collection or simulation methodology.
  Results therefore demonstrate the modeling workflow, not performance on
  verified production banking data.
- Cards recur across time periods, so this is a known-card temporal evaluation,
  not a new-card/cold-start benchmark.
- Fraud-label availability and confirmation delays are absent. Target-derived
  history is therefore excluded.
- The review costs are hypothetical fixed amounts and omit transaction value,
  recovery rates, customer friction, staffing, and downstream losses.
- The large prepared pickle retains PII-shaped names, SSNs, card/account
  numbers, DOB, and detailed locations. Public availability does not make
  identity memorization a useful fraud signal, so these fields remain out of
  the model and out of Git.
- The original choice of California was informed by full-year fraud counts.
  The corrected experiment treats California as a fixed declared scope, but a
  new project should choose scope independently of holdout labels.
- Category drift is severe, and the locked threshold violates test-period
  review capacity. Deployment is not recommended without drift controls,
  probability calibration, a label-delay contract, and prospective shadow
  evaluation.
- Feature importance is associative and must not be interpreted causally.

The exact legacy claims that need updating in the presentation or any external
portfolio are listed in `docs/legacy_claims.md`. Those external materials were
not edited.
