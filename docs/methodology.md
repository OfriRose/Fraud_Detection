# Methodology and implementation notes

This document expands the concise methodology in the project README. The authoritative implementation is in [`src/fraud_detection/`](../src/fraud_detection/).

## Experiment scope and split

The configured experiment uses California transactions from 2020 from the [Credit Card Fraud Mega Dataset](https://www.kaggle.com/datasets/karthikgangula/credit-card-fraud-mega-dataset). The source listing identifies an MIT license but does not document collection or simulation methodology; the project does not present it as verified bank production data.

[training.toml](../config/training.toml) fixes the chronological boundaries.
Split indices are asserted disjoint. Exact dates, counts, and fraud prevalence
are generated in the [split summary](../reports/split_summary.csv).

Validation chooses the candidate model and operating threshold. Test is not used for preprocessing, feature selection, hyperparameter selection, model selection, or threshold selection.

## Leakage controls and features

- Transactions are ordered by card and event time. Equal-time records are processed as a bucket, so they cannot contribute history to one another.
- Previous transaction count, cumulative amount, mean, sample standard deviation, and time since prior transaction use only earlier timestamps.
- Per-card 1-hour, 24-hour, and 7-day counts, maximum amounts, and mean amounts use `[timestamp - window, timestamp)` boundaries.
- First-card-event historical values are zero and `IS_FIRST_CARD_TX` records that state.
- Raw identifiers are excluded from modeling. `CC_BIN` is not generated.
- Target-derived history is neither generated nor configurable. The source has no `label_available_at` timestamp, so past labels cannot safely be assumed observable at scoring.
- Legacy global quantile bins were removed. The model does not require them.

The surviving `prepped_data.pkl` was created by a legacy notebook that consolidated `city`, `job`, and `merchant` over the complete year. These three columns are explicitly excluded to prevent that pre-split transformation from entering the model.

## Preprocessing and selection

Learned candidates use complete scikit-learn pipelines. The prevalence baseline uses only training-label frequencies and skips preprocessing. IQR clipping bounds, imputers, scalers, and one-hot category vocabularies are fitted on training data only; categorical encoding uses `handle_unknown="ignore"`. The target is asserted absent from each feature matrix, and no resampling is performed.

Fraud is rare, so accuracy is not a selection metric. Logistic regression is
an interpretable baseline with `class_weight="balanced"`; XGBoost captures
nonlinear interactions and uses the training negative/positive ratio for
`scale_pos_weight`. Two XGBoost settings provide a small validation comparison
without a large search. Validation PR-AUC selects the candidate; lower-ranking
alternatives are retained in the [model comparison](../reports/model_comparison.csv).
The selected model locks its threshold by minimizing the configured error cost
under validation review capacity. Exact rules are in
[evaluation.py](../src/fraud_detection/evaluation.py).

## Artifact and inference contract

The versioned artifact contains the fitted preprocessing/model pipeline, feature schema, threshold, and metadata. It is loaded with `fraud_detection.inference.load_artifact`; `score_transactions` returns a class-weighted risk score and decision at the locked threshold.

For an existing card, callers must provide only history known before each current transaction through `history=`. The inference interface rejects equal-time and future history. Scores are not calibrated probabilities of observed production fraud.

The full legacy failure analysis, including historical-feature and preprocessing leakage, is in the [audit](audit.md).

## Scoring example

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


Use the module training command because console-script shebangs are not portable
when the repository path contains spaces.
