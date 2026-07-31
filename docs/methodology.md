# Methodology and implementation notes

This document expands the concise methodology in the project README. The authoritative implementation is in [`src/fraud_detection/`](../src/fraud_detection/).

## Experiment scope and split

The configured experiment uses California transactions from 2020 from the [Credit Card Fraud Mega Dataset](https://www.kaggle.com/datasets/karthikgangula/credit-card-fraud-mega-dataset). The source listing identifies an MIT license but does not document collection or simulation methodology; the project does not present it as verified bank production data.

`config/training.toml` fixes the boundaries below. Split indices are asserted to be pairwise disjoint.

| Split | Date range | Transactions | Fraud cases | Fraud rate |
|---|---|---:|---:|---:|
| Train | 2020-01-01 00:01:13–2020-08-31 23:59:59 | 1,260,902 | 7,561 | 0.5997% |
| Validation | 2020-09-01 00:58:20–2020-10-31 23:59:56 | 301,906 | 1,867 | 0.6184% |
| Test | 2020-11-01 00:23:09–2020-12-31 23:59:48 | 451,137 | 1,466 | 0.3250% |

Validation chooses the candidate model and operating threshold. Test is not used for preprocessing, feature selection, hyperparameter selection, model selection, or threshold selection.

## Leakage controls and features

- Transactions are ordered by card and event time. Equal-time records are processed as a bucket, so they cannot contribute history to one another.
- Previous transaction count, cumulative amount, mean, sample standard deviation, and time since prior transaction use only earlier timestamps.
- Per-card 1-hour, 24-hour, and 7-day counts, maximum amounts, and mean amounts use `[timestamp - window, timestamp)` boundaries.
- First-card-event historical values are zero and `IS_FIRST_CARD_TX` records that state.
- `CC_BIN` extracts the first six card digits, but both it and raw identifiers are excluded from modeling.
- Corrected prior-fraud and historical-fraud-rate features are generated and regression-tested but disabled. The source has no `label_available_at` timestamp, so past labels cannot safely be assumed observable at scoring.
- Legacy global quantile bins were removed. The model does not require them.

The surviving `prepped_data.pkl` was created by a legacy notebook that consolidated `city`, `job`, and `merchant` over the complete year. These three columns are explicitly excluded to prevent that pre-split transformation from entering the model.

## Preprocessing and selection

Every candidate is a complete scikit-learn pipeline. IQR clipping bounds, imputers, scalers, and one-hot category vocabularies are fitted on training data only; categorical encoding uses `handle_unknown="ignore"`. The target is asserted absent from each feature matrix, and no resampling is performed.

Fraud is rare (0.5997% of train and 0.6184% of validation transactions), so accuracy is not a selection metric. Logistic regression uses `class_weight="balanced"`; XGBoost uses `scale_pos_weight` calculated as the training negative/positive ratio. Candidates are compared by validation PR-AUC, then reported with fraud-class precision, recall, and review volume. The selected configured XGBoost model locks its threshold by minimizing `false negatives × $500 + false positives × $5`, subject to reviewing no more than 5% of validation transactions. Exact implementation details are in [`evaluation.py`](../src/fraud_detection/evaluation.py) and fixed configuration is in [`training.toml`](../config/training.toml).

## Artifact and inference contract

The versioned artifact contains the fitted preprocessing/model pipeline, feature schema, threshold, and metadata. It is loaded with `fraud_detection.inference.load_artifact`; `score_transactions` returns a class-weighted risk score and decision at the locked threshold.

For an existing card, callers must provide only history known before each current transaction through `history=`. The inference interface rejects equal-time and future history. Scores are not calibrated probabilities of observed production fraud.

The full legacy failure analysis, including historical-feature and preprocessing leakage, is in the [audit](audit.md).
