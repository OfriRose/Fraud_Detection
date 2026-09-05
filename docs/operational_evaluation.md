# Operational evaluation, drift, and limitations

This document provides the context behind the concise README result. Generated source files are kept in [`reports/`](../reports/), including the [evaluation report](../reports/evaluation.md), [monthly test metrics](../reports/test_monthly_metrics.csv), and [train/test drift diagnostics](../reports/train_test_drift.csv).

## Operating point and capacity

The locked threshold, costs, and final outcomes are recorded in the generated
[evaluation report](../reports/evaluation.md) and [metrics](../reports/metrics.json).

The 5% condition applies during **validation threshold selection**. It is implemented by `select_operating_threshold` in [`src/fraud_detection/evaluation.py`](../src/fraud_detection/evaluation.py), which permits at most `floor(max_review_rate × validation rows)` reviewed transactions. Test labels are not passed to that function, and the threshold is not retuned on test.

The saved version 1.1.0 run stays below the configured review-rate target on
test. This is an observed outcome, not a guaranteed future capacity bound.
Absolute review counts cannot be compared directly across periods with different
transaction counts.

The scenario cost is an analytical comparison aid, not realized savings. It omits transaction amount, recoveries, staffing, customer friction, downstream losses, and any difference between review and blocking outcomes.

## Monthly and drift findings

The [monthly metrics](../reports/test_monthly_metrics.csv) show lower December
fraud prevalence and precision at the fixed threshold. This is a descriptive
association, not a causal attribution.

The [drift report](../reports/train_test_drift.csv) shows a large category shift,
expected shifts in calendar month and accumulating card history, and changes in
recent transaction velocity. Feature importance is associative, and these
diagnostics do not establish why performance changed.

## Extended limitations

- Evaluation covers one state and calendar year and recurring cards across periods. It is not evidence of new-card, geographic, or multi-year generalization.
- Dataset collection/simulation methods are undocumented on the source listing.
- There is no fraud-label availability or confirmation-delay timestamp; target-derived history remains disabled.
- The input contains PII-shaped fields. They are excluded from modeling and the prepared data is not committed, but public availability is not a basis for identity-based modeling.
- California was selected in legacy preparation using full-year fraud counts. The rebuilt experiment declares it as fixed scope; a future study should select scope without holdout labels.
- Deployment would require drift controls, calibrated probabilities, a label-delay contract, and prospective/shadow evaluation.

For the complete audit trail and legacy result reconciliation, see [audit.md](audit.md) and [legacy_claims.md](legacy_claims.md).
