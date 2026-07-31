# Operational evaluation, drift, and limitations

This document provides the context behind the concise README result. Generated source files are kept in [`reports/`](../reports/), including the [evaluation report](../reports/evaluation.md), [monthly test metrics](../reports/test_monthly_metrics.csv), and [train/test drift diagnostics](../reports/train_test_drift.csv).

## Operating point and capacity

The selected threshold is `0.6991600990`. It was selected on validation data by minimizing a scenario cost of $500 per false negative and $5 per false positive, subject to a maximum review rate of 5%.

The 5% condition applies during **validation threshold selection**. It is implemented by `select_operating_threshold` in [`src/fraud_detection/evaluation.py`](../src/fraud_detection/evaluation.py), which permits at most `floor(max_review_rate × validation rows)` reviewed transactions. Test labels are not passed to that function, and the threshold is not retuned on test.

At the locked threshold, validation reviewed 3,522 of 301,906 transactions (1.1666%). Test reviewed 7,994 of 451,137 (1.7720%). Therefore the test result is below the 5% configured limit; it does **not** violate review capacity under the only configured capacity definition. The rate increased by 0.6054 percentage points, or about 52% relative to validation. Absolute review counts cannot be compared directly because the periods contain different numbers of transactions.

## Final test outcomes

| Metric | Test result |
|---|---:|
| PR-AUC | 0.954273 |
| Precision | 18.1886% (95% Wilson: 17.36%–19.05%) |
| Recall | 99.1814% (95% Wilson: 98.57%–99.53%) |
| Fraud detected / missed | 1,454 / 12 |
| Legitimate flagged | 6,540 |
| Review rate | 1.7720% |
| Estimated scenario cost | $38,700 |

The scenario cost is an analytical comparison aid, not realized savings. It omits transaction amount, recoveries, staffing, customer friction, downstream losses, and any difference between review and blocking outcomes.

## Monthly and drift findings

November reviewed 2.0463% of transactions with 24.57% precision; December reviewed 1.6337% with 14.16% precision. Recall remained 98.83% and 99.57%, respectively. The lower December fraud prevalence (0.2323%, versus 0.5088% in November) is consistent with lower precision at the fixed threshold; this is an interpretation of the reported metrics, not a causal attribution.

The train/test drift file reports category total-variation distance of 0.9590; large expected shifts in calendar month (standardized mean difference 3.02) and accumulating card history (prior count 2.42 and cumulative amount 2.29); and higher test 24-hour and 7-day transaction counts (0.68 and 0.98). These history shifts are partly expected because the features accumulate over time, but they still warrant monitoring. Feature importance is associative, not causal, and drift diagnostics do not establish why performance changed.

## Extended limitations

- Evaluation covers one state and calendar year and recurring cards across periods. It is not evidence of new-card, geographic, or multi-year generalization.
- Dataset collection/simulation methods are undocumented on the source listing.
- There is no fraud-label availability or confirmation-delay timestamp; target-derived history remains disabled.
- The input contains PII-shaped fields. They are excluded from modeling and the prepared data is not committed, but public availability is not a basis for identity-based modeling.
- California was selected in legacy preparation using full-year fraud counts. The rebuilt experiment declares it as fixed scope; a future study should select scope without holdout labels.
- Deployment would require drift controls, calibrated probabilities, a label-delay contract, and prospective/shadow evaluation.

For the complete audit trail and legacy result reconciliation, see [audit.md](audit.md) and [legacy_claims.md](legacy_claims.md).
