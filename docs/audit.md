# Original implementation audit

This audit records the repository state before the corrected architecture was
introduced. No original performance claim should be attributed to the
chronological pipeline.

## Original flow

1. `[1]Data_Prep.ipynb` read the raw CSV, filtered 2020, selected California
   using the largest full-year fraud count, learned full-data top categories,
   normalized fields, and wrote `prepped_data.pkl`.
2. `[3]Data_cleanse.ipynb` learned IQR bounds from the complete scoped data,
   created an outlier flag, capped `amt` and `city_pop`, and claimed to write
   `clensed_data.pkl`.
3. `[4]Feature_Engineering.ipynb` built calendar, velocity, lifetime,
   recurrence, distance, and identity features; ran full-data `qcut`, one-hot
   encoding, and scaling; then used a shuffled stratified 70/15/15 split.
4. The same notebook selected 20 features using four fitted models.
5. `[5]Model Selection and Fine Tuning.ipynb` compared candidates and tried
   three XGBoost settings on the random validation partition.
6. `[6]Model Evaluation.ipynb` used the model’s default 0.5 decision threshold,
   evaluated the random test partition, generated plots/SHAP output, and saved
   a bare XGBoost estimator.

## Leakage and correctness findings

- IQR bounds/caps, quantile bins, category vocabularies, OHE columns, and
  scaling were all learned before splitting.
- The random split mixed the complete year across every partition. All 302,092
  original test rows shared a card and SSN with training; 2,139 of 2,171 cards
  appeared in both.
- Full-year user amount mean, card transaction count, SSN/account cardinality,
  and top-category lists used future observations.
- `CC_PREV_FRAUD` computed a grouped cumulative target and then shifted
  globally. The first transaction of 1,116 cards inherited another card’s
  history. Because the split was random, holdout labels also entered training
  histories.
- `CC_BIN = df["cc_num"][:6]` sliced six rows rather than six digits. Stored
  output showed only six non-null values.
- Rolling features used `rolling(...).shift(1)`, which shifts the previous
  row’s window rather than creating an exact window ending at the current
  timestamp. Equal-time ordering was undefined.
- Feature engineering duplicated the card velocity block.
- The requested adjacent duplicate `try:` block was not present in any working
  notebook, nonempty checkpoint, or the staged deleted notebook. All existing
  `try` blocks had matching `except` clauses. It appears to have existed in an
  earlier version; the notebook refactor removes all core exception logic.

## Modeling and reporting findings

- There was no explicit threshold-selection stage or persisted threshold.
- Candidate selection emphasized ROC-AUC; PR-AUC and operational cost were not
  reported.
- No confusion counts, review load, time-slice results, uncertainty, or drift
  report existed.
- Notebook 5 selected XGBoost set 1 (150 trees, depth 6, learning rate 0.1).
  Notebook 6 called it set 2 but hard-coded set 1. The README instead claimed
  depth 8 and learning rate 0.05.
- The README claimed test precision 0.5720, recall 0.9767, and F1 0.7215. The
  retained evaluation output showed 0.434830, 0.984088, and 0.603151.
- The README and presentation described the result as unseen, unbiased,
  robust, and production-ready despite the leakage and random split.
- The saved XGBoost JSON omitted all preprocessing, feature-state rules,
  schema, threshold, and versioned evaluation metadata.

## Reproducibility and artifact findings

- The original repository had no `pyproject.toml`, lock file, `.gitignore`,
  source package, tests, CI, or clean training command.
- The evaluation notebook referenced `plt` before importing it and depended on
  out-of-order notebook state.
- Embedded output accounted for almost all of the 7.3 MB EDA notebook.
- README/notebook claims referenced missing `clensed_data.pkl`, selected
  features, split pickles, and plot files.
- The README named `lasso_selected_features.npy`; code wrote
  `selected_features.npy`; neither existed.
- `prepped_data.pkl` retained sensitive identifiers despite a PII-removal
  claim.
- Git had no commits. The 11.19 GB raw CSV and an older data-cleaning notebook
  were staged additions but deleted from the working tree. That unusual state
  was preserved; no reset, unstage, restore, or data deletion was performed.

## Corrective outcome

The corrected implementation now provides:

- strict-past, equal-timestamp-safe history features;
- explicit chronological boundaries and split assertions;
- a train-fitted scikit-learn preprocessing/model pipeline;
- validation-only candidate and operating-threshold selection;
- one final latest-period test evaluation;
- PR-AUC, fraud-class metrics, confusion counts, review load, scenario cost,
  time slices, drift, uncertainty, and non-causal feature importance;
- a versioned pipeline/threshold/metadata artifact;
- validated single-row inference;
- a locked Poetry environment and automated regression/smoke tests.

Exact corrected results are generated in `reports/evaluation.md`.
