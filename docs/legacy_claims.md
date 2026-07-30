# Legacy claims requiring updates

The original random-split experiment is invalidated by preprocessing and
historical-feature leakage. Do not retain these claims in a portfolio, résumé,
presentation, or interview narrative.

## Repository presentation

- Slide 1: replace “Achieving 97.7% Fraud Detection.”
- Slide 9: remove the five-model/XGBoost consensus claim; the old feature
  selection used four models and was itself downstream of leakage.
- Slide 10: remove “universal predictive power.”
- Slide 12: replace the old random-validation model table.
- Slide 13: remove 0.9977 AUC, 97.7% recall, 57.2% precision, and 0.72 F1.
- Slide 14: remove the embedded legacy PR/ROC curves.
- Slide 15: remove “could save millions,” “manageable alert queue,”
  “production-ready,” “robustly generalizes,” and “deploying immediately.”
- Slide 8: remove nonexistent `ZIP_FRAUD_RATE` and clarify that
  `CC_PREV_FRAUD` is corrected but excluded without label-availability data.
- Slide 3: correct the statement that raw SSN was immediately dropped; the
  surviving prepared pickle retains SSN and other sensitive identifiers.

The PowerPoint was not edited because the user requested a claim list rather
than authorizing changes to external-facing portfolio material.

## Safe replacement wording

Use wording such as:

> Rebuilt a fraud-detection study around strict-past behavioral features,
> train-only preprocessing, chronological validation, validation-locked
> thresholding, and reproducible pipeline artifacts. The latest-period test
> exposed substantial category drift and alert-capacity risk that a random
> split had hidden.

If exact results are appropriate, label them as the corrected CA/2020
experiment:

- Validation PR-AUC 0.9020; validation precision 23.90% and recall 92.88% at
  the validation-locked threshold.
- Test PR-AUC 0.7798; test precision 2.78% and recall 94.61%.
- Test confusion counts: TP 1,387, FN 79, FP 48,592, TN 401,079.
- Test review load 11.08%, which exceeds the 5% validation capacity target and
  is a deployment blocker rather than a success claim.

Do not call the model production-ready or describe the cost scenario as
realized savings.
