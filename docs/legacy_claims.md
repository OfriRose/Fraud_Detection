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
  surviving prepared pickle retains SSN and other PII-shaped identifiers.
- Any repository, slide, or résumé wording that calls this a verified
  “real-world” banking dataset should instead name and link the Kaggle source.
  Its listing does not document collection or generation methodology.

The PowerPoint was not edited because the user requested a claim list rather
than authorizing changes to external-facing portfolio material.

## Safe replacement wording

Use wording such as:

> Rebuilt a fraud-detection study around strict-past behavioral features,
> train-only preprocessing, chronological validation, validation-locked
> thresholding, and reproducible pipeline artifacts. A validation-only
> ablation restored corrected velocity features and improved latest-period
> performance, while substantial category drift still limits deployment
> claims.

If exact results are appropriate, label them as the corrected CA/2020
experiment:

- Validation PR-AUC 0.9799; validation precision 52.44% and recall 98.93% at
  the validation-locked threshold.
- Test PR-AUC 0.9543; test precision 18.19% and recall 99.18%.
- Test confusion counts: TP 1,454, FN 12, FP 6,540, TN 443,131.
- Test review load 1.77%, below the 5% validation capacity target. This does
  not establish production readiness because evaluation covers one state and
  year from a dataset with undocumented collection or simulation methodology.

Do not call the model production-ready or describe the cost scenario as
realized savings.
