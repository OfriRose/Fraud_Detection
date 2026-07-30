from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from fraud_detection.config import load_config
from fraud_detection.inference import load_artifact
from fraud_detection.training import run_training

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_2020_transactions() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    cards = ("4111111111111111", "5222222222222222", "6333333333333333")
    for month in range(1, 13):
        for position in range(12):
            is_fraud = int(position in {0, 7})
            records.append(
                {
                    "cc_num": cards[position % len(cards)],
                    "trans_num": f"tx-{month:02d}-{position:02d}",
                    "trans_timestamp": pd.Timestamp(
                        year=2020,
                        month=month,
                        day=position + 1,
                        hour=(position * 2) % 24,
                    ),
                    "amt": 400.0 + month if is_fraud else 10.0 + position + month,
                    "is_fraud": is_fraud,
                    "category": "online" if is_fraud else ("grocery" if position % 2 else "fuel"),
                    "profile": "urban" if position % 3 else "rural",
                    "city_pop": 50_000 + 1_000 * position,
                    "dob": "1980-06-15",
                    "lat": 34.05,
                    "long": -118.24,
                    "merch_lat": 34.05 + position / 1_000,
                    "merch_long": -118.24 - position / 1_000,
                    "is_male": position % 2,
                    "state": "CA",
                }
            )
    return pd.DataFrame.from_records(records)


def test_run_training_end_to_end_on_small_chronological_dataset(tmp_path: Path) -> None:
    transactions = _synthetic_2020_transactions()
    timestamps = pd.to_datetime(transactions["trans_timestamp"])
    train_mask = timestamps <= pd.Timestamp("2020-08-31 23:59:59")
    validation_mask = (timestamps > pd.Timestamp("2020-08-31 23:59:59")) & (
        timestamps <= pd.Timestamp("2020-10-31 23:59:59")
    )
    test_mask = timestamps > pd.Timestamp("2020-10-31 23:59:59")

    assert [int(train_mask.sum()), int(validation_mask.sum()), int(test_mask.sum())] == [96, 24, 24]
    for mask in (train_mask, validation_mask, test_mask):
        assert set(transactions.loc[mask, "is_fraud"]) == {0, 1}

    base = load_config(PROJECT_ROOT / "config" / "training.toml")
    quick_config = replace(
        base,
        project=replace(base.project, model_version="0.0.1"),
        data=replace(
            base.data,
            input_path=tmp_path / "unused-synthetic-input.csv",
            train_end=pd.Timestamp("2020-08-31 23:59:59").to_pydatetime(),
            validation_end=pd.Timestamp("2020-10-31 23:59:59").to_pydatetime(),
        ),
        paths=replace(
            base.paths,
            artifact_dir=tmp_path / "artifacts",
            report_dir=tmp_path / "reports",
            model_filename="fraud_pipeline_smoke.joblib",
        ),
        threshold=replace(base.threshold, max_review_rate=0.25),
        models=replace(
            base.models,
            logistic_regression=replace(base.models.logistic_regression, max_iter=200),
            xgboost=replace(
                base.models.xgboost,
                n_estimators=3,
                max_depth=2,
                n_jobs=1,
            ),
        ),
    )

    result = run_training(quick_config, transactions=transactions)

    assert result["artifact_path"] == tmp_path / "artifacts" / "fraud_pipeline_smoke.joblib"
    assert result["artifact_path"].is_file()
    assert result["metadata_path"].is_file()
    assert result["report_path"].is_file()
    assert (tmp_path / "reports" / "split_summary.csv").is_file()
    assert (tmp_path / "reports" / "model_comparison.csv").is_file()
    assert (tmp_path / "reports" / "metrics.json").is_file()
    assert (tmp_path / "reports" / "figures" / "precision_recall_curve.png").is_file()
    assert (tmp_path / "reports" / "figures" / "confusion_matrix.png").is_file()

    split_summary = pd.read_csv(tmp_path / "reports" / "split_summary.csv")
    assert split_summary["transactions"].tolist() == [96, 24, 24]
    assert split_summary["fraud_cases"].tolist() == [16, 4, 4]
    assert split_summary["date_end"].str[:7].tolist() == ["2020-08", "2020-10", "2020-12"]

    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert metrics["champion_model"] in {
        "prevalence_baseline",
        "logistic_regression",
        "xgboost_conservative",
        "xgboost_configured",
    }
    assert 0.0 <= metrics["locked_threshold"] <= 1.0
    assert metrics["validation"]["review_rate"] <= quick_config.threshold.max_review_rate

    artifact = load_artifact(result["artifact_path"])
    assert artifact["model_version"] == "0.0.1"
    assert artifact["metadata"]["data"]["upstream"] == {
        "name": "Credit Card Fraud Mega Dataset",
        "url": ("https://www.kaggle.com/datasets/karthikgangula/credit-card-fraud-mega-dataset"),
        "license": "MIT (as listed by the Kaggle dataset page)",
    }
    assert "is_fraud" not in artifact["feature_schema"]["model_features"]
