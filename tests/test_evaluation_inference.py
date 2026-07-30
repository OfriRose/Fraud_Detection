from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fraud_detection.config import ThresholdConfig
from fraud_detection.evaluation import select_operating_threshold
from fraud_detection.inference import InferenceError, score_transactions
from fraud_detection.preprocessing import build_pipeline


def test_threshold_selection_finds_known_cost_optimum_within_review_cap() -> None:
    labels = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    probabilities = np.array([0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10])
    capped_cost = ThresholdConfig(
        false_negative_cost=10.0,
        false_positive_cost=1.0,
        max_review_rate=0.20,
    )

    selected = select_operating_threshold(labels, probabilities, capped_cost)

    # Reviewing the top two transactions costs one FP + one FN = 11.  The
    # globally cheaper top-three choice is deliberately outside the 20% cap.
    assert selected.threshold == pytest.approx(0.90)
    assert selected.reviewed_transactions == 2
    assert selected.review_rate == pytest.approx(0.20)
    assert selected.true_positives == 1
    assert selected.false_positives == 1
    assert selected.false_negatives == 1
    assert selected.estimated_cost == pytest.approx(11.0)
    assert selected.review_rate <= capped_cost.max_review_rate

    unrestricted = select_operating_threshold(
        labels,
        probabilities,
        ThresholdConfig(
            false_negative_cost=10.0,
            false_positive_cost=1.0,
            max_review_rate=1.0,
        ),
    )
    assert unrestricted.threshold == pytest.approx(0.80)
    assert unrestricted.estimated_cost == pytest.approx(1.0)


def _artifact() -> dict[str, object]:
    training_matrix = pd.DataFrame(
        {
            "amt": [10.0, 20.0, 80.0, 100.0],
            "category": ["grocery", "fuel", "grocery", "fuel"],
        }
    )
    pipeline = build_pipeline(
        LogisticRegression(solver="liblinear", random_state=11),
        numerical_features=("amt",),
        categorical_features=("category",),
        iqr_columns=("amt",),
    )
    pipeline.fit(training_matrix, pd.Series([0, 0, 1, 1], dtype="int8"))
    return {
        "artifact_type": "fraud_detection_pipeline",
        "model_version": "9.8.7",
        "pipeline": pipeline,
        "threshold": 0.5,
        "feature_schema": {"model_features": ["amt", "category"]},
        "metadata": {
            "model_name": "test_logistic",
            "training_cutoff": "2020-08-31 23:59:59",
        },
    }


def _current_transaction() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cc_num": ["4111111111111111"],
            "trans_timestamp": ["2020-11-15 12:00:00"],
            "amt": [125.0],
            "category": ["unseen-at-training"],
            "dob": ["1985-04-20"],
            "lat": [34.05],
            "long": [-118.24],
            "merch_lat": [34.06],
            "merch_long": [-118.25],
        },
        index=pd.Index([42], name="request_id"),
    )


def test_single_row_cold_start_inference_returns_versioned_decision() -> None:
    scored = score_transactions(
        _current_transaction(),
        _artifact(),
        history=pd.DataFrame(),
    )

    assert scored.index.tolist() == [42]
    assert scored.index.name == "request_id"
    assert {
        "fraud_probability",
        "fraud_decision",
        "model_version",
        "threshold",
    }.issubset(scored.columns)
    assert np.isfinite(scored.loc[42, "fraud_probability"])
    assert 0.0 <= scored.loc[42, "fraud_probability"] <= 1.0
    assert scored.loc[42, "fraud_decision"] == int(
        scored.loc[42, "fraud_probability"] >= scored.loc[42, "threshold"]
    )
    assert scored.loc[42, "model_version"] == "9.8.7"


@pytest.mark.parametrize(
    "history_timestamp",
    ["2020-11-15 12:00:00", "2020-11-16 12:00:00"],
    ids=["equal", "future"],
)
def test_inference_rejects_equal_or_future_history(history_timestamp: str) -> None:
    history = pd.DataFrame(
        {
            "cc_num": ["4111111111111111"],
            "trans_timestamp": [history_timestamp],
        }
    )

    with pytest.raises(InferenceError, match="must precede"):
        score_transactions(_current_transaction(), _artifact(), history=history)
