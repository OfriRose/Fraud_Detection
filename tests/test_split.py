from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from fraud_detection.split import (
    DatasetSplits,
    TemporalSplitConfig,
    assert_disjoint_splits,
    chronological_split,
    summarize_splits,
)


def _transactions() -> pd.DataFrame:
    # Deliberately unsorted, with stable non-positional indices.
    return pd.DataFrame(
        {
            "trans_timestamp": [
                "2024-01-03 00:00:01",
                "2024-01-01 12:00:00",
                "2024-01-02 00:00:00",
                "2024-01-03 00:00:00",
                "2024-01-01 23:59:59",
                "2024-01-02 12:00:00",
            ],
            "is_fraud": [1, 0, 1, 0, 0, 1],
        },
        index=pd.Index([106, 101, 103, 105, 102, 104], name="transaction_id"),
    )


def _config() -> TemporalSplitConfig:
    return TemporalSplitConfig(
        train_end="2024-01-01 23:59:59",
        validation_end="2024-01-03 00:00:00",
    )


def test_temporal_split_boundaries_are_explicit_and_indices_are_preserved() -> None:
    splits = chronological_split(_transactions(), _config())

    assert splits.train.index.tolist() == [101, 102]
    assert splits.validation.index.tolist() == [103, 104, 105]
    assert splits.test.index.tolist() == [106]

    # Equality goes to the earlier split at both boundaries.
    assert splits.train.index[-1] == 102
    assert splits.validation.index[-1] == 105
    assert all(
        pd.api.types.is_datetime64_any_dtype(frame["trans_timestamp"])
        for frame in (splits.train, splits.validation, splits.test)
    )
    assert splits.train.index.name == "transaction_id"


def test_config_is_immutable_and_validates_order() -> None:
    config = _config()

    with pytest.raises(FrozenInstanceError):
        config.train_end = pd.Timestamp("2024-01-02")  # type: ignore[misc]

    with pytest.raises(ValueError, match="earlier"):
        TemporalSplitConfig("2024-01-03", "2024-01-03")


def test_split_indices_have_exact_pairwise_non_overlap() -> None:
    splits = chronological_split(_transactions(), _config())

    assert_disjoint_splits(splits)
    all_indices = (
        splits.train.index.tolist() + splits.validation.index.tolist() + splits.test.index.tolist()
    )
    assert len(all_indices) == len(set(all_indices)) == len(_transactions())
    assert set(all_indices) == set(_transactions().index)

    overlapping = DatasetSplits(
        train=splits.train,
        validation=splits.validation,
        test=pd.concat([splits.test, splits.train.iloc[[0]]]),
    )
    with pytest.raises(AssertionError, match="overlap"):
        assert_disjoint_splits(overlapping)


def test_summarize_splits_reports_dates_counts_and_fraud_rates() -> None:
    summary = summarize_splits(chronological_split(_transactions(), _config()))

    assert summary.columns.tolist() == [
        "split",
        "date_start",
        "date_end",
        "transactions",
        "fraud_cases",
        "fraud_rate",
    ]
    assert summary["split"].tolist() == ["train", "validation", "test"]
    assert summary["transactions"].tolist() == [2, 3, 1]
    assert summary["fraud_cases"].tolist() == [0, 2, 1]
    assert summary["fraud_rate"].tolist() == pytest.approx([0.0, 2 / 3, 1.0])
    assert summary["date_start"].tolist() == [
        pd.Timestamp("2024-01-01 12:00:00"),
        pd.Timestamp("2024-01-02 00:00:00"),
        pd.Timestamp("2024-01-03 00:00:01"),
    ]
    assert summary["date_end"].tolist() == [
        pd.Timestamp("2024-01-01 23:59:59"),
        pd.Timestamp("2024-01-03 00:00:00"),
        pd.Timestamp("2024-01-03 00:00:01"),
    ]


@pytest.mark.parametrize("bad_timestamp", [None, "not-a-date"])
def test_split_rejects_missing_or_invalid_timestamps(bad_timestamp: object) -> None:
    transactions = _transactions()
    transactions.loc[101, "trans_timestamp"] = bad_timestamp

    with pytest.raises(ValueError, match="missing or invalid timestamp"):
        chronological_split(transactions, _config())


def test_split_rejects_empty_partitions() -> None:
    with pytest.raises(ValueError, match="empty partition"):
        chronological_split(
            _transactions(),
            TemporalSplitConfig(
                train_end="2023-12-01",
                validation_end="2023-12-31",
            ),
        )


def test_custom_timestamp_column_is_used_by_summary() -> None:
    transactions = _transactions().rename(columns={"trans_timestamp": "event_time"})
    config = TemporalSplitConfig(
        train_end="2024-01-01 23:59:59",
        validation_end="2024-01-03 00:00:00",
        timestamp_col="event_time",
    )

    summary = summarize_splits(chronological_split(transactions, config))

    assert summary.loc[0, "date_start"] == pd.Timestamp("2024-01-01 12:00:00")
