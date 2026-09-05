"""Chronological dataset splitting and split-level reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def _parse_boundary(value: Any, field_name: str) -> pd.Timestamp:
    """Parse and validate a single split boundary."""
    try:
        boundary = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a valid datetime") from exc

    if pd.isna(boundary):
        raise ValueError(f"{field_name} must be a valid datetime")
    return boundary


@dataclass(frozen=True, slots=True)
class TemporalSplitConfig:
    """Immutable boundaries for a train/validation/test temporal split.

    Transactions at a boundary belong to the earlier period: training includes
    ``train_end`` and validation includes ``validation_end``.
    """

    train_end: Any
    validation_end: Any
    timestamp_col: str = "trans_timestamp"

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_col, str) or not self.timestamp_col.strip():
            raise ValueError("timestamp_col must be a non-empty string")

        train_end = _parse_boundary(self.train_end, "train_end")
        validation_end = _parse_boundary(self.validation_end, "validation_end")
        try:
            boundaries_are_ordered = train_end < validation_end
        except TypeError as exc:
            raise ValueError(
                "train_end and validation_end must use compatible timezone information"
            ) from exc
        if not boundaries_are_ordered:
            raise ValueError("train_end must be earlier than validation_end")

        object.__setattr__(self, "train_end", train_end)
        object.__setattr__(self, "validation_end", validation_end)
        object.__setattr__(self, "timestamp_col", self.timestamp_col.strip())


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """The three chronologically ordered dataset partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _named_splits(splits: DatasetSplits) -> tuple[tuple[str, pd.DataFrame], ...]:
    if not isinstance(splits, DatasetSplits):
        raise TypeError("splits must be a DatasetSplits instance")
    return (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    )


def assert_disjoint_splits(splits: DatasetSplits) -> None:
    """Raise ``AssertionError`` if any original index occurs in two splits."""
    named_splits = _named_splits(splits)
    for name, frame in named_splits:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} split must be a pandas DataFrame")

    for left_position, (left_name, left_frame) in enumerate(named_splits):
        for right_name, right_frame in named_splits[left_position + 1 :]:
            overlap = left_frame.index.intersection(right_frame.index)
            if not overlap.empty:
                raise AssertionError(
                    f"{left_name} and {right_name} split indices overlap "
                    f"({len(overlap)} index value(s))"
                )


def _assert_strictly_ordered_periods(splits: DatasetSplits, timestamp_col: str) -> None:
    """Ensure every period ends before the following period starts."""
    pairs = (
        ("train", splits.train, "validation", splits.validation),
        ("validation", splits.validation, "test", splits.test),
    )
    for left_name, left, right_name, right in pairs:
        left_end = left[timestamp_col].max()
        right_start = right[timestamp_col].min()
        if not left_end < right_start:
            raise AssertionError(f"{left_name} and {right_name} periods are not strictly ordered")


def chronological_split(df: pd.DataFrame, config: TemporalSplitConfig) -> DatasetSplits:
    """Split transactions without shuffling, preserving their original indices.

    The returned frames are stably sorted by timestamp. Boundary equality is
    handled explicitly: ``train_end`` is assigned to train and
    ``validation_end`` is assigned to validation.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(config, TemporalSplitConfig):
        raise TypeError("config must be a TemporalSplitConfig instance")
    if config.timestamp_col not in df.columns:
        raise ValueError(f"timestamp column {config.timestamp_col!r} is missing from the dataset")

    working = df.copy()
    try:
        parsed_timestamps = pd.to_datetime(
            working[config.timestamp_col], errors="coerce", format="mixed"
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{config.timestamp_col!r} contains invalid timestamps") from exc

    invalid_timestamp_count = int(parsed_timestamps.isna().sum())
    if invalid_timestamp_count:
        raise ValueError(
            f"{config.timestamp_col!r} contains {invalid_timestamp_count} "
            "missing or invalid timestamp(s)"
        )
    working[config.timestamp_col] = parsed_timestamps

    try:
        train_mask = parsed_timestamps <= config.train_end
        validation_mask = (parsed_timestamps > config.train_end) & (
            parsed_timestamps <= config.validation_end
        )
        test_mask = parsed_timestamps > config.validation_end
    except TypeError as exc:
        raise ValueError(
            "transaction timestamps and split boundaries must use compatible timezone information"
        ) from exc

    def select(mask: pd.Series) -> pd.DataFrame:
        return working.loc[mask].sort_values(config.timestamp_col, kind="mergesort")

    splits = DatasetSplits(
        train=select(train_mask),
        validation=select(validation_mask),
        test=select(test_mask),
    )

    empty_splits = [name for name, frame in _named_splits(splits) if frame.empty]
    if empty_splits:
        raise ValueError(
            "chronological split produced empty partition(s): " + ", ".join(empty_splits)
        )

    assert_disjoint_splits(splits)
    _assert_strictly_ordered_periods(splits, config.timestamp_col)
    return splits


def summarize_splits(
    splits: DatasetSplits,
    target_col: str = "is_fraud",
    *,
    timestamp_col: str = "trans_timestamp",
) -> pd.DataFrame:
    """Return date ranges, transaction counts, and fraud prevalence by split."""
    if not isinstance(target_col, str) or not target_col:
        raise ValueError("target_col must be a non-empty string")

    named_splits = _named_splits(splits)
    empty_splits = [name for name, frame in named_splits if frame.empty]
    if empty_splits:
        raise ValueError("cannot summarize empty partition(s): " + ", ".join(empty_splits))

    rows: list[dict[str, Any]] = []
    for split_name, frame in named_splits:
        if timestamp_col not in frame.columns:
            raise ValueError(
                f"timestamp column {timestamp_col!r} is missing from {split_name} split"
            )
        if target_col not in frame.columns:
            raise ValueError(f"target column {target_col!r} is missing from {split_name} split")

        timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce", format="mixed")
        if timestamps.isna().any():
            raise ValueError(
                f"{timestamp_col!r} contains missing or invalid timestamps in {split_name} split"
            )

        target = frame[target_col]
        if target.isna().any():
            raise ValueError(f"{target_col!r} contains missing values in {split_name} split")
        if not target.isin([0, 1, False, True]).all():
            raise ValueError(f"{target_col!r} must contain only binary values")

        transaction_count = len(frame)
        fraud_cases = int(target.astype(int).sum())
        rows.append(
            {
                "split": split_name,
                "date_start": timestamps.min(),
                "date_end": timestamps.max(),
                "transactions": transaction_count,
                "fraud_cases": fraud_cases,
                "fraud_rate": fraud_cases / transaction_count,
            }
        )

    return pd.DataFrame.from_records(
        rows,
        columns=[
            "split",
            "date_start",
            "date_end",
            "transactions",
            "fraud_cases",
            "fraud_rate",
        ],
    )
