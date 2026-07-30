from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from fraud_detection.features import (
    ENGINEERED_FEATURE_COLUMNS,
    build_features,
    extract_cc_bin,
)


def _transactions(
    *,
    cards: list[object],
    timestamps: list[str],
    amounts: list[float],
    fraud: list[object] | None = None,
    index: list[int] | None = None,
) -> pd.DataFrame:
    size = len(cards)
    frame = pd.DataFrame(
        {
            "cc_num": cards,
            "trans_timestamp": timestamps,
            "amt": amounts,
            "dob": ["1990-06-15"] * size,
            "lat": [34.0522] * size,
            "long": [-118.2437] * size,
            "merch_lat": [34.0522] * size,
            "merch_long": [-118.2437] * size,
        },
        index=index,
    )
    if fraud is not None:
        frame["is_fraud"] = fraud
    return frame


def test_extract_cc_bin_handles_boundaries_separators_and_missing() -> None:
    cards = pd.Series(
        [
            "123456",
            "1234567",
            "12345",
            "1234-56 7890",
            987654321,
            None,
            pd.NA,
            "12345x678",
            "--  --",
        ],
        index=list("abcdefghi"),
        name="card",
        dtype="object",
    )

    actual = extract_cc_bin(cards)

    expected = pd.Series(
        [
            "123456",
            "123456",
            pd.NA,
            "123456",
            "987654",
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
        ],
        index=cards.index,
        name="card",
        dtype="string",
    )
    pdt.assert_series_equal(actual, expected)
    assert isinstance(actual.dtype, pd.StringDtype)


def test_extract_cc_bin_accepts_integer_valued_float_series() -> None:
    # Pandas promotes an integer column containing a missing value to float.
    cards = pd.Series([123456.0, 12345.0, np.nan])
    expected = pd.Series(["123456", pd.NA, pd.NA], dtype="string")
    pdt.assert_series_equal(extract_cc_bin(cards), expected)


@pytest.mark.parametrize("digits", [0, -1, 1.5, True])
def test_extract_cc_bin_rejects_invalid_width(digits: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        extract_cc_bin(pd.Series(["123456"]), digits=digits)  # type: ignore[arg-type]


def test_cc_previous_fraud_is_hand_verified_per_card_and_strictly_past() -> None:
    raw = _transactions(
        cards=[
            4000001111111111,
            5000001111111111,
            4000001111111111,
            5000001111111111,
            4000001111111111,
        ],
        timestamps=[
            "2020-01-03 10:00:00",
            "2020-01-02 10:00:00",
            "2020-01-01 10:00:00",
            "2020-01-04 10:00:00",
            "2020-01-02 10:00:00",
        ],
        amounts=[30, 50, 10, 60, 20],
        fraud=[0, 1, 1, 0, 1],
        index=[70, 10, 90, 30, 20],
    )
    original = raw.copy(deep=True)

    featured = build_features(raw)

    # Card 4 history in chronological order is 0, 1, 2; card 5 is 0, 1.
    assert featured["CC_PREV_FRAUD"].tolist() == [2, 0, 0, 1, 1]
    assert featured["PREV_TX_COUNT"].tolist() == [2, 0, 0, 1, 1]
    assert featured["CC_HIST_FRAUD_RATE"].tolist() == [1.0, 0.0, 0.0, 1.0, 1.0]
    assert featured.index.tolist() == raw.index.tolist()
    pdt.assert_frame_equal(raw, original)


def test_equal_timestamp_transactions_do_not_see_each_other() -> None:
    raw = _transactions(
        cards=[4111111111111111] * 4,
        timestamps=[
            "2020-01-01 12:00:00",
            "2020-01-01 12:00:00",
            "2020-01-01 13:00:00",
            "2020-01-01 13:00:00",
        ],
        amounts=[10, 30, 20, 100],
        fraud=[1, 0, 0, 1],
    )

    featured = build_features(raw)

    first_bucket = featured.iloc[:2]
    assert first_bucket["PREV_TX_COUNT"].tolist() == [0, 0]
    assert first_bucket["PREV_CUMULATIVE_AMT"].tolist() == [0.0, 0.0]
    assert first_bucket["CC_PREV_FRAUD"].tolist() == [0, 0]
    assert first_bucket["IS_FIRST_CARD_TX"].tolist() == [1, 1]
    assert first_bucket["AMT_VS_PREV_MEAN"].tolist() == [0.0, 0.0]

    second_bucket = featured.iloc[2:]
    assert second_bucket["PREV_TX_COUNT"].tolist() == [2, 2]
    assert second_bucket["PREV_CUMULATIVE_AMT"].tolist() == [40.0, 40.0]
    assert second_bucket["PREV_MEAN_AMT"].tolist() == [20.0, 20.0]
    assert second_bucket["PREV_STD_AMT"].tolist() == pytest.approx([np.sqrt(200.0), np.sqrt(200.0)])
    assert second_bucket["CC_PREV_FRAUD"].tolist() == [1, 1]
    assert second_bucket["CC_HIST_FRAUD_RATE"].tolist() == [0.5, 0.5]
    assert second_bucket["TIME_SINCE_LAST_TX"].tolist() == [3600.0, 3600.0]
    assert second_bucket["IS_FIRST_CARD_TX"].tolist() == [0, 0]
    assert second_bucket["AMT_VS_PREV_MEAN"].tolist() == [1.0, 5.0]


def test_mutating_a_future_transaction_cannot_change_earlier_features() -> None:
    raw = _transactions(
        cards=[4111111111111111] * 4,
        timestamps=[
            "2020-01-01 09:00:00",
            "2020-01-02 09:00:00",
            "2020-01-03 09:00:00",
            "2020-01-04 09:00:00",
        ],
        amounts=[10, 20, 30, 40],
        fraud=[0, 1, 0, 0],
        index=[8, 3, 12, 1],
    )
    baseline = build_features(raw)

    mutated = raw.copy()
    mutated.iloc[-1, mutated.columns.get_loc("amt")] = 99_999
    mutated.iloc[-1, mutated.columns.get_loc("is_fraud")] = 1
    changed = build_features(mutated)

    pdt.assert_frame_equal(
        baseline.iloc[:-1].loc[:, ENGINEERED_FEATURE_COLUMNS],
        changed.iloc[:-1].loc[:, ENGINEERED_FEATURE_COLUMNS],
    )


def test_unsorted_input_is_restored_and_history_uses_chronology() -> None:
    raw = _transactions(
        cards=[4111111111111111] * 3,
        timestamps=[
            "2020-03-03 10:00:00",
            "2020-03-01 10:00:00",
            "2020-03-02 10:00:00",
        ],
        amounts=[60, 10, 20],
        fraud=[0, 0, 0],
        index=[5, 5, 2],
    )

    featured = build_features(raw)

    assert featured.index.tolist() == [5, 5, 2]
    assert featured["PREV_TX_COUNT"].tolist() == [2, 0, 1]
    assert featured["PREV_CUMULATIVE_AMT"].tolist() == [30.0, 0.0, 10.0]
    assert featured["PREV_MEAN_AMT"].tolist() == [15.0, 0.0, 10.0]
    assert featured["AMT_VS_PREV_MEAN"].tolist() == [4.0, 0.0, 2.0]


def test_missing_target_is_unknown_and_numeric_features_are_complete() -> None:
    raw = _transactions(
        cards=[4111111111111111, 4111111111111111, 4111111111111111],
        timestamps=[
            "2020-06-14 23:00:00",
            "2020-06-15 23:00:00",
            "2020-06-16 23:00:00",
        ],
        amounts=[0, 0, 10],
        fraud=None,
    )

    featured = build_features(raw)

    assert featured["CC_PREV_FRAUD"].tolist() == [0, 0, 0]
    assert featured["CC_HIST_FRAUD_RATE"].tolist() == [0.0, 0.0, 0.0]
    assert featured["AMT_VS_PREV_MEAN"].tolist() == [0.0, 0.0, 0.0]
    numeric_features = [column for column in ENGINEERED_FEATURE_COLUMNS if column != "CC_BIN"]
    assert np.isfinite(featured.loc[:, numeric_features].to_numpy(dtype=np.float64)).all()


def test_partial_unknown_target_contributes_zero() -> None:
    raw = _transactions(
        cards=[4111111111111111] * 3,
        timestamps=[
            "2020-01-01 00:00:00",
            "2020-01-02 00:00:00",
            "2020-01-03 00:00:00",
        ],
        amounts=[10, 20, 30],
        fraud=[1, pd.NA, 0],
    )

    featured = build_features(raw)

    assert featured["CC_PREV_FRAUD"].tolist() == [0, 1, 1]
    assert featured["CC_HIST_FRAUD_RATE"].tolist() == [0.0, 1.0, 0.5]


def test_static_features_are_transaction_time_specific() -> None:
    raw = _transactions(
        cards=[4111111111111111],
        timestamps=["2020-06-14 23:15:00"],  # Sunday, one day before birthday.
        amounts=[10],
        fraud=[0],
    )
    raw["merch_lat"] = 35.0522

    featured = build_features(raw)

    assert featured.loc[0, "CC_BIN"] == "411111"
    assert featured.loc[0, "TX_HOUR"] == 23
    assert featured.loc[0, "TX_DAY_OF_WEEK"] == 6
    assert featured.loc[0, "TX_MONTH"] == 6
    assert featured.loc[0, "IS_WEEKEND"] == 1
    assert featured.loc[0, "AGE_AT_TX"] == 29
    assert featured.loc[0, "DIST_HOME_MERCH_KM"] == pytest.approx(111.195, rel=1e-3)


def test_build_features_validates_required_columns_and_timestamps() -> None:
    raw = _transactions(
        cards=[4111111111111111],
        timestamps=["not-a-timestamp"],
        amounts=[10],
        fraud=[0],
    )
    with pytest.raises(ValueError, match="valid timestamps"):
        build_features(raw)

    with pytest.raises(ValueError, match="missing required columns: amt"):
        build_features(raw.drop(columns="amt"))
