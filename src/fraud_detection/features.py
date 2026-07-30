"""Leakage-safe, deterministic transaction feature engineering.

All behavioral features in this module are based on transactions at strictly
earlier timestamps for the same card.  Rows sharing a timestamp are processed
as one bucket, so their amounts and labels cannot leak into one another.
"""

from __future__ import annotations

import numbers
from typing import Final

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "cc_num",
    "trans_timestamp",
    "amt",
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
)

ENGINEERED_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "CC_BIN",
    "TX_HOUR",
    "TX_DAY_OF_WEEK",
    "TX_MONTH",
    "IS_WEEKEND",
    "AGE_AT_TX",
    "DIST_HOME_MERCH_KM",
    "PREV_TX_COUNT",
    "PREV_CUMULATIVE_AMT",
    "PREV_MEAN_AMT",
    "PREV_STD_AMT",
    "CC_PREV_FRAUD",
    "CC_HIST_FRAUD_RATE",
    "TIME_SINCE_LAST_TX",
    "IS_FIRST_CARD_TX",
    "AMT_VS_PREV_MEAN",
)

_NUMERIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in ENGINEERED_FEATURE_COLUMNS if column != "CC_BIN"
)
_HISTORY_ZERO_AT_COLD_START: Final[tuple[str, ...]] = (
    "PREV_TX_COUNT",
    "PREV_CUMULATIVE_AMT",
    "PREV_MEAN_AMT",
    "PREV_STD_AMT",
    "CC_PREV_FRAUD",
    "CC_HIST_FRAUD_RATE",
    "TIME_SINCE_LAST_TX",
    "AMT_VS_PREV_MEAN",
)
_EARTH_RADIUS_KM: Final[float] = 6_371.0088


def _stringify_card_values(values: pd.Series) -> pd.Series:
    """Return nullable strings without adding ``.0`` to integer-valued floats."""

    text = values.astype("string")

    if pd.api.types.is_float_dtype(values.dtype):
        numeric = pd.to_numeric(values, errors="coerce")
        integral = numeric.notna() & np.isfinite(numeric) & numeric.eq(numeric.round())
        if integral.any():
            text.loc[integral] = numeric.loc[integral].map(lambda value: str(int(value)))
    elif pd.api.types.is_object_dtype(values.dtype):
        # Mixed object Series are common in validation tests and ingestion code.
        # Only the relatively unusual float elements need scalar normalization.
        float_positions = (
            values.map(
                lambda value: (
                    isinstance(value, float | np.floating)
                    and np.isfinite(value)
                    and float(value).is_integer()
                ),
                na_action="ignore",
            )
            .astype("boolean")
            .fillna(False)
        )
        if float_positions.any():
            text.loc[float_positions] = values.loc[float_positions].map(
                lambda value: str(int(value))
            )

    return text


def extract_cc_bin(values: pd.Series, digits: int = 6) -> pd.Series:
    """Extract a fixed-width issuer-identification prefix from card values.

    Spaces and hyphens are ignored.  A value must otherwise contain only
    digits and have at least ``digits`` digits.  Invalid, short, or missing
    values become :data:`pandas.NA`.  The returned Series always uses pandas'
    nullable :class:`~pandas.StringDtype` and preserves the input index/name.

    Parameters
    ----------
    values:
        Card-number values.
    digits:
        Number of leading digits to return.  Must be a positive integer.
    """

    if not isinstance(values, pd.Series):
        raise TypeError("values must be a pandas Series")
    if isinstance(digits, bool) or not isinstance(digits, numbers.Integral) or digits <= 0:
        raise ValueError("digits must be a positive integer")

    width = int(digits)
    cleaned = _stringify_card_values(values).str.replace(r"[\s-]+", "", regex=True)
    valid = cleaned.str.fullmatch(r"[0-9]+", na=False) & cleaned.str.len().ge(width)
    result = cleaned.str.slice(stop=width).where(valid, pd.NA).astype("string")
    result.name = values.name
    return result


def _parse_required_datetimes(values: pd.Series, column: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{column!r} must contain valid timestamps") from exc

    if not isinstance(parsed, pd.Series):
        parsed = pd.Series(parsed, index=values.index, name=values.name)
    if parsed.isna().any():
        raise ValueError(f"{column!r} must not contain missing timestamps")

    try:
        # Mixed timezone offsets can produce object dtype, which is ambiguous
        # for chronological feature generation.
        _ = parsed.dt.year
    except AttributeError as exc:
        raise ValueError(f"{column!r} must use one consistent datetime timezone") from exc
    return parsed


def _finite_numeric_column(
    transactions: pd.DataFrame,
    column: str,
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    try:
        numeric = pd.to_numeric(transactions[column], errors="raise")
        array = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column!r} must contain finite numeric values") from exc

    if not np.isfinite(array).all():
        raise ValueError(f"{column!r} must contain finite numeric values")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{column!r} must not contain negative values")
    return array


def _target_contributions(
    transactions: pd.DataFrame,
    target_col: str,
) -> np.ndarray:
    """Return known binary labels, with absent/missing labels contributing zero."""

    if target_col not in transactions.columns:
        return np.zeros(len(transactions), dtype=np.int64)

    raw_target = transactions[target_col]
    known = raw_target.notna()
    try:
        numeric = pd.to_numeric(raw_target, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{target_col!r} must contain only binary labels") from exc

    invalid = known & (numeric.isna() | ~numeric.isin((0, 1)))
    if invalid.any():
        raise ValueError(f"{target_col!r} must contain only 0, 1, or missing values")
    return numeric.fillna(0).to_numpy(dtype=np.int64)


def _haversine_km(
    latitude: np.ndarray,
    longitude: np.ndarray,
    merchant_latitude: np.ndarray,
    merchant_longitude: np.ndarray,
) -> np.ndarray:
    if np.any((latitude < -90) | (latitude > 90)):
        raise ValueError("'lat' must be between -90 and 90 degrees")
    if np.any((merchant_latitude < -90) | (merchant_latitude > 90)):
        raise ValueError("'merch_lat' must be between -90 and 90 degrees")
    if np.any((longitude < -180) | (longitude > 180)):
        raise ValueError("'long' must be between -180 and 180 degrees")
    if np.any((merchant_longitude < -180) | (merchant_longitude > 180)):
        raise ValueError("'merch_long' must be between -180 and 180 degrees")

    lat_1 = np.radians(latitude)
    lat_2 = np.radians(merchant_latitude)
    delta_lat = lat_2 - lat_1
    delta_lon = np.radians(merchant_longitude - longitude)
    haversine = (
        np.sin(delta_lat / 2.0) ** 2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(delta_lon / 2.0) ** 2
    )
    haversine = np.clip(haversine, 0.0, 1.0)
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine))


def _validate_engineered_features(features: pd.DataFrame) -> None:
    """Assert invariants that would indicate leakage or corrupt history."""

    numeric = features.loc[:, _NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise AssertionError("engineered numeric features contain missing/infinite values")

    nonnegative_columns = (
        "AGE_AT_TX",
        "DIST_HOME_MERCH_KM",
        "PREV_TX_COUNT",
        "PREV_CUMULATIVE_AMT",
        "PREV_MEAN_AMT",
        "PREV_STD_AMT",
        "CC_PREV_FRAUD",
        "CC_HIST_FRAUD_RATE",
        "TIME_SINCE_LAST_TX",
        "AMT_VS_PREV_MEAN",
    )
    if (features.loc[:, nonnegative_columns].to_numpy(dtype=np.float64) < 0).any():
        raise AssertionError("engineered history must be nonnegative")

    previous_count = features["PREV_TX_COUNT"].to_numpy(dtype=np.int64)
    previous_fraud = features["CC_PREV_FRAUD"].to_numpy(dtype=np.int64)
    if np.any(previous_fraud > previous_count):
        raise AssertionError("previous fraud count cannot exceed transaction count")

    expected_rate = np.divide(
        previous_fraud,
        previous_count,
        out=np.zeros(len(features), dtype=np.float64),
        where=previous_count > 0,
    )
    if not np.allclose(features["CC_HIST_FRAUD_RATE"], expected_rate):
        raise AssertionError("historical fraud rate is internally inconsistent")

    expected_mean = np.divide(
        features["PREV_CUMULATIVE_AMT"].to_numpy(dtype=np.float64),
        previous_count,
        out=np.zeros(len(features), dtype=np.float64),
        where=previous_count > 0,
    )
    if not np.allclose(features["PREV_MEAN_AMT"], expected_mean):
        raise AssertionError("previous amount mean is internally inconsistent")

    expected_first = (previous_count == 0).astype(np.int8)
    if not np.array_equal(features["IS_FIRST_CARD_TX"], expected_first):
        raise AssertionError("cold-start flag is internally inconsistent")

    cold_start = previous_count == 0
    if cold_start.any():
        cold_values = features.loc[cold_start, _HISTORY_ZERO_AT_COLD_START]
        if not np.allclose(cold_values.to_numpy(dtype=np.float64), 0.0):
            raise AssertionError("first-card-event history must be explicitly zero")


def build_features(
    transactions: pd.DataFrame,
    target_col: str = "is_fraud",
) -> pd.DataFrame:
    """Return a copy of ``transactions`` with static and strict-past features.

    Transactions are stable-sorted internally by card and timestamp.  The
    original row order and index are restored before return.  Historical
    features use only strictly earlier timestamp buckets for the same card;
    transactions tied on timestamp never observe each other.

    ``PREV_STD_AMT`` is the sample standard deviation (``ddof=1``), explicitly
    zero until at least two prior transactions exist. ``TIME_SINCE_LAST_TX`` is
    measured in seconds. Missing labels, or an entirely absent target column,
    contribute zero to historical fraud counts.
    """

    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame")
    if not isinstance(target_col, str) or not target_col:
        raise ValueError("target_col must be a non-empty string")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in transactions.columns]
    if missing_columns:
        raise ValueError("transactions is missing required columns: " + ", ".join(missing_columns))

    result = transactions.copy(deep=True)
    transaction_time = _parse_required_datetimes(transactions["trans_timestamp"], "trans_timestamp")
    date_of_birth = _parse_required_datetimes(transactions["dob"], "dob")
    amount = _finite_numeric_column(transactions, "amt", nonnegative=True)
    latitude = _finite_numeric_column(transactions, "lat")
    longitude = _finite_numeric_column(transactions, "long")
    merchant_latitude = _finite_numeric_column(transactions, "merch_lat")
    merchant_longitude = _finite_numeric_column(transactions, "merch_long")

    card_key = _stringify_card_values(transactions["cc_num"]).str.replace(r"[\s-]+", "", regex=True)
    if card_key.isna().any() or card_key.eq("").any():
        raise ValueError("'cc_num' must not contain missing or empty values")

    age = transaction_time.dt.year.to_numpy(dtype=np.int64) - date_of_birth.dt.year.to_numpy(
        dtype=np.int64
    )
    birthday_not_reached = (
        transaction_time.dt.month.to_numpy(dtype=np.int64)
        < date_of_birth.dt.month.to_numpy(dtype=np.int64)
    ) | (
        (
            transaction_time.dt.month.to_numpy(dtype=np.int64)
            == date_of_birth.dt.month.to_numpy(dtype=np.int64)
        )
        & (
            transaction_time.dt.day.to_numpy(dtype=np.int64)
            < date_of_birth.dt.day.to_numpy(dtype=np.int64)
        )
    )
    age = age - birthday_not_reached.astype(np.int64)
    if np.any(age < 0):
        raise ValueError("'dob' must not be later than 'trans_timestamp'")

    result["CC_BIN"] = pd.array(extract_cc_bin(transactions["cc_num"]), dtype="string")
    result["TX_HOUR"] = transaction_time.dt.hour.to_numpy(dtype=np.int64)
    result["TX_DAY_OF_WEEK"] = transaction_time.dt.dayofweek.to_numpy(dtype=np.int64)
    result["TX_MONTH"] = transaction_time.dt.month.to_numpy(dtype=np.int64)
    result["IS_WEEKEND"] = (transaction_time.dt.dayofweek.to_numpy(dtype=np.int64) >= 5).astype(
        np.int8
    )
    result["AGE_AT_TX"] = age
    result["DIST_HOME_MERCH_KM"] = _haversine_km(
        latitude,
        longitude,
        merchant_latitude,
        merchant_longitude,
    )

    if transactions.empty:
        integer_history = {"PREV_TX_COUNT", "CC_PREV_FRAUD", "IS_FIRST_CARD_TX"}
        for column in (
            "PREV_TX_COUNT",
            "PREV_CUMULATIVE_AMT",
            "PREV_MEAN_AMT",
            "PREV_STD_AMT",
            "CC_PREV_FRAUD",
            "CC_HIST_FRAUD_RATE",
            "TIME_SINCE_LAST_TX",
            "IS_FIRST_CARD_TX",
            "AMT_VS_PREV_MEAN",
        ):
            dtype = np.int64 if column in integer_history else np.float64
            result[column] = np.array([], dtype=dtype)
        _validate_engineered_features(result)
        return result

    target = _target_contributions(transactions, target_col)
    work = pd.DataFrame(
        {
            "_fd_position": np.arange(len(transactions), dtype=np.int64),
            "_fd_card": pd.array(card_key, dtype="string"),
            "_fd_timestamp": pd.array(transaction_time),
            "_fd_amount": amount,
            "_fd_target": target,
        }
    )
    work = work.sort_values(
        ["_fd_card", "_fd_timestamp", "_fd_position"],
        kind="mergesort",
    )
    with np.errstate(over="ignore"):
        work["_fd_amount_squared"] = work["_fd_amount"] ** 2
    if not np.isfinite(work["_fd_amount_squared"]).all():
        raise ValueError("'amt' values are too large for historical aggregation")

    work["_fd_bucket_id"] = work.groupby(
        ["_fd_card", "_fd_timestamp"],
        sort=False,
        observed=True,
        dropna=False,
    ).ngroup()

    buckets = (
        work.groupby("_fd_bucket_id", sort=False, observed=True)
        .agg(
            _fd_card=("_fd_card", "first"),
            _fd_timestamp=("_fd_timestamp", "first"),
            _fd_bucket_count=("_fd_amount", "size"),
            _fd_bucket_amount=("_fd_amount", "sum"),
            _fd_bucket_amount_squared=("_fd_amount_squared", "sum"),
            _fd_bucket_fraud=("_fd_target", "sum"),
        )
        .reset_index()
    )
    card_buckets = buckets.groupby("_fd_card", sort=False, observed=True, dropna=False)

    buckets["_fd_previous_count"] = (
        card_buckets["_fd_bucket_count"].cumsum() - buckets["_fd_bucket_count"]
    ).astype(np.int64)
    buckets["_fd_previous_amount"] = np.maximum(
        card_buckets["_fd_bucket_amount"].cumsum() - buckets["_fd_bucket_amount"],
        0.0,
    )
    buckets["_fd_previous_amount_squared"] = np.maximum(
        card_buckets["_fd_bucket_amount_squared"].cumsum() - buckets["_fd_bucket_amount_squared"],
        0.0,
    )
    buckets["_fd_previous_fraud"] = (
        card_buckets["_fd_bucket_fraud"].cumsum() - buckets["_fd_bucket_fraud"]
    ).astype(np.int64)

    previous_count = buckets["_fd_previous_count"].to_numpy(dtype=np.int64)
    previous_amount = buckets["_fd_previous_amount"].to_numpy(dtype=np.float64)
    previous_amount_squared = buckets["_fd_previous_amount_squared"].to_numpy(dtype=np.float64)
    previous_fraud = buckets["_fd_previous_fraud"].to_numpy(dtype=np.int64)

    previous_mean = np.divide(
        previous_amount,
        previous_count,
        out=np.zeros(len(buckets), dtype=np.float64),
        where=previous_count > 0,
    )
    variance_numerator = np.maximum(
        previous_amount_squared
        - np.divide(
            previous_amount**2,
            previous_count,
            out=np.zeros(len(buckets), dtype=np.float64),
            where=previous_count > 0,
        ),
        0.0,
    )
    previous_std = np.sqrt(
        np.divide(
            variance_numerator,
            previous_count - 1,
            out=np.zeros(len(buckets), dtype=np.float64),
            where=previous_count > 1,
        )
    )
    historical_fraud_rate = np.divide(
        previous_fraud,
        previous_count,
        out=np.zeros(len(buckets), dtype=np.float64),
        where=previous_count > 0,
    )

    previous_timestamp = card_buckets["_fd_timestamp"].shift(1)
    time_since_previous = (
        (buckets["_fd_timestamp"] - previous_timestamp).dt.total_seconds().fillna(0.0)
    )

    buckets["_fd_previous_mean"] = previous_mean
    buckets["_fd_previous_std"] = previous_std
    buckets["_fd_historical_fraud_rate"] = historical_fraud_rate
    buckets["_fd_time_since_previous"] = time_since_previous
    buckets["_fd_is_first"] = (previous_count == 0).astype(np.int8)

    bucket_lookup = buckets.set_index("_fd_bucket_id")
    bucket_feature_map = {
        "PREV_TX_COUNT": "_fd_previous_count",
        "PREV_CUMULATIVE_AMT": "_fd_previous_amount",
        "PREV_MEAN_AMT": "_fd_previous_mean",
        "PREV_STD_AMT": "_fd_previous_std",
        "CC_PREV_FRAUD": "_fd_previous_fraud",
        "CC_HIST_FRAUD_RATE": "_fd_historical_fraud_rate",
        "TIME_SINCE_LAST_TX": "_fd_time_since_previous",
        "IS_FIRST_CARD_TX": "_fd_is_first",
    }
    for public_name, bucket_name in bucket_feature_map.items():
        work[public_name] = work["_fd_bucket_id"].map(bucket_lookup[bucket_name])

    previous_mean_per_row = work["PREV_MEAN_AMT"].to_numpy(dtype=np.float64)
    previous_count_per_row = work["PREV_TX_COUNT"].to_numpy(dtype=np.int64)
    work["AMT_VS_PREV_MEAN"] = np.divide(
        work["_fd_amount"].to_numpy(dtype=np.float64),
        previous_mean_per_row,
        out=np.zeros(len(work), dtype=np.float64),
        where=(previous_count_per_row > 0) & (previous_mean_per_row > 0),
    )

    restored = work.sort_values("_fd_position", kind="mergesort")
    integer_history = {"PREV_TX_COUNT", "CC_PREV_FRAUD", "IS_FIRST_CARD_TX"}
    for feature_name in (*bucket_feature_map, "AMT_VS_PREV_MEAN"):
        dtype = np.int64 if feature_name in integer_history else np.float64
        result[feature_name] = restored[feature_name].to_numpy(dtype=dtype)

    _validate_engineered_features(result)
    if not result.index.equals(transactions.index):
        raise AssertionError("feature generation changed the original row index/order")
    return result


__all__ = [
    "ENGINEERED_FEATURE_COLUMNS",
    "REQUIRED_COLUMNS",
    "build_features",
    "extract_cc_bin",
]
