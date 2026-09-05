"""Leakage-safe, deterministic transaction feature engineering.

All behavioral features in this module are based on transactions at strictly
earlier timestamps for the same card.  Rows sharing a timestamp are processed
as one bucket, so their amounts and labels cannot leak into one another.
"""

from __future__ import annotations

from collections import deque
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

VELOCITY_WINDOWS: Final[tuple[tuple[str, pd.Timedelta], ...]] = (
    ("1H", pd.Timedelta(hours=1)),
    ("24H", pd.Timedelta(hours=24)),
    ("7D", pd.Timedelta(days=7)),
)
VELOCITY_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    feature
    for suffix, _ in VELOCITY_WINDOWS
    for feature in (
        f"TX_COUNT_{suffix}",
        f"AMT_MAX_{suffix}",
        f"AMT_MEAN_{suffix}",
    )
)
ENGINEERED_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
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
    "TIME_SINCE_LAST_TX",
    "IS_FIRST_CARD_TX",
    "AMT_VS_PREV_MEAN",
    *VELOCITY_FEATURE_COLUMNS,
)

_HISTORY_ZERO_AT_COLD_START: Final[tuple[str, ...]] = (
    "PREV_TX_COUNT",
    "PREV_CUMULATIVE_AMT",
    "PREV_MEAN_AMT",
    "PREV_STD_AMT",
    "TIME_SINCE_LAST_TX",
    "AMT_VS_PREV_MEAN",
    *VELOCITY_FEATURE_COLUMNS,
)
_EARTH_RADIUS_KM: Final[float] = 6_371.0088


def _add_strict_past_velocity_features(buckets: pd.DataFrame) -> None:
    """Add per-card window aggregates over ``[timestamp - window, timestamp)``."""

    bucket_count = buckets["_fd_bucket_count"].to_numpy(dtype=np.int64)
    bucket_amount = buckets["_fd_bucket_amount"].to_numpy(dtype=np.float64)
    bucket_max = buckets["_fd_bucket_max"].to_numpy(dtype=np.float64)

    for suffix, window in VELOCITY_WINDOWS:
        counts = np.zeros(len(buckets), dtype=np.int64)
        means = np.zeros(len(buckets), dtype=np.float64)
        maxima = np.zeros(len(buckets), dtype=np.float64)
        window_ns = int(window.value)

        for positions in buckets.groupby(
            "_fd_card",
            sort=False,
            observed=True,
            dropna=False,
        ).indices.values():
            positions = np.asarray(positions, dtype=np.int64)
            timestamps = (
                buckets.iloc[positions]["_fd_timestamp"].astype("int64").to_numpy(dtype=np.int64)
            )
            local_count = bucket_count[positions]
            local_amount = bucket_amount[positions]
            left = np.searchsorted(timestamps, timestamps - window_ns, side="left")
            right = np.arange(len(positions), dtype=np.int64)

            count_prefix = np.concatenate(([0], np.cumsum(local_count, dtype=np.int64)))
            amount_prefix = np.concatenate(([0.0], np.cumsum(local_amount, dtype=np.float64)))
            window_counts = count_prefix[right] - count_prefix[left]
            window_amounts = amount_prefix[right] - amount_prefix[left]
            counts[positions] = window_counts
            means[positions] = np.divide(
                window_amounts,
                window_counts,
                out=np.zeros(len(positions), dtype=np.float64),
                where=window_counts > 0,
            )

            candidates: deque[int] = deque()
            for local_position, timestamp in enumerate(timestamps):
                lower_bound = timestamp - window_ns
                while candidates and timestamps[candidates[0]] < lower_bound:
                    candidates.popleft()
                if candidates:
                    maxima[positions[local_position]] = bucket_max[positions[candidates[0]]]
                while (
                    candidates
                    and bucket_max[positions[candidates[-1]]]
                    <= bucket_max[positions[local_position]]
                ):
                    candidates.pop()
                candidates.append(local_position)

        buckets[f"TX_COUNT_{suffix}"] = counts
        buckets[f"AMT_MAX_{suffix}"] = maxima
        buckets[f"AMT_MEAN_{suffix}"] = means


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

    numeric = features.loc[:, ENGINEERED_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise AssertionError("engineered numeric features contain missing/infinite values")

    nonnegative_columns = (
        "AGE_AT_TX",
        "DIST_HOME_MERCH_KM",
        "PREV_TX_COUNT",
        "PREV_CUMULATIVE_AMT",
        "PREV_MEAN_AMT",
        "PREV_STD_AMT",
        "TIME_SINCE_LAST_TX",
        "AMT_VS_PREV_MEAN",
    )
    if (features.loc[:, nonnegative_columns].to_numpy(dtype=np.float64) < 0).any():
        raise AssertionError("engineered history must be nonnegative")

    previous_count = features["PREV_TX_COUNT"].to_numpy(dtype=np.int64)
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


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``transactions`` with static and strict-past features.

    Transactions are stable-sorted internally by card and timestamp.  The
    original row order and index are restored before return.  Historical
    features use only strictly earlier timestamp buckets for the same card;
    transactions tied on timestamp never observe each other.

    ``PREV_STD_AMT`` is the sample standard deviation (``ddof=1``), explicitly
    zero until at least two prior transactions exist. ``TIME_SINCE_LAST_TX`` is
    measured in seconds. Labels are never used to generate features because
    their availability at scoring time is unknown.
    """

    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame")

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
        integer_history = {
            "PREV_TX_COUNT",
            "IS_FIRST_CARD_TX",
            *(feature for feature in VELOCITY_FEATURE_COLUMNS if feature.startswith("TX_COUNT_")),
        }
        for column in (
            "PREV_TX_COUNT",
            "PREV_CUMULATIVE_AMT",
            "PREV_MEAN_AMT",
            "PREV_STD_AMT",
            "TIME_SINCE_LAST_TX",
            "IS_FIRST_CARD_TX",
            "AMT_VS_PREV_MEAN",
            *VELOCITY_FEATURE_COLUMNS,
        ):
            dtype = np.int64 if column in integer_history else np.float64
            result[column] = np.array([], dtype=dtype)
        _validate_engineered_features(result)
        return result

    work = pd.DataFrame(
        {
            "_fd_position": np.arange(len(transactions), dtype=np.int64),
            "_fd_card": pd.array(card_key, dtype="string"),
            "_fd_timestamp": pd.array(transaction_time),
            "_fd_amount": amount,
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
            _fd_bucket_max=("_fd_amount", "max"),
            _fd_bucket_amount_squared=("_fd_amount_squared", "sum"),
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
    previous_count = buckets["_fd_previous_count"].to_numpy(dtype=np.int64)
    previous_amount = buckets["_fd_previous_amount"].to_numpy(dtype=np.float64)
    previous_amount_squared = buckets["_fd_previous_amount_squared"].to_numpy(dtype=np.float64)

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
    previous_timestamp = card_buckets["_fd_timestamp"].shift(1)
    time_since_previous = (
        (buckets["_fd_timestamp"] - previous_timestamp).dt.total_seconds().fillna(0.0)
    )

    buckets["_fd_previous_mean"] = previous_mean
    buckets["_fd_previous_std"] = previous_std
    buckets["_fd_time_since_previous"] = time_since_previous
    buckets["_fd_is_first"] = (previous_count == 0).astype(np.int8)
    _add_strict_past_velocity_features(buckets)

    bucket_lookup = buckets.set_index("_fd_bucket_id")
    bucket_feature_map = {
        "PREV_TX_COUNT": "_fd_previous_count",
        "PREV_CUMULATIVE_AMT": "_fd_previous_amount",
        "PREV_MEAN_AMT": "_fd_previous_mean",
        "PREV_STD_AMT": "_fd_previous_std",
        "TIME_SINCE_LAST_TX": "_fd_time_since_previous",
        "IS_FIRST_CARD_TX": "_fd_is_first",
        **{feature: feature for feature in VELOCITY_FEATURE_COLUMNS},
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
    integer_history = {
        "PREV_TX_COUNT",
        "IS_FIRST_CARD_TX",
        *(feature for feature in VELOCITY_FEATURE_COLUMNS if feature.startswith("TX_COUNT_")),
    }
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
    "VELOCITY_FEATURE_COLUMNS",
    "build_features",
]
