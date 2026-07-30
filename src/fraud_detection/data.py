"""Data loading and fixed-scope preparation.

The project supports either the original CSV or the surviving prepared pickle.
The pickle is trusted local data only: pickle must never be loaded from an
untrusted source.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "is_fraud"
TIMESTAMP_COLUMN = "trans_timestamp"
ENTITY_COLUMN = "cc_num"

REQUIRED_MODEL_SOURCE_COLUMNS = {
    ENTITY_COLUMN,
    "trans_num",
    TIMESTAMP_COLUMN,
    "amt",
    TARGET_COLUMN,
    "category",
    "profile",
    "city_pop",
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "is_male",
}

STRING_COLUMNS = (
    "ssn",
    ENTITY_COLUMN,
    "acct_num",
    "trans_num",
    "city",
    "job",
    "profile",
    "category",
    "merchant",
)

MODEL_EXCLUDED_PRECONSOLIDATED_COLUMNS = ("city", "job", "merchant")


class DataValidationError(ValueError):
    """Raised when the transaction data cannot satisfy the training contract."""


def _read_csv_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    dtypes = {
        "ssn": "string",
        ENTITY_COLUMN: "string",
        "acct_num": "string",
        "trans_num": "string",
    }
    yield from pd.read_csv(path, chunksize=chunksize, dtype=dtypes, low_memory=False)


def _prepare_raw_chunk(chunk: pd.DataFrame, state: str, year: int) -> pd.DataFrame:
    if "trans_date" not in chunk or "state" not in chunk:
        raise DataValidationError("Raw CSV must contain trans_date and state columns")
    trans_date = pd.to_datetime(chunk["trans_date"], errors="coerce")
    selected = chunk.loc[(trans_date.dt.year == year) & (chunk["state"] == state)].copy()
    selected["trans_date"] = trans_date.loc[selected.index]
    return selected


def load_transactions(
    path: str | Path,
    *,
    state: str = "CA",
    year: int = 2020,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Load transactions and apply the predeclared state/year scope.

    CSV scoping is fixed by configuration and never selected from target
    outcomes. Pickle input is the repository's already-scoped legacy data; its
    globally consolidated city/job/merchant fields are retained for audit
    provenance but excluded from every model feature set.
    """

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Transaction input not found: {input_path}")

    if input_path.suffix.lower() in {".pkl", ".pickle"}:
        frame = pd.read_pickle(input_path)  # noqa: S301 - explicitly trusted local artifact
    elif input_path.suffix.lower() == ".csv":
        selected_chunks = [
            selected
            for chunk in _read_csv_chunks(input_path, chunksize)
            if not (selected := _prepare_raw_chunk(chunk, state, year)).empty
        ]
        if not selected_chunks:
            raise DataValidationError(f"No rows matched fixed scope state={state!r}, year={year}")
        frame = pd.concat(selected_chunks, ignore_index=True)
    else:
        raise DataValidationError("Input must be a .csv, .pkl, or .pickle file")

    return prepare_transactions(frame, state=state, year=year)


def _build_timestamp(frame: pd.DataFrame) -> pd.Series:
    if TIMESTAMP_COLUMN in frame:
        return pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
    required = {"trans_date", "trans_time"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(
            "Cannot construct trans_timestamp; missing " + ", ".join(sorted(missing))
        )
    date_text = pd.to_datetime(frame["trans_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return pd.to_datetime(date_text + " " + frame["trans_time"].astype("string"), errors="coerce")


def prepare_transactions(
    frame: pd.DataFrame,
    *,
    state: str = "CA",
    year: int = 2020,
) -> pd.DataFrame:
    """Normalize schema without learning any distributional parameters."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError("Transaction input must be a non-empty pandas DataFrame")

    prepared = frame.copy()
    prepared[TIMESTAMP_COLUMN] = _build_timestamp(prepared)
    invalid_timestamps = int(prepared[TIMESTAMP_COLUMN].isna().sum())
    if invalid_timestamps:
        raise DataValidationError(f"Found {invalid_timestamps} missing or invalid timestamps")

    if "state" in prepared:
        wrong_state = prepared["state"].astype("string").ne(state).sum()
        if wrong_state:
            raise DataValidationError(f"Found {wrong_state} rows outside configured state {state}")
    wrong_year = prepared[TIMESTAMP_COLUMN].dt.year.ne(year).sum()
    if wrong_year:
        raise DataValidationError(f"Found {wrong_year} rows outside configured year {year}")

    if "gender" in prepared and "is_male" not in prepared:
        prepared["is_male"] = prepared["gender"].astype("string").str.upper().eq("M").astype("int8")

    for column in STRING_COLUMNS:
        if column in prepared:
            prepared[column] = prepared[column].astype("string")
    for column in ("city", "job", "profile", "category", "merchant"):
        if column in prepared:
            prepared[column] = prepared[column].str.strip().str.lower()

    if "dob" in prepared:
        prepared["dob"] = pd.to_datetime(prepared["dob"], errors="coerce")

    for column in ("amt", "city_pop", "lat", "long", "merch_lat", "merch_long", "is_male"):
        if column in prepared:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    missing_required = REQUIRED_MODEL_SOURCE_COLUMNS.difference(prepared.columns)
    if missing_required:
        raise DataValidationError(
            "Missing required transaction columns: " + ", ".join(sorted(missing_required))
        )

    target = pd.to_numeric(prepared[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise DataValidationError("is_fraud must contain only non-missing 0/1 values")
    prepared[TARGET_COLUMN] = target.astype("int8")

    if prepared[ENTITY_COLUMN].isna().any():
        raise DataValidationError("cc_num must not be missing")
    if prepared["trans_num"].isna().any() or prepared["trans_num"].duplicated().any():
        raise DataValidationError("trans_num must be non-missing and unique")

    # A fresh unique index makes the split-overlap assertions unambiguous.
    prepared = prepared.reset_index(drop=True)
    prepared.index = pd.RangeIndex(len(prepared), name="row_id")
    if not prepared.index.is_unique:
        raise AssertionError("Prepared transaction index must be unique")
    if not np.isfinite(prepared["amt"].dropna()).all():
        raise DataValidationError("amt contains infinite values")
    return prepared


__all__ = [
    "DataValidationError",
    "ENTITY_COLUMN",
    "MODEL_EXCLUDED_PRECONSOLIDATED_COLUMNS",
    "REQUIRED_MODEL_SOURCE_COLUMNS",
    "TARGET_COLUMN",
    "TIMESTAMP_COLUMN",
    "load_transactions",
    "prepare_transactions",
]
