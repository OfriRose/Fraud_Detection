"""Versioned fraud-model loading and leakage-safe scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fraud_detection.features import REQUIRED_COLUMNS, build_features


class InferenceError(ValueError):
    """Raised when an artifact or scoring request violates its contract."""


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate a trusted local, versioned pipeline artifact."""

    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    required = {
        "artifact_type",
        "model_version",
        "pipeline",
        "threshold",
        "feature_schema",
        "metadata",
    }
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise InferenceError("Artifact is not a supported fraud-detection pipeline bundle")
    if artifact["artifact_type"] != "fraud_detection_pipeline":
        raise InferenceError("Unexpected artifact_type")
    if not hasattr(artifact["pipeline"], "predict_proba"):
        raise InferenceError("Artifact pipeline does not support probability prediction")
    return artifact


def _validate_scoring_history(
    current: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    if history.empty:
        return
    for column in ("cc_num", "trans_timestamp"):
        if column not in history:
            raise InferenceError(f"History is missing required column {column!r}")
    current_times = pd.to_datetime(current["trans_timestamp"], errors="coerce")
    history_times = pd.to_datetime(history["trans_timestamp"], errors="coerce")
    if current_times.isna().any() or history_times.isna().any():
        raise InferenceError("Current and historical timestamps must be valid")

    current_minimum = (
        pd.DataFrame({"cc_num": current["cc_num"].astype("string"), "time": current_times})
        .groupby("cc_num", observed=True)["time"]
        .min()
    )
    history_maximum = (
        pd.DataFrame({"cc_num": history["cc_num"].astype("string"), "time": history_times})
        .groupby("cc_num", observed=True)["time"]
        .max()
    )
    shared_cards = current_minimum.index.intersection(history_maximum.index)
    invalid = history_maximum.loc[shared_cards] >= current_minimum.loc[shared_cards]
    if invalid.any():
        raise InferenceError(
            "Every history row for a card must precede all current rows for that card"
        )


def score_transactions(
    transactions: pd.DataFrame,
    artifact: dict[str, Any] | str | Path,
    *,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score one or more current transactions.

    `history` is optional and, when provided, must contain only rows known
    before every current row for the same card. Labels are never used to build
    features, so a batch cannot leak its labels into later scores.
    With empty history, the explicit cold-start values are used.
    """

    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        raise InferenceError("transactions must be a non-empty pandas DataFrame")
    bundle = load_artifact(artifact) if isinstance(artifact, str | Path) else artifact
    if not isinstance(bundle, dict):
        raise InferenceError("artifact must be a loaded bundle or path")

    missing = set(REQUIRED_COLUMNS).difference(transactions.columns)
    if missing:
        raise InferenceError(
            "Current transactions are missing required columns: " + ", ".join(sorted(missing))
        )
    current = transactions.copy()
    current["_fd_current_position"] = range(len(current))
    current["_fd_is_current"] = True

    historical = pd.DataFrame() if history is None else history.copy()
    _validate_scoring_history(current, historical)
    if not historical.empty:
        historical["_fd_is_current"] = False
        historical["_fd_current_position"] = -1
        combined = pd.concat([historical, current], ignore_index=True, sort=False)
    else:
        combined = current.reset_index(drop=True)

    featured = build_features(combined)
    current_features = (
        featured.loc[featured["_fd_is_current"]]
        .sort_values("_fd_current_position", kind="mergesort")
        .copy()
    )
    feature_names = list(bundle["feature_schema"]["model_features"])
    missing_features = set(feature_names).difference(current_features.columns)
    if missing_features:
        raise InferenceError(
            "Artifact feature schema cannot be satisfied: " + ", ".join(sorted(missing_features))
        )

    probabilities = bundle["pipeline"].predict_proba(current_features[feature_names])[:, 1]
    threshold = float(bundle["threshold"])
    result = pd.DataFrame(
        {
            "fraud_probability": probabilities,
            "fraud_decision": (probabilities >= threshold).astype("int8"),
            "model_version": bundle["model_version"],
            "threshold": threshold,
            "model_name": bundle["metadata"]["model_name"],
            "training_cutoff": bundle["metadata"]["training_cutoff"],
        },
        index=transactions.index,
    )
    return result


__all__ = ["InferenceError", "load_artifact", "score_transactions"]
