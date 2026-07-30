"""Validation threshold selection and honest fraud-class evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_detection.config import ThresholdConfig

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class ThresholdResult:
    """A locked operating point chosen exclusively from validation data."""

    threshold: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    reviewed_transactions: int
    review_rate: float
    estimated_cost: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _validated_arrays(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or scores.ndim != 1 or len(target) != len(scores):
        raise ValueError("Target and probabilities must be one-dimensional and equal length")
    if len(target) == 0:
        raise ValueError("Evaluation arrays must not be empty")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("Target must contain only 0 and 1")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    return target, scores


def metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float,
    cost: ThresholdConfig,
) -> ThresholdResult:
    """Compute fraud-class outcomes at a fixed, already-selected threshold."""

    target, scores = _validated_arrays(y_true, probabilities)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    predicted = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    reviewed = int(tp + fp)
    return ThresholdResult(
        threshold=float(threshold),
        precision=float(precision_score(target, predicted, zero_division=0)),
        recall=float(recall_score(target, predicted, zero_division=0)),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        reviewed_transactions=reviewed,
        review_rate=reviewed / len(target),
        estimated_cost=float(fn * cost.false_negative_cost + fp * cost.false_positive_cost),
    )


def select_operating_threshold(
    y_validation: pd.Series | np.ndarray,
    validation_probabilities: pd.Series | np.ndarray,
    cost: ThresholdConfig,
) -> ThresholdResult:
    """Choose the exact minimum-cost validation threshold under review capacity.

    Candidate cutoffs occur only between distinct score values, so tied scores
    are never split arbitrarily. Test labels are not accepted by this function.
    """

    target, scores = _validated_arrays(y_validation, validation_probabilities)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_target = target[order]
    cumulative_tp = np.cumsum(sorted_target, dtype=np.int64)
    cumulative_fp = np.cumsum(1 - sorted_target, dtype=np.int64)

    # Evaluate after the final member of each score tie.
    tie_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    reviewed = tie_ends + 1
    capacity = math.floor(cost.max_review_rate * len(target))
    allowed = reviewed <= capacity

    total_positive = int(target.sum())
    rows: list[ThresholdResult] = []
    no_review_threshold = float(np.nextafter(sorted_scores[0], np.inf))
    rows.append(metrics_at_threshold(target, scores, no_review_threshold, cost))
    for position in tie_ends[allowed]:
        tp = int(cumulative_tp[position])
        fp = int(cumulative_fp[position])
        fn = total_positive - tp
        tn = len(target) - tp - fp - fn
        reviewed_count = int(position + 1)
        rows.append(
            ThresholdResult(
                threshold=float(sorted_scores[position]),
                precision=tp / reviewed_count if reviewed_count else 0.0,
                recall=tp / total_positive if total_positive else 0.0,
                true_positives=tp,
                false_positives=fp,
                true_negatives=tn,
                false_negatives=fn,
                reviewed_transactions=reviewed_count,
                review_rate=reviewed_count / len(target),
                estimated_cost=float(fn * cost.false_negative_cost + fp * cost.false_positive_cost),
            )
        )

    return min(
        rows,
        key=lambda row: (
            row.estimated_cost,
            -row.recall,
            row.false_positives,
            -row.threshold,
        ),
    )


def ranking_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
) -> dict[str, float]:
    target, scores = _validated_arrays(y_true, probabilities)
    metrics = {
        "pr_auc": float(average_precision_score(target, scores)),
        "fraud_prevalence": float(target.mean()),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(target, scores)) if np.unique(target).size == 2 else float("nan")
    )
    return metrics


def full_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float,
    cost: ThresholdConfig,
) -> dict[str, Any]:
    """Combine ranking, operating-point, and simple binomial uncertainty metrics."""

    result = metrics_at_threshold(y_true, probabilities, threshold, cost)
    metrics: dict[str, Any] = ranking_metrics(y_true, probabilities)
    metrics.update(result.to_dict())
    metrics["precision_wilson_95"] = wilson_interval(
        result.true_positives, result.true_positives + result.false_positives
    )
    metrics["recall_wilson_95"] = wilson_interval(
        result.true_positives, result.true_positives + result.false_negatives
    )
    metrics["fraud_cases_detected"] = result.true_positives
    metrics["fraud_cases_missed"] = result.false_negatives
    metrics["legitimate_transactions_flagged"] = result.false_positives
    return metrics


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    """Return a Wilson score interval for a binomial proportion."""

    if trials <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = (
        z * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def threshold_comparison_table(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    cost: ThresholdConfig,
    selected_threshold: float,
) -> pd.DataFrame:
    """Compare fixed score cutoffs and review-capacity operating points."""

    target, scores = _validated_arrays(y_true, probabilities)
    candidates = {0.5, float(selected_threshold)}
    for review_rate in (0.001, 0.0025, 0.005, 0.01, 0.02, cost.max_review_rate):
        if review_rate <= 0 or review_rate > 1:
            continue
        rank = max(1, min(len(scores), int(math.ceil(review_rate * len(scores)))))
        candidates.add(float(np.partition(scores, len(scores) - rank)[len(scores) - rank]))

    records = [
        metrics_at_threshold(target, scores, threshold, cost).to_dict()
        for threshold in sorted(candidates, reverse=True)
    ]
    table = pd.DataFrame.from_records(records)
    table["selected"] = np.isclose(table["threshold"], selected_threshold)
    return table.sort_values(["review_rate", "threshold"], kind="mergesort").reset_index(drop=True)


def slice_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    groups: pd.Series,
    threshold: float,
    cost: ThresholdConfig,
    *,
    minimum_rows: int = 100,
) -> pd.DataFrame:
    """Evaluate time or transaction segments without making causal claims."""

    if len(y_true) != len(probabilities) or len(y_true) != len(groups):
        raise ValueError("Target, probabilities, and groups must have equal length")
    records: list[dict[str, Any]] = []
    working = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "probability": np.asarray(probabilities),
            "group": groups.astype("string").fillna("<missing>").to_numpy(),
        },
        index=y_true.index,
    )
    for group_name, group in working.groupby("group", sort=True):
        if len(group) < minimum_rows:
            continue
        group_metrics = full_metrics(group["target"], group["probability"], threshold, cost)
        records.append(
            {
                "segment": str(group_name),
                "transactions": len(group),
                **{
                    key: group_metrics[key]
                    for key in (
                        "fraud_prevalence",
                        "pr_auc",
                        "precision",
                        "recall",
                        "false_positives",
                        "false_negatives",
                        "review_rate",
                        "estimated_cost",
                    )
                },
            }
        )
    return pd.DataFrame.from_records(records)


def save_evaluation_plots(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float,
    output_dir: str | Path,
) -> None:
    """Save the primary precision-recall curve and absolute confusion matrix."""

    target, scores = _validated_arrays(y_true, probabilities)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(target, scores)
    average_precision = average_precision_score(target, scores)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(recall, precision, label=f"PR-AUC (AP) = {average_precision:.4f}")
    axis.axhline(target.mean(), color="gray", linestyle="--", label="Prevalence baseline")
    axis.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve (test)")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path / "precision_recall_curve.png", dpi=160)
    plt.close(figure)

    predicted = (scores >= threshold).astype(np.int8)
    matrix = confusion_matrix(target, predicted, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        xlabel="Predicted",
        ylabel="Actual",
        title=f"Confusion matrix at locked threshold {threshold:.6g}",
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path / "confusion_matrix.png", dpi=160)
    plt.close(figure)


__all__ = [
    "ThresholdResult",
    "full_metrics",
    "metrics_at_threshold",
    "ranking_metrics",
    "save_evaluation_plots",
    "select_operating_threshold",
    "slice_metrics",
    "threshold_comparison_table",
    "wilson_interval",
]
