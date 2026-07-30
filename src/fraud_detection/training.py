"""Single-command chronological training, selection, evaluation, and packaging."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.pipeline import Pipeline

from fraud_detection.config import DEFAULT_CONFIG_PATH, TrainingConfig, load_config
from fraud_detection.data import (
    MODEL_EXCLUDED_PRECONSOLIDATED_COLUMNS,
    load_transactions,
    prepare_transactions,
)
from fraud_detection.evaluation import (
    full_metrics,
    ranking_metrics,
    save_evaluation_plots,
    select_operating_threshold,
    slice_metrics,
    threshold_comparison_table,
)
from fraud_detection.features import build_features
from fraud_detection.modeling import (
    FeatureSchema,
    build_candidate_pipelines,
    get_feature_schema,
    public_model_parameters,
    select_model_matrix,
)
from fraud_detection.split import (
    TemporalSplitConfig,
    chronological_split,
    summarize_splits,
)

matplotlib.use("Agg")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path | datetime | date | pd.Timestamp):
        return str(value)
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _predict_probabilities(pipeline: Pipeline, matrix: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(pipeline.predict_proba(matrix)[:, 1], dtype=float)
    if probabilities.shape != (len(matrix),):
        raise AssertionError("Probability output shape is invalid")
    if not np.isfinite(probabilities).all():
        raise AssertionError("Model produced non-finite probabilities")
    return probabilities


def _model_comparison(
    candidates: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    config: TrainingConfig,
    schema: FeatureSchema,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray], dict[str, float]]:
    records: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    probabilities_by_model: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    for name, pipeline in candidates.items():
        started = time.monotonic()
        pipeline.fit(X_train, y_train)
        elapsed = time.monotonic() - started
        validation_probabilities = _predict_probabilities(pipeline, X_validation)
        ranking = ranking_metrics(y_validation, validation_probabilities)
        operating_point = select_operating_threshold(
            y_validation,
            validation_probabilities,
            config.threshold,
        )
        records.append(
            {
                "model_name": name,
                "parameters": json.dumps(public_model_parameters(pipeline), sort_keys=True),
                "feature_set": json.dumps(list(schema.model_features)),
                "training_period_start": str(X_train.attrs.get("date_start", "")),
                "training_period_end": str(X_train.attrs.get("date_end", "")),
                "validation_pr_auc": ranking["pr_auc"],
                "validation_roc_auc": ranking["roc_auc"],
                "validation_precision": operating_point.precision,
                "validation_recall": operating_point.recall,
                "validation_false_positives": operating_point.false_positives,
                "validation_false_negatives": operating_point.false_negatives,
                "validation_review_rate": operating_point.review_rate,
                "validation_estimated_cost": operating_point.estimated_cost,
                "selected_threshold": operating_point.threshold,
                "fit_seconds": elapsed,
            }
        )
        fitted[name] = pipeline
        probabilities_by_model[name] = validation_probabilities
        thresholds[name] = operating_point.threshold

    comparison = pd.DataFrame.from_records(records).sort_values(
        ["validation_pr_auc", "validation_estimated_cost"],
        ascending=[False, True],
        kind="mergesort",
    )
    return comparison.reset_index(drop=True), fitted, probabilities_by_model, thresholds


def _numeric_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    schema: FeatureSchema,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for feature in schema.numerical:
        train_values = pd.to_numeric(train[feature], errors="coerce")
        test_values = pd.to_numeric(test[feature], errors="coerce")
        train_std = float(train_values.std(ddof=0))
        difference = float(test_values.mean() - train_values.mean())
        records.append(
            {
                "feature": feature,
                "type": "numeric",
                "train_mean": float(train_values.mean()),
                "test_mean": float(test_values.mean()),
                "standardized_mean_difference": difference / train_std if train_std > 0 else 0.0,
                "train_missing_rate": float(train_values.isna().mean()),
                "test_missing_rate": float(test_values.isna().mean()),
            }
        )
    for feature in schema.categorical:
        train_distribution = (
            train[feature].astype("string").fillna("<missing>").value_counts(normalize=True)
        )
        test_distribution = (
            test[feature].astype("string").fillna("<missing>").value_counts(normalize=True)
        )
        categories = train_distribution.index.union(test_distribution.index)
        total_variation = (
            0.5
            * (
                train_distribution.reindex(categories, fill_value=0)
                - test_distribution.reindex(categories, fill_value=0)
            )
            .abs()
            .sum()
        )
        records.append(
            {
                "feature": feature,
                "type": "categorical",
                "total_variation_distance": float(total_variation),
                "unseen_test_category_count": int(
                    len(test_distribution.index.difference(train_distribution.index))
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    estimator = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_, dtype=float)).reshape(-1)
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    if len(names) != len(importance):
        raise AssertionError("Feature-importance length does not match transformed schema")
    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )


def _render_evaluation_report(
    *,
    split_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    champion_name: str,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    config: TrainingConfig,
    target_history_enabled: bool,
) -> str:
    cost_formula = (
        f"false negatives × ${config.threshold.false_negative_cost:,.2f} + "
        f"false positives × ${config.threshold.false_positive_cost:,.2f}"
    )

    def metric_line(label: str, payload: dict[str, Any], key: str, decimals: int = 6) -> str:
        return f"| {label} | {payload[key]:.{decimals}f} |"

    split_table = split_summary.to_markdown(index=False, floatfmt=".6f")
    comparison_columns = [
        "model_name",
        "validation_pr_auc",
        "validation_precision",
        "validation_recall",
        "validation_review_rate",
        "validation_estimated_cost",
        "selected_threshold",
    ]
    comparison_table = comparison[comparison_columns].to_markdown(index=False, floatfmt=".6f")
    validation_table = "\n".join(
        [
            "| Metric | Validation |",
            "|---|---:|",
            metric_line("PR-AUC", validation_metrics, "pr_auc"),
            metric_line("ROC-AUC", validation_metrics, "roc_auc"),
            metric_line("Precision", validation_metrics, "precision"),
            metric_line("Recall", validation_metrics, "recall"),
            f"| False positives | {validation_metrics['false_positives']:,} |",
            f"| False negatives | {validation_metrics['false_negatives']:,} |",
            metric_line("Review rate", validation_metrics, "review_rate"),
            f"| Estimated cost | ${validation_metrics['estimated_cost']:,.2f} |",
        ]
    )
    test_table = "\n".join(
        [
            "| Metric | Test |",
            "|---|---:|",
            metric_line("PR-AUC", test_metrics, "pr_auc"),
            metric_line("ROC-AUC", test_metrics, "roc_auc"),
            metric_line("Precision", test_metrics, "precision"),
            metric_line("Recall", test_metrics, "recall"),
            f"| True positives / fraud detected | {test_metrics['true_positives']:,} |",
            f"| False negatives / fraud missed | {test_metrics['false_negatives']:,} |",
            f"| False positives / legitimate flagged | {test_metrics['false_positives']:,} |",
            f"| True negatives | {test_metrics['true_negatives']:,} |",
            metric_line("Review rate", test_metrics, "review_rate"),
            f"| Estimated cost | ${test_metrics['estimated_cost']:,.2f} |",
        ]
    )
    return f"""# Corrected chronological evaluation

This report is generated by the leakage-safe source pipeline. The final test
period was not used for preprocessing, model selection, or threshold selection.

## Temporal split

{split_table}

## Validation model comparison

{comparison_table}

Champion selected by validation PR-AUC: `{champion_name}`.

## Locked validation operating point

{validation_table}

The threshold was selected on validation data by minimizing
`{cost_formula}`
while reviewing at most {config.threshold.max_review_rate:.1%} of transactions.

## Final test evaluation

{test_table}

Precision 95% Wilson interval: {test_metrics["precision_wilson_95"]}.
Recall 95% Wilson interval: {test_metrics["recall_wilson_95"]}.

Target-derived history enabled in the model: `{target_history_enabled}`. The
corrected `CC_PREV_FRAUD` and fraud-rate features are generated and regression
tested, but are disabled by default because no label-availability timestamp is
present in the source data.

Feature importance is associative, not causal. Cost values are scenario
estimates under fixed assumptions, not observed business savings.
"""


def run_training(
    config: TrainingConfig,
    *,
    input_path: str | Path | None = None,
    transactions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the finalized methodology and evaluate test exactly once."""

    if (
        config.data.target_column != "is_fraud"
        or config.data.timestamp_column != "trans_timestamp"
        or config.data.entity_column != "cc_num"
    ):
        raise ValueError("This release requires canonical is_fraud/trans_timestamp/cc_num names")

    source_path = Path(input_path) if input_path is not None else config.data.input_path
    if transactions is None:
        raw = load_transactions(source_path, state=config.data.state, year=config.data.year)
    else:
        raw = prepare_transactions(
            transactions,
            state=config.data.state,
            year=config.data.year,
        )
    featured = build_features(raw, target_col=config.data.target_column)
    if featured.loc[:, featured.columns != config.data.target_column].isna().all(axis=1).any():
        raise AssertionError("Feature generation created an unusable all-missing row")

    split_config = TemporalSplitConfig(
        train_end=config.data.train_end,
        validation_end=config.data.validation_end,
        timestamp_col=config.data.timestamp_column,
    )
    splits = chronological_split(featured, split_config)
    split_summary = summarize_splits(splits, target_col=config.data.target_column)

    schema = get_feature_schema(config.data.include_target_history)
    X_train = select_model_matrix(splits.train, schema)
    X_validation = select_model_matrix(splits.validation, schema)
    X_test = select_model_matrix(splits.test, schema)
    for matrix, frame in (
        (X_train, splits.train),
        (X_validation, splits.validation),
        (X_test, splits.test),
    ):
        matrix.attrs["date_start"] = frame[config.data.timestamp_column].min()
        matrix.attrs["date_end"] = frame[config.data.timestamp_column].max()
        if config.data.target_column in matrix:
            raise AssertionError("Target leaked into a model matrix")

    y_train = splits.train[config.data.target_column]
    y_validation = splits.validation[config.data.target_column]
    y_test = splits.test[config.data.target_column]

    candidates = build_candidate_pipelines(config, y_train, schema)
    comparison, fitted, validation_probabilities, thresholds = _model_comparison(
        candidates,
        X_train,
        y_train,
        X_validation,
        y_validation,
        config,
        schema,
    )
    champion_name = str(comparison.iloc[0]["model_name"])
    champion = fitted[champion_name]
    locked_threshold = thresholds[champion_name]
    champion_validation_probabilities = validation_probabilities[champion_name]
    validation_metrics = full_metrics(
        y_validation,
        champion_validation_probabilities,
        locked_threshold,
        config.threshold,
    )

    # Methodology, model, and threshold are now locked. This is the only point
    # at which the test matrix is scored.
    test_probabilities = _predict_probabilities(champion, X_test)
    test_metrics = full_metrics(
        y_test,
        test_probabilities,
        locked_threshold,
        config.threshold,
    )

    artifact_dir = config.paths.artifact_dir
    report_dir = config.paths.report_dir
    figure_dir = report_dir / "figures"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    split_summary.to_csv(report_dir / "split_summary.csv", index=False)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    threshold_comparison_table(
        y_validation,
        champion_validation_probabilities,
        config.threshold,
        locked_threshold,
    ).to_csv(report_dir / "validation_thresholds.csv", index=False)
    _numeric_drift(splits.train, splits.test, schema).to_csv(
        report_dir / "train_test_drift.csv", index=False
    )
    test_month = splits.test[config.data.timestamp_column].dt.to_period("M").astype("string")
    slice_metrics(
        y_test,
        test_probabilities,
        test_month,
        locked_threshold,
        config.threshold,
    ).to_csv(report_dir / "test_monthly_metrics.csv", index=False)
    slice_metrics(
        y_test,
        test_probabilities,
        splits.test["category"],
        locked_threshold,
        config.threshold,
    ).to_csv(report_dir / "test_category_metrics.csv", index=False)
    _feature_importance(champion).to_csv(report_dir / "feature_importance.csv", index=False)
    save_evaluation_plots(y_test, test_probabilities, locked_threshold, figure_dir)

    source_metadata: dict[str, Any] = {
        "path": str(source_path),
        "format": source_path.suffix.lower(),
        "rows": len(raw),
        "upstream": {
            "name": config.data.source_name,
            "url": config.data.source_url,
            "license": config.data.source_license,
        },
        "scope": {"state": config.data.state, "year": config.data.year},
        "legacy_preconsolidated_columns_excluded": list(MODEL_EXCLUDED_PRECONSOLIDATED_COLUMNS),
    }
    if transactions is None and source_path.exists():
        source_metadata.update(
            {
                "bytes": source_path.stat().st_size,
                "sha256": _sha256(source_path),
            }
        )

    metadata = {
        "model_name": champion_name,
        "model_version": config.project.model_version,
        "training_cutoff": str(config.data.train_end),
        "validation_cutoff": str(config.data.validation_end),
        "selected_threshold": locked_threshold,
        "selection_metric": "validation_pr_auc",
        "feature_schema": schema.to_dict(),
        "target_history_policy": (
            "enabled; assumes prior labels are available immediately"
            if schema.target_history_enabled
            else "disabled because label-availability timestamps are absent"
        ),
        "cost_assumptions": asdict(config.threshold),
        "data": source_metadata,
        "random_seed": config.project.seed,
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    artifact = {
        "artifact_type": "fraud_detection_pipeline",
        "model_version": config.project.model_version,
        "pipeline": champion,
        "threshold": locked_threshold,
        "feature_schema": schema.to_dict(),
        "metadata": metadata,
    }
    artifact_path = artifact_dir / config.paths.model_filename
    joblib.dump(artifact, artifact_path, compress=3)
    _write_json(artifact_dir / "model_metadata.json", metadata)

    metrics_payload = {
        "validation": validation_metrics,
        "test": test_metrics,
        "locked_threshold": locked_threshold,
        "champion_model": champion_name,
        "cost_assumptions": asdict(config.threshold),
    }
    _write_json(report_dir / "metrics.json", metrics_payload)
    (report_dir / "evaluation.md").write_text(
        _render_evaluation_report(
            split_summary=split_summary,
            comparison=comparison,
            champion_name=champion_name,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            config=config,
            target_history_enabled=schema.target_history_enabled,
        ),
        encoding="utf-8",
    )
    return {
        "artifact_path": artifact_path,
        "metadata_path": artifact_dir / "model_metadata.json",
        "report_path": report_dir / "evaluation.md",
        **metrics_payload,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="TOML configuration path",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override the configured CSV/pickle input path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        result = run_training(load_config(args.config), input_path=args.input)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_jsonable(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
