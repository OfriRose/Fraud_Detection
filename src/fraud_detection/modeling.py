"""Feature schema and candidate end-to-end model pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_detection.config import TrainingConfig, XGBoostConfig
from fraud_detection.features import VELOCITY_FEATURE_COLUMNS
from fraud_detection.preprocessing import build_pipeline

BASE_NUMERICAL_FEATURES = (
    "amt",
    "city_pop",
    "is_male",
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
TARGET_HISTORY_FEATURES = ("CC_PREV_FRAUD", "CC_HIST_FRAUD_RATE")
CATEGORICAL_FEATURES = ("category", "profile")
BINARY_FEATURES = ("is_male", "IS_WEEKEND", "IS_FIRST_CARD_TX")
IDENTIFIER_FEATURES = ("ssn", "cc_num", "acct_num", "trans_num", "CC_BIN")
TIMESTAMP_FEATURES = ("trans_date", "trans_timestamp", "dob")
EXCLUDED_FEATURES = (
    "is_fraud",
    "city",
    "job",
    "merchant",
    "zip",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "Unnamed: 0",
)


@dataclass(frozen=True)
class FeatureSchema:
    """Explicit feature roles used by training and inference."""

    numerical: tuple[str, ...]
    categorical: tuple[str, ...]
    binary: tuple[str, ...]
    identifiers: tuple[str, ...]
    timestamps: tuple[str, ...]
    excluded: tuple[str, ...]
    target_history_enabled: bool

    @property
    def model_features(self) -> tuple[str, ...]:
        return self.numerical + self.categorical

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_features"] = list(self.model_features)
        return payload


def get_feature_schema(include_target_history: bool = False) -> FeatureSchema:
    numerical = BASE_NUMERICAL_FEATURES
    excluded = EXCLUDED_FEATURES
    if include_target_history:
        numerical += TARGET_HISTORY_FEATURES
    else:
        excluded += TARGET_HISTORY_FEATURES
    schema = FeatureSchema(
        numerical=numerical,
        categorical=CATEGORICAL_FEATURES,
        binary=BINARY_FEATURES,
        identifiers=IDENTIFIER_FEATURES,
        timestamps=TIMESTAMP_FEATURES,
        excluded=excluded,
        target_history_enabled=include_target_history,
    )
    if "is_fraud" in schema.model_features:
        raise AssertionError("Target must not be included in model features")
    return schema


def select_model_matrix(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    missing = set(schema.model_features).difference(frame.columns)
    if missing:
        raise ValueError("Model features missing from frame: " + ", ".join(sorted(missing)))
    matrix = frame.loc[:, list(schema.model_features)].copy()
    if "is_fraud" in matrix.columns:
        raise AssertionError("Target must not be included in model matrix")
    return matrix


def _xgboost_estimator(
    model_config: XGBoostConfig,
    *,
    seed: int,
    scale_pos_weight: float,
) -> XGBClassifier:
    return XGBClassifier(
        objective=model_config.objective,
        eval_metric=model_config.eval_metric,
        n_estimators=model_config.n_estimators,
        learning_rate=model_config.learning_rate,
        max_depth=model_config.max_depth,
        min_child_weight=model_config.min_child_weight,
        subsample=model_config.subsample,
        colsample_bytree=model_config.colsample_bytree,
        reg_alpha=model_config.reg_alpha,
        reg_lambda=model_config.reg_lambda,
        tree_method=model_config.tree_method,
        n_jobs=model_config.n_jobs,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )


def build_candidate_pipelines(
    config: TrainingConfig,
    y_train: pd.Series,
    schema: FeatureSchema,
) -> dict[str, Pipeline]:
    """Build baseline and tuned candidates with identical train-fitted preprocessing."""

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Training target must contain both fraud and legitimate cases")
    scale_pos_weight = negatives / positives

    logistic_config = config.models.logistic_regression
    logistic_parameters: dict[str, Any] = {
        "solver": logistic_config.solver,
        "C": logistic_config.C,
        "class_weight": logistic_config.class_weight,
        "max_iter": logistic_config.max_iter,
        "random_state": config.project.seed,
        "tol": 1e-3,
    }
    # Scikit-learn 1.8+ infers ordinary L2 regularization when `penalty` is
    # omitted. Avoid passing its deprecated spelling while retaining explicit
    # support for non-default configurations on older supported releases.
    if logistic_config.penalty != "l2":
        logistic_parameters["penalty"] = logistic_config.penalty
    logistic = LogisticRegression(
        **logistic_parameters,
    )

    configured_xgb = config.models.xgboost
    conservative_xgb = replace(
        configured_xgb,
        max_depth=max(2, configured_xgb.max_depth - 1),
        min_child_weight=max(10.0, configured_xgb.min_child_weight),
        reg_alpha=max(0.5, configured_xgb.reg_alpha),
    )

    estimators: dict[str, object] = {
        "prevalence_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": logistic,
        "xgboost_conservative": _xgboost_estimator(
            conservative_xgb,
            seed=config.project.seed,
            scale_pos_weight=scale_pos_weight,
        ),
        "xgboost_configured": _xgboost_estimator(
            configured_xgb,
            seed=config.project.seed,
            scale_pos_weight=scale_pos_weight,
        ),
    }
    return {
        name: build_pipeline(
            estimator,
            numerical_features=schema.numerical,
            categorical_features=schema.categorical,
        )
        for name, estimator in estimators.items()
    }


def public_model_parameters(pipeline: Pipeline) -> dict[str, Any]:
    """Return estimator parameters suitable for an experiment ledger."""

    estimator = pipeline.named_steps["model"]
    parameters = estimator.get_params(deep=False)
    serializable: dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, str | int | float | bool) or value is None:
            serializable[key] = value
    return serializable


__all__ = [
    "BASE_NUMERICAL_FEATURES",
    "BINARY_FEATURES",
    "CATEGORICAL_FEATURES",
    "EXCLUDED_FEATURES",
    "FeatureSchema",
    "IDENTIFIER_FEATURES",
    "TARGET_HISTORY_FEATURES",
    "TIMESTAMP_FEATURES",
    "build_candidate_pipelines",
    "get_feature_schema",
    "public_model_parameters",
    "select_model_matrix",
]
