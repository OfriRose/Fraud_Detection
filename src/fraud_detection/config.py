"""Typed loading and validation for the training configuration."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config/training.toml")
_SEMANTIC_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


class ConfigError(ValueError):
    """Raised when a training configuration is missing or invalid."""


@dataclass(frozen=True)
class ProjectConfig:
    """Experiment-wide reproducibility and version settings."""

    seed: int
    model_version: str


@dataclass(frozen=True)
class DataConfig:
    """Input schema and chronological split boundaries."""

    input_path: Path
    state: str
    year: int
    include_target_history: bool
    target_column: str
    timestamp_column: str
    entity_column: str
    train_end: datetime
    validation_end: datetime


@dataclass(frozen=True)
class PathsConfig:
    """Locations for reproducible outputs."""

    artifact_dir: Path
    report_dir: Path
    model_filename: str


@dataclass(frozen=True)
class ThresholdConfig:
    """Operational assumptions used to lock a validation threshold."""

    false_negative_cost: float
    false_positive_cost: float
    max_review_rate: float


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Candidate logistic-regression parameters."""

    solver: str
    penalty: str
    C: float
    class_weight: str | None
    max_iter: int


@dataclass(frozen=True)
class XGBoostConfig:
    """Candidate XGBoost parameters sized for the full transaction dataset."""

    objective: str
    eval_metric: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    tree_method: str
    n_jobs: int


@dataclass(frozen=True)
class ModelCandidatesConfig:
    """Parameters for models compared under the same temporal protocol."""

    logistic_regression: LogisticRegressionConfig
    xgboost: XGBoostConfig


@dataclass(frozen=True)
class TrainingConfig:
    """Complete immutable configuration for a training run."""

    project: ProjectConfig
    data: DataConfig
    paths: PathsConfig
    threshold: ThresholdConfig
    models: ModelCandidatesConfig


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid [{name}] section")
    return value


def _value(section: dict[str, Any], key: str, expected_type: type[Any]) -> Any:
    if key not in section:
        raise ConfigError(f"Missing required configuration value: {key}")
    value = section[key]
    if expected_type is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise ConfigError(f"{key} must be an integer")
    if expected_type is float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ConfigError(f"{key} must be a number")
        return float(value)
    if not isinstance(value, expected_type):
        raise ConfigError(f"{key} must be a {expected_type.__name__}")
    return value


def _optional_string(section: dict[str, Any], key: str) -> str | None:
    value = section.get(key)
    if value is not None and not isinstance(value, str):
        raise ConfigError(f"{key} must be a string or null")
    return value


def _datetime_value(section: dict[str, Any], key: str) -> datetime:
    value = section.get(key)
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise ConfigError(f"{key} must be an ISO-8601 datetime") from exc
    raise ConfigError(f"{key} must be an ISO-8601 datetime")


def _non_empty_string(section: dict[str, Any], key: str) -> str:
    value = _value(section, key, str)
    if not value.strip():
        raise ConfigError(f"{key} must not be empty")
    return value


def _unit_interval(section: dict[str, Any], key: str) -> float:
    value = _value(section, key, float)
    if not 0.0 < value <= 1.0:
        raise ConfigError(f"{key} must be in the interval (0, 1]")
    return value


def _non_negative(section: dict[str, Any], key: str) -> float:
    value = _value(section, key, float)
    if value < 0.0:
        raise ConfigError(f"{key} must be non-negative")
    return value


def _positive_int(section: dict[str, Any], key: str) -> int:
    value = _value(section, key, int)
    if value <= 0:
        raise ConfigError(f"{key} must be positive")
    return value


def _load_project(section: dict[str, Any]) -> ProjectConfig:
    seed = _value(section, "seed", int)
    if seed < 0:
        raise ConfigError("seed must be non-negative")
    model_version = _non_empty_string(section, "model_version")
    if not _SEMANTIC_VERSION.fullmatch(model_version):
        raise ConfigError("model_version must use MAJOR.MINOR.PATCH format")
    return ProjectConfig(seed=seed, model_version=model_version)


def _load_data(section: dict[str, Any]) -> DataConfig:
    train_end = _datetime_value(section, "train_end")
    validation_end = _datetime_value(section, "validation_end")
    if train_end >= validation_end:
        raise ConfigError("train_end must be earlier than validation_end")
    year = _positive_int(section, "year")
    if year < 1900 or year > 9999:
        raise ConfigError("year must be between 1900 and 9999")
    return DataConfig(
        input_path=Path(_non_empty_string(section, "input_path")),
        state=_non_empty_string(section, "state"),
        year=year,
        include_target_history=_value(section, "include_target_history", bool),
        target_column=_non_empty_string(section, "target_column"),
        timestamp_column=_non_empty_string(section, "timestamp_column"),
        entity_column=_non_empty_string(section, "entity_column"),
        train_end=train_end,
        validation_end=validation_end,
    )


def _load_paths(section: dict[str, Any]) -> PathsConfig:
    model_filename = _non_empty_string(section, "model_filename")
    if Path(model_filename).name != model_filename:
        raise ConfigError("model_filename must be a filename, not a path")
    return PathsConfig(
        artifact_dir=Path(_non_empty_string(section, "artifact_dir")),
        report_dir=Path(_non_empty_string(section, "report_dir")),
        model_filename=model_filename,
    )


def _load_threshold(section: dict[str, Any]) -> ThresholdConfig:
    false_negative_cost = _non_negative(section, "false_negative_cost")
    false_positive_cost = _non_negative(section, "false_positive_cost")
    if false_negative_cost == false_positive_cost == 0.0:
        raise ConfigError("At least one error cost must be positive")
    return ThresholdConfig(
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
        max_review_rate=_unit_interval(section, "max_review_rate"),
    )


def _load_logistic(section: dict[str, Any]) -> LogisticRegressionConfig:
    c_value = _value(section, "C", float)
    if c_value <= 0.0:
        raise ConfigError("C must be positive")
    return LogisticRegressionConfig(
        solver=_non_empty_string(section, "solver"),
        penalty=_non_empty_string(section, "penalty"),
        C=c_value,
        class_weight=_optional_string(section, "class_weight"),
        max_iter=_positive_int(section, "max_iter"),
    )


def _load_xgboost(section: dict[str, Any]) -> XGBoostConfig:
    learning_rate = _value(section, "learning_rate", float)
    min_child_weight = _non_negative(section, "min_child_weight")
    subsample = _unit_interval(section, "subsample")
    colsample_bytree = _unit_interval(section, "colsample_bytree")
    reg_alpha = _non_negative(section, "reg_alpha")
    reg_lambda = _non_negative(section, "reg_lambda")
    if learning_rate <= 0.0:
        raise ConfigError("learning_rate must be positive")
    return XGBoostConfig(
        objective=_non_empty_string(section, "objective"),
        eval_metric=_non_empty_string(section, "eval_metric"),
        n_estimators=_positive_int(section, "n_estimators"),
        learning_rate=learning_rate,
        max_depth=_positive_int(section, "max_depth"),
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        tree_method=_non_empty_string(section, "tree_method"),
        n_jobs=_value(section, "n_jobs", int),
    )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> TrainingConfig:
    """Load and validate a TOML training configuration."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            raw = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc

    project = _load_project(_section(raw, "project"))
    data = _load_data(_section(raw, "data"))
    paths = _load_paths(_section(raw, "paths"))
    threshold = _load_threshold(_section(raw, "threshold"))
    models_section = _section(raw, "models")
    models = ModelCandidatesConfig(
        logistic_regression=_load_logistic(_section(models_section, "logistic_regression")),
        xgboost=_load_xgboost(_section(models_section, "xgboost")),
    )
    return TrainingConfig(
        project=project,
        data=data,
        paths=paths,
        threshold=threshold,
        models=models,
    )


__all__ = [
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "DataConfig",
    "LogisticRegressionConfig",
    "ModelCandidatesConfig",
    "PathsConfig",
    "ProjectConfig",
    "ThresholdConfig",
    "TrainingConfig",
    "XGBoostConfig",
    "load_config",
]
