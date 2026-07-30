"""Train-fitted preprocessing for fraud models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip configured DataFrame columns to train-fitted IQR bounds."""

    def __init__(self, columns: Sequence[str], factor: float = 1.5) -> None:
        self.columns = columns
        self.factor = factor

    def fit(self, X: pd.DataFrame, y: object = None) -> IQRClipper:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("IQRClipper requires a pandas DataFrame")
        if self.factor < 0:
            raise ValueError("factor must be non-negative")
        missing = set(self.columns).difference(X.columns)
        if missing:
            raise ValueError("IQR columns missing from input: " + ", ".join(sorted(missing)))

        numeric = X.loc[:, list(self.columns)].apply(pd.to_numeric, errors="coerce")
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ("lower_bounds_", "upper_bounds_", "feature_names_in_"))
        if not isinstance(X, pd.DataFrame):
            raise TypeError("IQRClipper requires a pandas DataFrame")
        missing = set(self.columns).difference(X.columns)
        if missing:
            raise ValueError("IQR columns missing from input: " + ", ".join(sorted(missing)))

        transformed = X.copy()
        for column in self.columns:
            transformed[column] = pd.to_numeric(transformed[column], errors="coerce").clip(
                lower=self.lower_bounds_[column],
                upper=self.upper_bounds_[column],
            )
        return transformed

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features, dtype=object)


def build_preprocessor(
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Create numeric and categorical branches with safe unseen-category handling."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numerical_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def build_pipeline(
    estimator: object,
    *,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    iqr_columns: Sequence[str] = ("amt", "city_pop"),
) -> Pipeline:
    """Bundle all learned transformations and the estimator in one pipeline."""

    all_features = tuple(numerical_features) + tuple(categorical_features)
    if len(set(all_features)) != len(all_features):
        raise ValueError("Feature lists contain duplicates")
    if "is_fraud" in all_features:
        raise AssertionError("Target is not allowed in the feature matrix")
    invalid_iqr = set(iqr_columns).difference(all_features)
    if invalid_iqr:
        raise ValueError("IQR columns must be modeled features: " + ", ".join(sorted(invalid_iqr)))

    return Pipeline(
        steps=[
            ("iqr", IQRClipper(columns=tuple(iqr_columns))),
            ("preprocessor", build_preprocessor(numerical_features, categorical_features)),
            ("model", estimator),
        ]
    )


__all__ = ["IQRClipper", "build_pipeline", "build_preprocessor"]
