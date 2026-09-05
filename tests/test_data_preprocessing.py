from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fraud_detection.data import prepare_transactions
from fraud_detection.features import VELOCITY_FEATURE_COLUMNS, build_features
from fraud_detection.modeling import get_feature_schema, select_model_matrix
from fraud_detection.preprocessing import build_pipeline


def _source_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cc_num": ["4111111111111111", "4111111111111111"],
            "trans_num": ["tx-1", "tx-2"],
            "trans_timestamp": ["2020-01-01 10:00:00", "2020-01-02 10:00:00"],
            "amt": [10.0, 20.0],
            "is_fraud": [0, 1],
            "category": [" Grocery ", "ONLINE"],
            "profile": [" Urban ", "RURAL"],
            "city_pop": [100_000, 25_000],
            "dob": ["1980-05-10", "1980-05-10"],
            "lat": [34.05, 34.05],
            "long": [-118.24, -118.24],
            "merch_lat": [34.06, 34.06],
            "merch_long": [-118.25, -118.25],
            "gender": ["M", "F"],
            "state": ["CA", "CA"],
        }
    )


def test_data_preparation_normalizes_schema_and_target_is_never_modeled() -> None:
    prepared = prepare_transactions(_source_transactions(), state="CA", year=2020)

    assert prepared["category"].tolist() == ["grocery", "online"]
    assert prepared["profile"].tolist() == ["urban", "rural"]
    assert prepared["is_male"].tolist() == [1, 0]
    assert prepared["is_fraud"].dtype == np.dtype("int8")
    assert pd.api.types.is_datetime64_any_dtype(prepared["trans_timestamp"])

    schema = get_feature_schema()
    matrix = select_model_matrix(build_features(prepared), schema)

    assert matrix.columns.tolist() == list(schema.model_features)
    assert set(VELOCITY_FEATURE_COLUMNS).issubset(schema.model_features)
    assert "is_fraud" not in matrix.columns
    assert "CC_PREV_FRAUD" not in matrix.columns
    assert "CC_HIST_FRAUD_RATE" not in matrix.columns


def test_preprocessing_is_train_fitted_and_unseen_categories_predict_safely() -> None:
    train = pd.DataFrame(
        {
            "amt": [10.0, 20.0, 30.0, 40.0],
            "city_pop": [100.0, 200.0, 300.0, 400.0],
            "category": ["food", "fuel", "food", "fuel"],
            "profile": ["urban", "urban", "rural", "rural"],
            # Deliberately present in the input: ColumnTransformer must drop it.
            "is_fraud": [0, 0, 1, 1],
        }
    )
    target = train["is_fraud"]
    validation = pd.DataFrame(
        {
            "amt": [1_000_000.0, -1_000_000.0],
            "city_pop": [9_000_000.0, -9_000_000.0],
            "category": ["crypto", "travel"],
            "profile": ["new-profile", "another-new-profile"],
        }
    )
    pipeline = build_pipeline(
        LogisticRegression(solver="liblinear", random_state=7),
        numerical_features=("amt", "city_pop"),
        categorical_features=("category", "profile"),
        iqr_columns=("amt", "city_pop"),
    )

    pipeline.fit(train, target)

    clipper = pipeline.named_steps["iqr"]
    assert clipper.lower_bounds_.to_dict() == pytest.approx({"amt": -5.0, "city_pop": -50.0})
    assert clipper.upper_bounds_.to_dict() == pytest.approx({"amt": 55.0, "city_pop": 550.0})
    clipped_validation = clipper.transform(validation)
    assert clipped_validation["amt"].tolist() == pytest.approx([55.0, -5.0])
    assert clipped_validation["city_pop"].tolist() == pytest.approx([550.0, -50.0])

    preprocessor = pipeline.named_steps["preprocessor"]
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    assert scaler.mean_.tolist() == pytest.approx([25.0, 250.0])
    assert scaler.scale_.tolist() == pytest.approx(
        [np.std([10.0, 20.0, 30.0, 40.0]), np.std([100.0, 200.0, 300.0, 400.0])]
    )

    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    assert encoder.categories_[0].tolist() == ["food", "fuel"]
    assert encoder.categories_[1].tolist() == ["rural", "urban"]
    assert "crypto" not in encoder.categories_[0]
    assert "new-profile" not in encoder.categories_[1]
    assert "is_fraud" not in preprocessor.get_feature_names_out()

    transformed = preprocessor.transform(clipped_validation)
    probabilities = pipeline.predict_proba(validation)[:, 1]
    assert transformed.shape == (2, 6)
    assert np.isfinite(transformed.toarray()).all()
    assert probabilities.shape == (2,)
    assert np.isfinite(probabilities).all()
    assert ((0.0 <= probabilities) & (probabilities <= 1.0)).all()
