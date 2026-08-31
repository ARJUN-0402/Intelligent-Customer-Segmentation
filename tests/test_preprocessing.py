"""Tests for src.preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CustomerDataPreprocessor,
    remove_duplicates,
    scale_features,
    select_numeric_features,
)
from src.utils import ANNUAL_INCOME, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, GENRE, SPENDING_SCORE


def test_remove_duplicates_counts():
    raw = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    cleaned, removed = remove_duplicates(raw)
    assert removed == 1
    assert len(cleaned) == 2


def test_select_numeric_features(df: pd.DataFrame):
    out = select_numeric_features(df, features=FEATURE_COLUMNS)
    assert list(out.columns) == [ANNUAL_INCOME, SPENDING_SCORE]


def test_scale_features_returns_fitted():
    X = np.array([[0.0], [2.0], [4.0]])
    scaled, scaler = scale_features(X)
    assert scaled.shape == X.shape
    assert scaled.mean() == pytest.approx(0.0, abs=1e-9)
    assert np.isclose(scaled.std(), 1.0, atol=1e-6)


def test_preprocessor_fit_transform_shape(df: pd.DataFrame):
    pre = CustomerDataPreprocessor(
        numeric_features=FEATURE_COLUMNS, categorical_features=[]
    )
    X, cleaned = pre.fit_transform(df)
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert X.shape[0] == len(cleaned)
    assert ANNUAL_INCOME in cleaned.columns


def test_preprocessor_with_categorical(df: pd.DataFrame):
    pre = CustomerDataPreprocessor(
        numeric_features=FEATURE_COLUMNS,
        categorical_features=CATEGORICAL_COLUMNS,
    )
    X, cleaned = pre.fit_transform(df)
    n_out = len(FEATURE_COLUMNS) + cleaned[GENRE].nunique()
    assert X.shape[1] == n_out


def test_inverse_transform_centers_roundtrip(df: pd.DataFrame):
    pre = CustomerDataPreprocessor(
        numeric_features=FEATURE_COLUMNS, categorical_features=[]
    )
    X, cleaned = pre.fit_transform(df)
    recovered = pre.inverse_transform_centers(X)
    assert list(recovered.columns) == FEATURE_COLUMNS
    np.testing.assert_allclose(
        recovered.values, cleaned[FEATURE_COLUMNS].values, atol=1e-6
    )


def test_preprocessor_requires_fit():
    pre = CustomerDataPreprocessor()
    with pytest.raises(RuntimeError):
        pre.transform(pd.DataFrame({ANNUAL_INCOME: [1], SPENDING_SCORE: [2]}))
