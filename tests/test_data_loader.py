"""Tests for src.data_loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    EmptyDatasetError,
    FileNotFoundError_,
    MissingColumnError,
    SchemaValidationError,
    check_empty,
    check_missing_columns,
    load_data,
    summarize_dtypes,
    validate_dataframe,
)
from src.data_loader import REQUIRED_COLUMNS as REQUIRED
from src.utils import ANNUAL_INCOME


@pytest.fixture(scope="module")
def good_df(data_path: Path) -> pd.DataFrame:
    return load_data(data_path)


def test_load_data_returns_dataframe(good_df: pd.DataFrame):
    assert isinstance(good_df, pd.DataFrame)
    assert not good_df.empty
    assert len(good_df) == 200


def test_required_columns_present(good_df: pd.DataFrame):
    for col in REQUIRED:
        assert col in good_df.columns


def test_check_empty_raises():
    with pytest.raises(EmptyDatasetError):
        check_empty(pd.DataFrame(columns=["a"]))


def test_check_missing_columns_raises():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(MissingColumnError):
        check_missing_columns(df, required=["b", "c"])


def test_schema_validation_passes(good_df: pd.DataFrame):
    validate_dataframe(good_df)


def test_schema_validation_fails_on_bad_dtype(data_path: Path):
    df = pd.read_csv(data_path)
    df[ANNUAL_INCOME] = df[ANNUAL_INCOME].astype(str)
    with pytest.raises(SchemaValidationError):
        validate_dataframe(df)


def test_file_not_found(tmp_path: Path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError_):
        load_data(missing)


def test_empty_csv_raises(tmp_path: Path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("CustomerID,Genre,Age,Annual Income (k$),Spending Score (1-100)\n")
    with pytest.raises(EmptyDatasetError):
        load_data(empty_csv)


def test_missing_column_raises(tmp_path: Path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("CustomerID,Age\n1,19\n2,21\n")
    with pytest.raises(MissingColumnError):
        load_data(bad_csv)


def test_summarize_dtypes_shape(good_df: pd.DataFrame):
    summary = summarize_dtypes(good_df)
    assert "dtype" in summary.columns
    assert "missing" in summary.columns


def test_malformed_csv_raises(tmp_path: Path):
    bad_csv = tmp_path / "malformed.csv"
    bad_csv.write_text("CustomerID,Genre,Age,Annual Income (k$),Spending Score (1-100)\n1,Male,not_a_number,60,50\n")
    with pytest.raises(Exception):
        load_data(bad_csv)


def test_invalid_numeric_values_raises(tmp_path: Path):
    bad_csv = tmp_path / "bad_numeric.csv"
    bad_csv.write_text("CustomerID,Genre,Age,Annual Income (k$),Spending Score (1-100)\n1,Male,abc,60,50\n")
    with pytest.raises(Exception):
        load_data(bad_csv)
