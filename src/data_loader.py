"""Reusable dataset loading with validation, schema checks, and clear exceptions.

The loader is intentionally decoupled from any hard-coded path: callers pass a
:class:`pathlib.Path` and receive a validated :class:`pandas.DataFrame`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .utils import (
    ANNUAL_INCOME,
    AGE,
    CATEGORICAL_COLUMNS,
    CUSTOMER_ID,
    FEATURE_COLUMNS,
    GENRE,
    LABEL_MAP,
    NUMERIC_COLUMNS,
    SPENDING_SCORE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------
# Each column maps to an expected dtype *kind*: "numeric" or "categorical".
EXPECTED_SCHEMA: dict[str, str] = {
    CUSTOMER_ID: "numeric",
    GENRE: "categorical",
    AGE: "numeric",
    ANNUAL_INCOME: "numeric",
    SPENDING_SCORE: "numeric",
}

REQUIRED_COLUMNS: list[str] = list(EXPECTED_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class DataError(Exception):
    """Base exception for all data-loading and validation failures."""


class FileNotFoundError_(DataError):  # noqa: N818 - intentional, clear name
    """Raised when the dataset file cannot be located."""


class EmptyDatasetError(DataError):
    """Raised when the loaded dataset contains zero rows."""

class MissingColumnError(DataError):
    """Raised when one or more required columns are absent."""


class SchemaValidationError(DataError):
    """Raised when a column's dtype does not match the expected schema."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _check_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError_(f"Dataset file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError_(f"Path is not a file: {path}")


def check_empty(df: pd.DataFrame) -> None:
    """Raise :class:`EmptyDatasetError` if *df* has no rows."""
    if df.empty:
        raise EmptyDatasetError(
            "The dataset contains zero rows. Please provide a valid CSV file."
        )


def check_missing_columns(
    df: pd.DataFrame, required: Optional[Sequence[str]] = None
) -> None:
    """Raise :class:`MissingColumnError` for any missing *required* columns."""
    required_cols = list(required) if required is not None else REQUIRED_COLUMNS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise MissingColumnError(
            f"Missing required columns: {missing}. "
            f"Expected columns: {required_cols}."
        )


def validate_schema(
    df: pd.DataFrame,
    expected: Optional[dict[str, str]] = None,
) -> None:
    """Validate that each expected column has the correct dtype kind.

    Parameters
    ----------
    df:
        DataFrame to validate.
    expected:
        Mapping of column name to ``"numeric"`` or ``"categorical"``.
        Defaults to :data:`EXPECTED_SCHEMA`.
    """
    schema = expected if expected is not None else EXPECTED_SCHEMA
    for col, kind in schema.items():
        if col not in df.columns:
            continue
        series = df[col]
        if kind == "numeric":
            if not pd.api.types.is_numeric_dtype(series):
                raise SchemaValidationError(
                    f"Column '{col}' is expected to be numeric but has "
                    f"dtype '{series.dtype}'."
                )
        elif kind == "categorical":
            if not (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):
                raise SchemaValidationError(
                    f"Column '{col}' is expected to be categorical but has "
                    f"dtype '{series.dtype}'."
                )


def validate_dataframe(
    df: pd.DataFrame,
    required: Optional[Sequence[str]] = None,
    expected: Optional[dict[str, str]] = None,
) -> None:
    """Run the full validation pipeline: empty / missing-columns / schema."""
    check_empty(df)
    check_missing_columns(df, required)
    validate_schema(df, expected)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(
    path: Path,
    required_columns: Optional[Sequence[str]] = None,
    expected_schema: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Load and validate a CSV dataset from *path*.

    Parameters
    ----------
    path:
        Location of the CSV file (a :class:`pathlib.Path`).
    required_columns:
        Columns that must be present. Defaults to :data:`REQUIRED_COLUMNS`.
    expected_schema:
        Optional dtype schema used with :func:`validate_schema`.

    Returns
    -------
    pandas.DataFrame
        Validated customer dataframe.

    Raises
    ------
    FileNotFoundError_
        If *path* does not exist.
    EmptyDatasetError
        If the file parses to zero rows.
    MissingColumnError
        If any required column is absent.
    SchemaValidationError
        If a column's dtype does not match the schema.
    """
    _check_file_exists(path)
    logger.info("Loading dataset from %s", path)
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise DataError(f"Failed to parse CSV file '{path.name}': {exc}") from exc

    validate_dataframe(df, required=required_columns, expected=expected_schema)

    logger.info(
        "Loaded %d rows x %d columns from %s",
        len(df),
        len(df.columns),
        path.name,
    )
    return df


def summarize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact dtype summary table for quick inspection."""
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": df.isna().sum(),
            "unique": df.nunique(),
        }
    )
    return summary
