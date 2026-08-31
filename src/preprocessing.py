"""Preprocessing: feature selection, missing-value / duplicate handling,
categorical encoding, and feature scaling in a single reusable pipeline.

The :class:`CustomerDataPreprocessor` wraps a scikit-learn ``ColumnTransformer``
so the same class can be reused for full profiling (numeric + categorical) or
for clustering-only workflows (numeric-only).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import (
    CATEGORICAL_COLUMNS,
    CUSTOMER_ID,
    NUMERIC_COLUMNS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone helpers (thin wrappers kept for reuse / testing)
# ---------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return a deduplicated copy of *df* and the number of rows removed."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    if removed:
        logger.info("Removed %d duplicate rows", removed)
    return cleaned, removed


def select_numeric_features(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return only the *features* columns (defaults to numeric dataset cols)."""
    cols = list(features) if features is not None else NUMERIC_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot select numeric features; missing: {missing}")
    return df[cols].copy()


def build_scaler() -> StandardScaler:
    """Return a fresh :class:`StandardScaler` instance."""
    return StandardScaler()


def scale_features(
    X: np.ndarray, scaler: Optional[StandardScaler] = None
) -> tuple[np.ndarray, StandardScaler]:
    """Fit-or-transform *X*; returns scaled values and the fitted scaler."""
    if scaler is None:
        scaler = build_scaler()
        return scaler.fit_transform(X), scaler
    return scaler.transform(X), scaler


# ---------------------------------------------------------------------------
# Reusable preprocessor
# ---------------------------------------------------------------------------
class CustomerDataPreprocessor:
    """Reusable sklearn-compatible preprocessing pipeline.

    Handles duplicate removal, missing-value imputation, one-hot encoding of
    categoricals, and standard scaling of numerics - all within a single
    :class:`~sklearn.compose.ColumnTransformer`.

    Parameters
    ----------
    numeric_features:
        Numeric columns to impute + scale. Defaults to :data:`NUMERIC_COLUMNS`.
    categorical_features:
        Categorical columns to impute + one-hot encode. Defaults to
        :data:`CATEGORICAL_COLUMNS`.
    id_column:
        Identifier column dropped from the feature matrix but preserved in
        the cleaned dataframe. Defaults to ``"CustomerID"``.
    drop_duplicates:
        Whether to remove duplicate rows before transforming.

    Attributes
    ----------
    transformer_ :
        The fitted :class:`ColumnTransformer` (available after :meth:`fit`).
    feature_names_out_ :
        Output feature names after transformation.
    """

    def __init__(
        self,
        numeric_features: Optional[Sequence[str]] = None,
        categorical_features: Optional[Sequence[str]] = None,
        id_column: str = CUSTOMER_ID,
        drop_duplicates: bool = True,
    ) -> None:
        self.numeric_features: list[str] = (
            list(numeric_features) if numeric_features is not None else NUMERIC_COLUMNS
        )
        self.categorical_features: list[str] = (
            list(categorical_features)
            if categorical_features is not None
            else CATEGORICAL_COLUMNS
        )
        self.id_column: str = id_column
        self.drop_duplicates: bool = drop_duplicates
        self.transformer_: Optional[ColumnTransformer] = None

    # -- internal ----------------------------------------------------------
    def _build_transformer(self) -> ColumnTransformer:
        transformers: list[tuple[str, object, list[str]]] = [
            (
                "num",
                SkPipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                self.numeric_features,
            )
        ]
        if self.categorical_features:
            transformers.append(
                (
                    "cat",
                    SkPipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    self.categorical_features,
                )
            )
        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    # -- API ---------------------------------------------------------------
    @property
    def output_feature_names(self) -> list[str]:
        """Feature names produced by the fitted transformer."""
        if self.transformer_ is None:
            return self.numeric_features + self.categorical_features
        return list(self.transformer_.get_feature_names_out())

    def fit(self, df: pd.DataFrame) -> "CustomerDataPreprocessor":
        """Fit the internal transformer on *df*."""
        transformer = self._build_transformer()
        transformer.fit(df)
        self.transformer_ = transformer
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform *df* into the scaled/encoded feature matrix."""
        if self.transformer_ is None:
            raise RuntimeError("Transformer is not fitted; call fit() first.")
        return self.transformer_.transform(df)

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Clean, fit, and transform *df* in one call.

        Returns
        -------
        tuple
            ``(X, cleaned_df)`` where ``X`` is the transformed feature matrix
            and ``cleaned_df`` is the deduplicated dataframe (id preserved)
            whose row order matches ``X``.
        """
        cleaned = df.copy()
        if self.drop_duplicates:
            cleaned, removed = remove_duplicates(cleaned)
            logger.info(
                "Preprocessing: %d rows retained after cleaning", len(cleaned)
            )

        transformer = self._build_transformer()
        X = transformer.fit_transform(cleaned)
        self.transformer_ = transformer
        return X, cleaned

    @property
    def n_features_out(self) -> int:
        """Number of features after transformation."""
        return len(self.output_feature_names)

    def inverse_transform_centers(
        self, scaled_centers: np.ndarray
    ) -> pd.DataFrame:
        """Inverse-transform cluster centres back to original feature scale.

        Works for a **numeric-only** preprocessor: the fitted ``StandardScaler``
        is retrieved from the numeric pipeline and applied directly to the
        scaled centres.

        Parameters
        ----------
        scaled_centers:
            Cluster centres in the transformed (scaled) feature space.

        Returns
        -------
        pandas.DataFrame
            Centres mapped back to the original numeric feature scale.
        """
        if self.transformer_ is None:
            raise RuntimeError("Transformer is not fitted; call fit() first.")
        if not self.numeric_features:
            raise RuntimeError(
                "No numeric features configured; cannot inverse-transform centres."
            )
        if self.categorical_features:
            raise RuntimeError(
                "inverse_transform_centers requires a numeric-only preprocessor "
                "(categorical_features must be empty)."
            )
        num_pipe = self.transformer_.named_transformers_["num"]
        scaler = num_pipe.named_steps["scaler"]
        inverse = scaler.inverse_transform(scaled_centers)
        return pd.DataFrame(inverse, columns=list(self.numeric_features))
