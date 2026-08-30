"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_data
from src.preprocessing import CustomerDataPreprocessor
from src.utils import DEFAULT_DATA_PATH, FEATURE_COLUMNS


@pytest.fixture(scope="session")
def data_path() -> Path:
    return DEFAULT_DATA_PATH


@pytest.fixture(scope="session")
def df(data_path: Path) -> pd.DataFrame:
    return load_data(data_path)


@pytest.fixture(scope="session")
def X_scaled(df: pd.DataFrame) -> tuple[np.ndarray, CustomerDataPreprocessor]:
    pre = CustomerDataPreprocessor(
        numeric_features=FEATURE_COLUMNS, categorical_features=[]
    )
    X, cleaned = pre.fit_transform(df)
    return X, pre
