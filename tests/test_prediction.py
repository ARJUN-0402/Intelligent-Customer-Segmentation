"""Tests for app.predict_new and the prediction pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import predict_new
from src.data_loader import load_data
from src.preprocessing import CustomerDataPreprocessor
from src.utils import DEFAULT_DATA_PATH, FEATURE_COLUMNS


@pytest.fixture(scope="session")
def prediction_bundle():
    df = load_data(DEFAULT_DATA_PATH)
    pre = CustomerDataPreprocessor(
        numeric_features=FEATURE_COLUMNS, categorical_features=[]
    )
    X, cleaned = pre.fit_transform(df)

    from src.clustering import run_kmeans

    result = run_kmeans(X, n_clusters=5)
    from src.personas import assign_personas_from_data

    num_pipe = pre.transformer_.named_transformers_["num"]
    scaler = num_pipe.named_steps["scaler"]
    centers_orig = pd.DataFrame(
        scaler.inverse_transform(result.cluster_centers),
        columns=FEATURE_COLUMNS,
    )
    personas = assign_personas_from_data(centers_orig, cleaned)

    return {
        "pre": pre,
        "scaled_centers": result.cluster_centers,
        "centers_orig": centers_orig,
        "personas": personas,
        "algorithm": "kmeans",
        "config": {"algorithm": "kmeans", "n_clusters": 5},
    }


class TestPredictNew:
    def test_returns_dict_with_cluster_and_persona(self, prediction_bundle):
        pred = predict_new(prediction_bundle, age=30, genre="Male", income=60.0, spending=50.0)
        assert isinstance(pred, dict)
        assert "cluster_id" in pred
        assert "persona" in pred
        assert "distance" in pred
        assert pred["persona"] is not None

    def test_cluster_id_is_valid(self, prediction_bundle):
        pred = predict_new(prediction_bundle, age=30, genre="Female", income=80.0, spending=80.0)
        assert pred["cluster_id"] in range(5)

    def test_distance_is_non_negative(self, prediction_bundle):
        pred = predict_new(prediction_bundle, age=25, genre="Male", income=40.0, spending=40.0)
        assert pred["distance"] >= 0.0

    def test_none_for_empty_bundle(self):
        assert predict_new(None, age=30, genre="Male", income=60.0, spending=50.0) is None

    def test_none_for_empty_personas(self, prediction_bundle):
        bundle = dict(prediction_bundle)
        bundle["personas"] = {}
        assert predict_new(bundle, age=30, genre="Male", income=60.0, spending=50.0) is None

    def test_none_for_none_centers(self, prediction_bundle):
        bundle = dict(prediction_bundle)
        bundle["scaled_centers"] = None
        assert predict_new(bundle, age=30, genre="Male", income=60.0, spending=50.0) is None

    def test_none_for_empty_centers(self, prediction_bundle):
        bundle = dict(prediction_bundle)
        bundle["scaled_centers"] = np.empty((0, 2))
        assert predict_new(bundle, age=30, genre="Male", income=60.0, spending=50.0) is None

    def test_none_for_all_nan_centers(self, prediction_bundle):
        bundle = dict(prediction_bundle)
        bundle["scaled_centers"] = np.full((5, 2), np.nan)
        assert predict_new(bundle, age=30, genre="Male", income=60.0, spending=50.0) is None

    def test_deterministic_for_same_input(self, prediction_bundle):
        pred1 = predict_new(prediction_bundle, age=35, genre="Male", income=70.0, spending=60.0)
        pred2 = predict_new(prediction_bundle, age=35, genre="Male", income=70.0, spending=60.0)
        assert pred1["cluster_id"] == pred2["cluster_id"]
        assert pred1["distance"] == pytest.approx(pred2["distance"])

    def test_different_inputs_may_differ(self, prediction_bundle):
        pred1 = predict_new(prediction_bundle, age=20, genre="Male", income=20.0, spending=20.0)
        pred2 = predict_new(prediction_bundle, age=60, genre="Female", income=100.0, spending=90.0)
        assert pred1 is not None and pred2 is not None
        assert pred1["cluster_id"] != pred2["cluster_id"] or pred1["distance"] != pred2["distance"]
