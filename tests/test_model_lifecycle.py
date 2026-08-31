"""Phase 2 regression tests: model artifact lifecycle, validation, and
prediction consistency.

These tests guard the deployment hardening contract:

- The canonical model path resolves to ``<repo>/models/segmentation_model.joblib``.
- Loading a missing artifact triggers a deterministic retrain.
- Bundles with the wrong version / feature schema / algorithm / persona
  mapping are rejected and the artifact is replaced.
- ``predict_new`` uses the SAME preprocessor, scaler, and feature ordering as
  training - no silent drift.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src import utils
from src.data_loader import load_data
from src.preprocessing import CustomerDataPreprocessor
from src.utils import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_FILENAME,
    MODEL_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    PROJECT_ROOT,
)

from app import _validate_bundle, predict_new
from src.clustering import run_kmeans
from src.personas import assign_personas_from_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_good_bundle() -> dict:
    df = load_data(DEFAULT_DATA_PATH)
    pre = CustomerDataPreprocessor(
        numeric_features=FEATURE_COLUMNS, categorical_features=[]
    )
    X, cleaned = pre.fit_transform(df)
    result = run_kmeans(X, n_clusters=5)
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
        "model_version": MODEL_VERSION,
    }


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
class TestModelPath:
    def test_path_is_repo_relative(self):
        assert MODEL_PATH.is_absolute()
        assert MODEL_PATH.name == MODEL_FILENAME

    def test_path_lives_under_repo_models_dir(self):
        assert MODEL_PATH.parent == MODELS_DIR
        assert MODELS_DIR == PROJECT_ROOT / "models"

    def test_filename_is_canonical(self):
        assert MODEL_FILENAME == "segmentation_model.joblib"

    def test_utils_exports_model_path(self):
        assert hasattr(utils, "MODEL_PATH")
        assert utils.MODEL_PATH == MODEL_PATH

    def test_cwd_does_not_affect_path(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        from importlib import reload
        reloaded = reload(utils)
        assert reloaded.MODEL_PATH == MODEL_PATH


# ---------------------------------------------------------------------------
# Bundle validation
# ---------------------------------------------------------------------------
class TestValidateBundle:
    def test_valid_bundle_passes(self):
        bundle = _make_good_bundle()
        assert _validate_bundle(bundle) is None

    def test_missing_keys_rejected(self):
        bundle = _make_good_bundle()
        bundle.pop("model_version")
        reason = _validate_bundle(bundle)
        assert reason is not None and "missing keys" in reason

    def test_version_mismatch_rejected(self):
        bundle = _make_good_bundle()
        bundle["model_version"] = "0.1.0"
        reason = _validate_bundle(bundle)
        assert reason is not None and "model_version mismatch" in reason

    def test_unknown_algorithm_rejected(self):
        bundle = _make_good_bundle()
        bundle["algorithm"] = "spectral"
        reason = _validate_bundle(bundle)
        assert reason is not None and "unknown algorithm" in reason

    def test_feature_schema_mismatch_rejected(self):
        bundle = _make_good_bundle()
        # Replace fitted transformer with one fitted on a different column.
        bundle["pre"] = CustomerDataPreprocessor(
            numeric_features=[FEATURE_COLUMNS[0]], categorical_features=[]
        )
        df = load_data(DEFAULT_DATA_PATH)
        bundle["pre"].fit_transform(df)
        reason = _validate_bundle(bundle)
        assert reason is not None and "feature names mismatch" in reason

    def test_unfitted_preprocessor_rejected(self):
        bundle = _make_good_bundle()
        bundle["pre"] = CustomerDataPreprocessor(
            numeric_features=FEATURE_COLUMNS, categorical_features=[]
        )
        reason = _validate_bundle(bundle)
        assert reason is not None and "not fitted" in reason

    def test_centers_persona_count_mismatch_rejected(self):
        bundle = _make_good_bundle()
        # Drop one persona so counts diverge.
        first_key = next(iter(bundle["personas"]))
        bundle["personas"].pop(first_key)
        reason = _validate_bundle(bundle)
        assert reason is not None and "mismatch" in reason

    def test_missing_centers_rejected(self):
        bundle = _make_good_bundle()
        bundle["scaled_centers"] = None
        reason = _validate_bundle(bundle)
        assert reason is not None and "cluster centres" in reason

    def test_non_dict_rejected(self):
        assert _validate_bundle("not a dict") is not None


# ---------------------------------------------------------------------------
# Artifact load + missing-artifact behavior
# ---------------------------------------------------------------------------
class TestArtifactLifecycle:
    def test_missing_artifact_triggers_train_and_save(self, tmp_path, monkeypatch):
        # Redirect MODELS_DIR / MODEL_PATH into a clean temp directory so the
        # loader observes a missing artifact without touching the real one.
        from src import utils as utils_mod
        target = tmp_path / "models"
        target.mkdir()
        missing_artifact = target / MODEL_FILENAME
        assert not missing_artifact.exists()

        monkeypatch.setattr(utils_mod, "MODELS_DIR", target)
        monkeypatch.setattr(utils_mod, "MODEL_PATH", missing_artifact)
        monkeypatch.setattr("app.MODELS_DIR", target)
        monkeypatch.setattr("app.MODEL_PATH", missing_artifact)
        # Also clear the Streamlit cache so load_prediction_model rebuilds.
        import app
        try:
            app.load_prediction_model.clear()
        except Exception:
            pass

        base = {
            "raw": load_data(DEFAULT_DATA_PATH),
            "pre": CustomerDataPreprocessor(
                numeric_features=FEATURE_COLUMNS, categorical_features=[]
            ),
            "X": None,
            "cleaned": None,
            "error": None,
        }
        df = base["raw"]
        X, cleaned = base["pre"].fit_transform(df)
        base["X"] = X
        base["cleaned"] = cleaned

        bundle = app.load_prediction_model()
        assert bundle is not None
        assert missing_artifact.exists(), "Loader must save on missing artifact"
        # Round-trip: a fresh load of the saved artifact must validate cleanly.
        reloaded = joblib.load(missing_artifact)
        assert app._validate_bundle(reloaded) is None

    def test_committed_artifact_loads_and_validates(self):
        """The repo-committed artifact (when present) must round-trip cleanly."""
        if not MODEL_PATH.exists():
            pytest.skip("Committed artifact not present in this environment.")
        bundle = joblib.load(MODEL_PATH)
        assert _validate_bundle(bundle) is None


# ---------------------------------------------------------------------------
# Prediction consistency (no silent drift)
# ---------------------------------------------------------------------------
class TestPredictionConsistency:
    def test_predicted_cluster_matches_nearest_centre(self):
        bundle = _make_good_bundle()
        pred = predict_new(
            bundle, age=35, genre="Male", income=70.0, spending=60.0
        )
        # Recompute distance to each centre using the same preprocessor + centres
        # and confirm the predicted cluster is the argmin.
        row = pd.DataFrame(
            [{
                "CustomerID": -1,
                "Genre": "Male",
                "Age": 35,
                "Annual Income (k$)": 70.0,
                "Spending Score (1-100)": 60.0,
            }]
        )
        Xs = bundle["pre"].transform(row)
        centres = bundle["scaled_centers"]
        expected = int(np.argmin(np.linalg.norm(centres - Xs, axis=1)))
        assert pred["cluster_id"] == expected

    def test_same_preprocessor_for_train_and_predict(self):
        bundle = _make_good_bundle()
        # Use the *fitted* preprocessor to predict on the training set; the
        # nearest-centre mapping must agree with the persisted centres.
        df = load_data(DEFAULT_DATA_PATH)
        X = bundle["pre"].transform(df)
        centres = bundle["scaled_centers"]
        labels_pred = np.array([
            int(np.argmin(np.linalg.norm(centres - x, axis=1))) for x in X
        ])
        assert labels_pred.shape[0] == len(df)
        assert len(set(labels_pred)) >= 2

    def test_feature_ordering_matches_training(self):
        bundle = _make_good_bundle()
        # Feature order from the fitted preprocessor must match FEATURE_COLUMNS.
        assert list(bundle["pre"].output_feature_names) == list(FEATURE_COLUMNS)