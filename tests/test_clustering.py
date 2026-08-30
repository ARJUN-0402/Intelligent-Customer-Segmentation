"""Tests for src.clustering — all algorithms and edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

from src.clustering import (
    ClusterResult,
    KOptimizationResult,
    build_agglomerative,
    build_dbscan,
    build_gmm,
    build_kmeans,
    evaluate_k_range,
    labels_to_series,
    run_agglomerative,
    run_clustering,
    run_dbscan,
    run_gmm,
    run_kmeans,
)
from src.utils import DEFAULT_N_CLUSTERS


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
class TestKMeans:
    def test_build_kmeans_config(self):
        model = build_kmeans(n_clusters=4)
        assert isinstance(model, KMeans)
        assert model.n_clusters == 4
        assert model.init == "k-means++"

    def test_build_kmeans_accepts_params(self):
        model = build_kmeans(n_clusters=3, random_state=0)
        assert model.random_state == 0

    def test_run_kmeans_returns_result(self, X_scaled):
        X, _ = X_scaled
        result = run_kmeans(X, n_clusters=DEFAULT_N_CLUSTERS)
        assert isinstance(result, ClusterResult)
        assert result.n_clusters == DEFAULT_N_CLUSTERS
        assert result.labels.shape[0] == X.shape[0]
        assert result.cluster_centers.shape == (DEFAULT_N_CLUSTERS, X.shape[1])
        assert len(np.unique(result.labels)) <= DEFAULT_N_CLUSTERS
        assert result.inertia is not None
        assert result.inertia > 0
        assert result.algorithm == "kmeans"
        assert result.valid

    def test_run_kmeans_labels_match_samples(self, X_scaled):
        X, _ = X_scaled
        result = run_kmeans(X, n_clusters=3)
        assert result.labels.shape == (X.shape[0],)
        assert set(result.labels).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# Agglomerative
# ---------------------------------------------------------------------------
class TestAgglomerative:
    def test_build_agglomerative_default(self):
        model = build_agglomerative(n_clusters=3)
        assert isinstance(model, AgglomerativeClustering)
        assert model.n_clusters == 3
        assert model.linkage == "ward"

    def test_run_agglomerative_returns_result(self, X_scaled):
        X, _ = X_scaled
        result = run_agglomerative(X, n_clusters=4)
        assert isinstance(result, ClusterResult)
        assert result.n_clusters == 4
        assert result.labels.shape[0] == X.shape[0]
        assert result.cluster_centers is not None
        assert result.cluster_centers.shape == (4, X.shape[1])
        assert result.algorithm == "agglomerative_ward"
        assert result.valid

    def test_run_agglomerative_all_linkages(self, X_scaled):
        X, _ = X_scaled
        for linkage in ("ward", "complete", "average", "single"):
            result = run_agglomerative(X, n_clusters=3, linkage=linkage)
            assert result.n_clusters == 3
            assert result.algorithm == f"agglomerative_{linkage}"

    def test_run_agglomerative_invalid_linkage(self, X_scaled):
        X, _ = X_scaled
        with pytest.raises(ValueError, match="linkage must be one of"):
            run_agglomerative(X, n_clusters=3, linkage="bogus")

    def test_run_agglomerative_ward_rejects_non_euclidean(self, X_scaled):
        X, _ = X_scaled
        with pytest.raises(ValueError, match="ward linkage only supports"):
            run_agglomerative(X, n_clusters=3, linkage="ward", metric="manhattan")


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------
class TestDBSCAN:
    def test_build_dbscan_default(self):
        model = build_dbscan(eps=0.5, min_samples=5)
        assert isinstance(model, DBSCAN)
        assert model.eps == 0.5
        assert model.min_samples == 5

    def test_run_dbscan_returns_result(self, X_scaled):
        X, _ = X_scaled
        result = run_dbscan(X, eps=5, min_samples=4)
        assert isinstance(result, ClusterResult)
        assert result.labels.shape[0] == X.shape[0]
        assert result.algorithm == "dbscan"
        assert result.noise_points >= 0
        assert result.noise_points == int(np.sum(result.labels == -1))

    def test_run_dbscan_detects_clusters(self, X_scaled):
        X, _ = X_scaled
        result = run_dbscan(X, eps=5, min_samples=4)
        assert result.n_clusters >= 1

    def test_run_dbscan_all_noise(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=(20, 2))
        result = run_dbscan(X, eps=0.01, min_samples=5)
        assert result.n_clusters == 0
        assert result.noise_points == 20
        assert not result.valid
        assert result.cluster_centers is None

    def test_run_dbscan_invalid_eps(self, X_scaled):
        X, _ = X_scaled
        with pytest.raises(ValueError, match="eps must be positive"):
            run_dbscan(X, eps=-1, min_samples=5)

    def test_run_dbscan_invalid_min_samples(self, X_scaled):
        X, _ = X_scaled
        with pytest.raises(ValueError, match="min_samples must be >= 1"):
            run_dbscan(X, eps=0.5, min_samples=0)


# ---------------------------------------------------------------------------
# Gaussian Mixture
# ---------------------------------------------------------------------------
class TestGMM:
    def test_build_gmm_default(self):
        model = build_gmm(n_components=3)
        assert isinstance(model, GaussianMixture)
        assert model.n_components == 3
        assert model.covariance_type == "full"

    def test_run_gmm_returns_result(self, X_scaled):
        X, _ = X_scaled
        result = run_gmm(X, n_components=3)
        assert isinstance(result, ClusterResult)
        assert result.n_clusters == 3
        assert result.labels.shape[0] == X.shape[0]
        assert result.cluster_centers is not None
        assert result.cluster_centers.shape == (3, X.shape[1])
        assert result.bic is not None
        assert result.aic is not None
        assert result.algorithm == "gmm"
        assert result.valid

    def test_run_gmm_all_covariance_types(self, X_scaled):
        X, _ = X_scaled
        for cov in ("full", "tied", "diag", "spherical"):
            result = run_gmm(X, n_components=3, covariance_type=cov)
            assert result.n_clusters == 3

    def test_run_gmm_invalid_covariance(self, X_scaled):
        X, _ = X_scaled
        with pytest.raises(ValueError, match="covariance_type must be one of"):
            run_gmm(X, n_components=3, covariance_type="bogus")


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------
class TestDispatcher:
    def test_run_clustering_kmeans(self, X_scaled):
        X, _ = X_scaled
        result = run_clustering(X, algorithm="kmeans", n_clusters=3)
        assert result.algorithm == "kmeans"
        assert result.n_clusters == 3

    def test_run_clustering_agglomerative(self, X_scaled):
        X, _ = X_scaled
        result = run_clustering(X, algorithm="agglomerative", n_clusters=3)
        assert "agglomerative" in result.algorithm

    def test_run_clustering_dbscan(self, X_scaled):
        X, _ = X_scaled
        result = run_clustering(X, algorithm="dbscan", eps=5, min_samples=4)
        assert result.algorithm == "dbscan"

    def test_run_clustering_gmm(self, X_scaled):
        X, _ = X_scaled
        result = run_clustering(X, algorithm="gmm", n_components=3)
        assert result.algorithm == "gmm"

    def test_run_clustering_unknown_algorithm(self, X_scaled):
        X, _ = X_scaled
        with pytest.raises(ValueError, match="Unknown algorithm"):
            run_clustering(X, algorithm="spectral")

    def test_run_clustering_case_insensitive(self, X_scaled):
        X, _ = X_scaled
        result = run_clustering(X, algorithm="K-Means", n_clusters=3)
        assert result.algorithm == "kmeans"


# ---------------------------------------------------------------------------
# K optimization
# ---------------------------------------------------------------------------
class TestKEvaluation:
    def test_evaluate_k_range_returns_result(self, X_scaled):
        X, _ = X_scaled
        result = evaluate_k_range(X, k_min=2, k_max=6)
        assert isinstance(result, KOptimizationResult)
        assert result.k_values == [2, 3, 4, 5, 6]
        assert len(result.inertias) == 5
        assert len(result.silhouette_scores) == 5
        assert result.optimal_k in result.k_values

    def test_evaluate_k_range_inertias_decrease(self, X_scaled):
        X, _ = X_scaled
        result = evaluate_k_range(X, k_min=2, k_max=8)
        for i in range(1, len(result.inertias)):
            assert result.inertias[i] <= result.inertias[i - 1]

    def test_evaluate_k_range_optimal_is_highest_silhouette(self, X_scaled):
        X, _ = X_scaled
        result = evaluate_k_range(X, k_min=2, k_max=8)
        best_idx = np.argmax(result.silhouette_scores)
        assert result.optimal_k == result.k_values[best_idx]

    def test_evaluate_k_range_invalid_k_min(self, X_scaled):
        X, _ = X_scaled
        with pytest.raises(ValueError, match="k_min must be >= 2"):
            evaluate_k_range(X, k_min=1, k_max=5)

    def test_evaluate_k_range_k_max_lt_k_min(self, X_scaled):
        X, _ = X_scaled
        with pytest.raises(ValueError, match="k_max .* must be >= k_min"):
            evaluate_k_range(X, k_min=5, k_max=3)

    def test_evaluate_k_range_k_max_too_large(self, X_scaled):
        X, _ = X_scaled
        with pytest.raises(ValueError, match="k_max .* must be less than n_samples"):
            evaluate_k_range(X, k_min=2, k_max=X.shape[0] + 10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_too_few_samples_kmeans(self):
        X = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="at least 2 samples"):
            run_kmeans(X, n_clusters=2)

    def test_too_few_samples_agglomerative(self):
        X = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="at least 2 samples"):
            run_agglomerative(X, n_clusters=2)

    def test_too_few_samples_dbscan(self):
        X = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="at least 2 samples"):
            run_dbscan(X, eps=0.5, min_samples=5)

    def test_too_few_samples_gmm(self):
        X = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="at least 2 samples"):
            run_gmm(X, n_components=2)

    def test_invalid_k_too_large(self, X_scaled):
        X, _ = X_scaled
        with pytest.raises(ValueError, match="must be less than n_samples"):
            run_kmeans(X, n_clusters=X.shape[0] + 1)

    def test_invalid_k_less_than_two(self, X_scaled):
        X, _ = X_scaled
        with pytest.raises(ValueError, match="n_clusters must be an integer >= 2"):
            run_kmeans(X, n_clusters=1)

    def test_nan_input_raises(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0]])
        with pytest.raises(ValueError, match="NaN"):
            run_kmeans(X, n_clusters=2)

    def test_1d_input_raises(self):
        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2D array"):
            run_kmeans(X, n_clusters=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_labels_to_series(self):
        labels = np.array([0, 1, 0, 1])
        s = labels_to_series(labels)
        assert s.name == "Cluster"
        assert np.array_equal(s.values, labels)

    def test_labels_to_series_with_index(self):
        labels = np.array([0, 1, 2])
        idx = pd.Index(["a", "b", "c"])
        s = labels_to_series(labels, index=idx)
        assert list(s.index) == ["a", "b", "c"]

    def test_cluster_result_unique_labels_with_noise(self):
        labels = np.array([-1, -1, 0, 1, 1])
        result = ClusterResult(
            labels=labels, model=None, n_clusters=2, algorithm="dbscan"
        )
        assert list(result.unique_labels) == [-1, 0, 1]

    def test_cluster_result_valid_false_when_no_clusters(self):
        result = ClusterResult(
            labels=np.array([-1, -1]), model=None, n_clusters=0, algorithm="dbscan"
        )
        assert not result.valid
