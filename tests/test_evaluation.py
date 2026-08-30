"""Tests for src.evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clustering import ClusterResult, run_kmeans, run_dbscan
from src.evaluation import (
    ComparisonTable,
    MetricResult,
    ModelEvaluationResult,
    compare_models,
    compute_elbow_curve,
    compute_silhouette_scores,
    evaluate_model,
    find_optimal_k,
    run_full_evaluation,
    select_best_clustering_model,
    save_evaluation_results,
)
from src.utils import MAX_K, MIN_K


# ---------------------------------------------------------------------------
# Legacy Phase 3 tests
# ---------------------------------------------------------------------------
class TestLegacyEvaluation:
    def test_compute_elbow_curve(self, X_scaled):
        X, _ = X_scaled
        result = compute_elbow_curve(X)
        assert len(result.k_values) == len(result.inertias)
        assert result.k_values[0] == MIN_K
        assert result.k_values[-1] == MAX_K
        assert all(i > 0 for i in result.inertias)

    def test_compute_silhouette_scores(self, X_scaled):
        X, _ = X_scaled
        result = compute_silhouette_scores(X)
        assert len(result.scores) == len(result.k_values)
        assert MIN_K <= result.optimal_k <= MAX_K
        best = max(result.scores)
        assert result.scores[result.k_values.index(result.optimal_k)] == pytest.approx(best)

    def test_find_optimal_k_returns_highest_silhouette(self, X_scaled):
        X, _ = X_scaled
        sil = compute_silhouette_scores(X)
        k = find_optimal_k(sil)
        assert k == sil.optimal_k

    def test_find_optimal_k_empty_raises(self):
        with pytest.raises(ValueError, match="SilhouetteResult contains no scores"):
            find_optimal_k(
                type("SilhouetteResult", (), {"scores": [], "k_values": []})()
            )

    def test_optimal_k_is_five_for_mall_dataset(self, X_scaled):
        X, _ = X_scaled
        sil = compute_silhouette_scores(X)
        assert sil.optimal_k == 5


# ---------------------------------------------------------------------------
# MetricResult dataclass
# ---------------------------------------------------------------------------
class TestMetricResult:
    def test_available_metric(self):
        m = MetricResult(name="Silhouette", value=0.7, available=True)
        assert m.value == 0.7
        assert m.available is True

    def test_unavailable_metric(self):
        m = MetricResult(name="Inertia", value=None, available=False, note="K-Means only")
        assert m.value is None
        assert m.available is False
        assert m.note == "K-Means only"


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------
class TestEvaluateModel:
    def test_kmeans_metrics(self, X_scaled):
        X, _ = X_scaled
        res = run_kmeans(X, n_clusters=5)
        ev = evaluate_model(X, res)
        assert isinstance(ev, ModelEvaluationResult)
        assert ev.algorithm == "kmeans"
        assert ev.n_clusters == 5

        sil = ev.metrics["Silhouette Score"]
        assert sil.available is True
        assert sil.value is not None
        assert 0.0 <= sil.value <= 1.0

        ch = ev.metrics["Calinski-Harabasz Score"]
        assert ch.available is True
        assert ch.value is not None
        assert ch.value > 0

        db = ev.metrics["Davies-Bouldin Score"]
        assert db.available is True
        assert db.value is not None
        assert db.value > 0

        inertia = ev.metrics["Inertia"]
        assert inertia.available is True
        assert inertia.value is not None
        assert inertia.value > 0

    def test_agglomerative_no_inertia(self, X_scaled):
        from src.clustering import run_agglomerative

        X, _ = X_scaled
        res = run_agglomerative(X, n_clusters=4)
        ev = evaluate_model(X, res)
        assert ev.metrics["Inertia"].available is False
        assert ev.metrics["Inertia"].note == "Only available for K-Means."

    def test_dbscan_single_cluster_metrics_unavailable(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 2))
        res = run_dbscan(X, eps=0.1, min_samples=2)
        ev = evaluate_model(X, res)
        assert ev.n_clusters == 1
        assert ev.metrics["Silhouette Score"].available is False
        assert ev.metrics["Calinski-Harabasz Score"].available is False
        assert ev.metrics["Davies-Bouldin Score"].available is False

    def test_dbscan_all_noise(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=(20, 2))
        res = run_dbscan(X, eps=0.01, min_samples=5)
        ev = evaluate_model(X, res)
        assert ev.n_clusters == 0
        assert not ev.metrics["Silhouette Score"].available
        assert not ev.metrics["Calinski-Harabasz Score"].available
        assert not ev.metrics["Davies-Bouldin Score"].available


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------
class TestCompareModels:
    def test_returns_structured_rows(self, X_scaled):
        X, _ = X_scaled
        results = [
            run_kmeans(X, n_clusters=3),
            run_kmeans(X, n_clusters=5),
        ]
        table = compare_models(X, results)
        assert isinstance(table, ComparisonTable)
        assert len(table.rows) == 2
        assert table.rows[0]["Algorithm"] == "kmeans"
        assert table.rows[0]["Clusters"] == 3
        assert table.rows[1]["Clusters"] == 5

    def test_silhouette_values_present(self, X_scaled):
        X, _ = X_scaled
        results = [run_kmeans(X, n_clusters=3)]
        table = compare_models(X, results)
        row = table.rows[0]
        assert row["Silhouette Score"] is not None
        assert row["Calinski-Harabasz Score"] is not None
        assert row["Davies-Bouldin Score"] is not None
        assert row["Inertia"] is not None

    def test_mixed_algorithms(self, X_scaled):
        from src.clustering import run_agglomerative

        X, _ = X_scaled
        results = [
            run_kmeans(X, n_clusters=4),
            run_agglomerative(X, n_clusters=4),
        ]
        table = compare_models(X, results)
        assert len(table.rows) == 2
        algos = {r["Algorithm"] for r in table.rows}
        assert "kmeans" in algos
        assert "agglomerative_ward" in algos

    def test_empty_results_returns_empty_table(self, X_scaled):
        X, _ = X_scaled
        table = compare_models(X, [])
        assert table.rows == []
        assert table.recommendation is None
        assert table.ranking == []

    def test_dbscan_skipped_if_invalid(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=(20, 2))
        results = [run_dbscan(X, eps=0.01, min_samples=5)]
        table = compare_models(X, results)
        assert len(table.rows) == 0


# ---------------------------------------------------------------------------
# select_best_clustering_model
# ---------------------------------------------------------------------------
class TestSelectBestClusteringModel:
    def test_returns_best_and_table(self, X_scaled):
        X, _ = X_scaled
        candidates = [
            run_kmeans(X, n_clusters=3),
            run_kmeans(X, n_clusters=5),
        ]
        best, table = select_best_clustering_model(X, candidates)
        assert isinstance(best, ModelEvaluationResult)
        assert isinstance(table, ComparisonTable)
        assert table.recommendation == best.algorithm
        assert best.algorithm == table.ranking[0][0]

    def test_empty_candidates_raises(self, X_scaled):
        X, _ = X_scaled
        with pytest.raises(ValueError, match="empty candidate list"):
            select_best_clustering_model(X, [])

    def test_recommendation_in_ranking(self, X_scaled):
        X, _ = X_scaled
        candidates = [
            run_kmeans(X, n_clusters=2),
            run_kmeans(X, n_clusters=5),
        ]
        _, table = select_best_clustering_model(X, candidates)
        assert table.recommendation is not None
        assert table.recommendation == table.ranking[0][0]

    def test_all_invalid_candidates_raises(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=(20, 2))
        candidates = [run_dbscan(X, eps=0.01, min_samples=5)]
        with pytest.raises(RuntimeError, match="No valid clustering metrics"):
            select_best_clustering_model(X, candidates)


# ---------------------------------------------------------------------------
# run_full_evaluation
# ---------------------------------------------------------------------------
class TestRunFullEvaluation:
    def test_runs_default_suite(self, X_scaled):
        X, _ = X_scaled
        table = run_full_evaluation(X)
        assert isinstance(table, ComparisonTable)
        assert len(table.rows) > 0
        algorithms = {r["Algorithm"] for r in table.rows}
        assert any("kmeans" in a for a in algorithms)
        assert table.recommendation is not None

    def test_saves_results(self, X_scaled, tmp_path):
        from src.evaluation import save_evaluation_results

        X, _ = X_scaled
        table = run_full_evaluation(X)
        out = save_evaluation_results(table, path=tmp_path / "eval.csv")
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == len(table.rows)
        assert "Algorithm" in df.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_inertia_none_for_non_kmeans(self):
        res = ClusterResult(
            labels=np.array([0, 1, 0, 1]),
            model=None,
            n_clusters=2,
            algorithm="agglomerative",
        )
        assert res.inertia is None

    def test_evaluate_model_labels_copied(self, X_scaled):
        X, _ = X_scaled
        res = run_kmeans(X, n_clusters=3)
        ev = evaluate_model(X, res)
        assert ev.labels is not None
        np.testing.assert_array_equal(ev.labels, res.labels)
        ev.labels[0] = 99
        assert res.labels[0] != 99

    def test_compare_models_single_candidate(self, X_scaled):
        X, _ = X_scaled
        table = compare_models(X, [run_kmeans(X, n_clusters=3)])
        assert len(table.rows) == 1
        assert table.recommendation == table.rows[0]["Algorithm"]
