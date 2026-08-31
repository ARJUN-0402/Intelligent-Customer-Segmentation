"""Tests for src.analytics — cluster profiles, segment comparison, feature
separation, cluster stability, and analytical insights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    AnalyticalInsight,
    ClusterProfile,
    ClusterStabilityResult,
    FeatureSeparation,
    SegmentComparison,
    compare_segments_vs_overall,
    comparison_to_dataframe,
    compute_cluster_stability,
    compute_feature_separation,
    generate_analytical_insights,
    insights_to_dataframe,
)
from src.business_insights import build_cluster_profiles
from src.utils import ANNUAL_INCOME, AGE, GENRE, NUMERIC_COLUMNS, SPENDING_SCORE


# ===========================================================================
# Helpers
# ===========================================================================
def _make_df() -> pd.DataFrame:
    """Return a small reproducible DataFrame for testing."""
    return pd.DataFrame({
        ANNUAL_INCOME: [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0],
        AGE: [22.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
        SPENDING_SCORE: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        GENRE: ["Male", "Female"] * 4,
    })


def _make_labels() -> np.ndarray:
    """Two balanced clusters."""
    return np.array([0, 0, 0, 0, 1, 1, 1, 1])


# ===========================================================================
# build_cluster_profiles (from business_insights)
# ===========================================================================
class TestComputeClusterProfiles:
    def test_returns_dataframe(self):
        df = _make_df()
        labels = _make_labels()
        result = build_cluster_profiles(df, labels)
        assert isinstance(result, pd.DataFrame)

    def test_has_count_and_percentage(self):
        df = _make_df()
        labels = _make_labels()
        result = build_cluster_profiles(df, labels)
        assert "count" in result.columns
        assert "percentage" in result.columns

    def test_count_values(self):
        df = _make_df()
        labels = _make_labels()
        result = build_cluster_profiles(df, labels)
        assert result.loc[0, "count"] == 4
        assert result.loc[1, "count"] == 4

    def test_percentage_sums_to_100(self):
        df = _make_df()
        labels = _make_labels()
        result = build_cluster_profiles(df, labels)
        total_pct = result["percentage"].sum()
        assert abs(total_pct - 100.0) < 0.01

    def test_has_mean_median_min_max_std_per_feature(self):
        df = _make_df()
        labels = _make_labels()
        result = build_cluster_profiles(df, labels)
        for feat in NUMERIC_COLUMNS:
            assert f"{feat}_mean" in result.columns
            assert f"{feat}_median" in result.columns
            assert f"{feat}_min" in result.columns
            assert f"{feat}_max" in result.columns
            assert f"{feat}_std" in result.columns

    def test_mean_correctness(self):
        df = _make_df()
        labels = _make_labels()
        result = build_cluster_profiles(df, labels)
        cluster0_income = df[labels == 0][ANNUAL_INCOME]
        assert abs(result.loc[0, f"{ANNUAL_INCOME}_mean"] - cluster0_income.mean()) < 1e-6

    def test_median_correctness(self):
        df = _make_df()
        labels = _make_labels()
        result = build_cluster_profiles(df, labels)
        cluster0_spending = df[labels == 0][SPENDING_SCORE]
        assert abs(result.loc[0, f"{SPENDING_SCORE}_median"] - cluster0_spending.median()) < 1e-6

    def test_min_max_correctness(self):
        df = _make_df()
        labels = _make_labels()
        result = build_cluster_profiles(df, labels)
        cluster1_age = df[labels == 1][AGE]
        assert result.loc[1, f"{AGE}_min"] == cluster1_age.min()
        assert result.loc[1, f"{AGE}_max"] == cluster1_age.max()

    def test_std_non_negative(self):
        df = _make_df()
        labels = _make_labels()
        result = build_cluster_profiles(df, labels)
        for feat in NUMERIC_COLUMNS:
            col = f"{feat}_std"
            for val in result[col]:
                assert pd.isna(val) or val >= 0.0

    def test_three_clusters(self):
        df = _make_df()
        labels = np.array([0, 0, 1, 1, 1, 2, 2, 2])
        result = build_cluster_profiles(df, labels)
        assert len(result) == 3

    def test_length_mismatch_raises(self):
        df = _make_df()
        labels = np.array([0, 1])
        with pytest.raises(ValueError, match="must match labels length"):
            build_cluster_profiles(df, labels)

    def test_has_top_genre_when_genre_present(self):
        df = _make_df()
        labels = _make_labels()
        result = build_cluster_profiles(df, labels)
        assert "top_genre" in result.columns

    def test_profile_dataclass_creation(self):
        df = _make_df()
        labels = _make_labels()
        result = build_cluster_profiles(df, labels)
        for cid in result.index:
            row = result.loc[cid]
            features = {
                f: {
                    "mean": float(row[f"{f}_mean"]),
                    "median": float(row[f"{f}_median"]),
                    "min": float(row[f"{f}_min"]),
                    "max": float(row[f"{f}_max"]),
                    "std": float(row[f"{f}_std"]) if not pd.isna(row[f"{f}_std"]) else 0.0,
                }
                for f in NUMERIC_COLUMNS
            }
            p = ClusterProfile(
                cluster_id=int(cid),
                count=int(row["count"]),
                percentage=float(row["percentage"]),
                features=features,
            )
            assert p.count > 0
            assert abs(p.percentage - 100.0 * p.count / len(df)) < 0.01


# ===========================================================================
# compare_segments_vs_overall
# ===========================================================================
class TestCompareSegmentsVsOverall:
    def test_returns_list(self):
        df = _make_df()
        labels = _make_labels()
        result = compare_segments_vs_overall(df, labels)
        assert isinstance(result, list)

    def test_one_comparison_per_cluster(self):
        df = _make_df()
        labels = _make_labels()
        result = compare_segments_vs_overall(df, labels)
        assert len(result) == 2

    def test_segment_comparison_type(self):
        df = _make_df()
        labels = _make_labels()
        result = compare_segments_vs_overall(df, labels)
        assert all(isinstance(c, SegmentComparison) for c in result)

    def test_feature_deltas_present(self):
        df = _make_df()
        labels = _make_labels()
        result = compare_segments_vs_overall(df, labels)
        for c in result:
            assert ANNUAL_INCOME in c.feature_deltas
            assert SPENDING_SCORE in c.feature_deltas

    def test_interpretation_non_empty(self):
        df = _make_df()
        labels = _make_labels()
        result = compare_segments_vs_overall(df, labels)
        for c in result:
            assert len(c.interpretation) > 0

    def test_length_mismatch_raises(self):
        df = _make_df()
        labels = np.array([0, 1])
        with pytest.raises(ValueError, match="must match labels length"):
            compare_segments_vs_overall(df, labels)

    def test_delta_sign_convention(self):
        """Cluster with higher-than-average income should have positive delta."""
        df = _make_df()
        labels = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        result = compare_segments_vs_overall(df, labels)
        cluster0 = next(c for c in result if c.cluster_id == 0)
        cluster1 = next(c for c in result if c.cluster_id == 1)
        overall_income = df[ANNUAL_INCOME].mean()
        overall_spending = df[SPENDING_SCORE].mean()
        # Cluster 0 (6 of 8) should be below average on both
        assert cluster0.feature_deltas[ANNUAL_INCOME] < 0  # below avg
        assert cluster1.feature_deltas[ANNUAL_INCOME] > 0  # above avg

    def test_comparison_to_dataframe(self):
        df = _make_df()
        labels = _make_labels()
        comparisons = compare_segments_vs_overall(df, labels)
        df_out = comparison_to_dataframe(comparisons)
        assert isinstance(df_out, pd.DataFrame)
        assert "Cluster" in df_out.columns
        assert len(df_out) == 2


# ===========================================================================
# compute_feature_separation
# ===========================================================================
class TestComputeFeatureSeparation:
    def test_returns_list(self):
        df = _make_df()
        labels = _make_labels()
        result = compute_feature_separation(df, labels)
        assert isinstance(result, list)

    def test_one_entry_per_feature(self):
        df = _make_df()
        labels = _make_labels()
        result = compute_feature_separation(df, labels)
        assert len(result) == len(NUMERIC_COLUMNS)

    def test_feature_separation_type(self):
        df = _make_df()
        labels = _make_labels()
        result = compute_feature_separation(df, labels)
        assert all(isinstance(r, FeatureSeparation) for r in result)

    def test_f_ratio_positive(self):
        df = _make_df()
        labels = _make_labels()
        result = compute_feature_separation(df, labels)
        for r in result:
            assert r.f_ratio > 0.0 or np.isinf(r.f_ratio)

    def test_ranks_assigned(self):
        df = _make_df()
        labels = _make_labels()
        result = compute_feature_separation(df, labels)
        ranks = [r.rank for r in result]
        assert ranks == sorted(ranks)
        assert ranks == list(range(1, len(result) + 1))

    def test_most_differentiating_rank_one(self):
        df = _make_df()
        labels = _make_labels()
        result = compute_feature_separation(df, labels)
        best = max(result, key=lambda r: r.f_ratio)
        assert best.rank == 1

    def test_single_cluster_returns_empty(self):
        df = _make_df()
        labels = np.zeros(8, dtype=int)
        result = compute_feature_separation(df, labels)
        assert result == []

    def test_length_mismatch_raises(self):
        df = _make_df()
        labels = np.array([0, 1])
        with pytest.raises(ValueError, match="must match labels length"):
            compute_feature_separation(df, labels)

    def test_custom_feature_list(self):
        df = _make_df()
        labels = _make_labels()
        result = compute_feature_separation(df, labels, numeric_features=[AGE, ANNUAL_INCOME])
        assert len(result) == 2
        features = {r.feature for r in result}
        assert features == {AGE, ANNUAL_INCOME}


# ===========================================================================
# compute_cluster_stability
# ===========================================================================
class TestComputeClusterStability:
    @pytest.fixture(scope="session")
    def X_small(self):
        rng = np.random.default_rng(42)
        return rng.normal(0, 1, size=(80, 3))

    def test_returns_stability_result(self, X_small):
        result = compute_cluster_stability(X_small, n_clusters=3, n_runs=5)
        assert isinstance(result, ClusterStabilityResult)

    def test_correct_n_runs(self, X_small):
        result = compute_cluster_stability(X_small, n_clusters=3, n_runs=5)
        assert result.n_runs == 5

    def test_n_clusters_set(self, X_small):
        result = compute_cluster_stability(X_small, n_clusters=4, n_runs=5)
        assert result.n_clusters == 4

    def test_ari_between_0_and_1(self, X_small):
        result = compute_cluster_stability(X_small, n_clusters=3, n_runs=5)
        assert 0.0 <= result.mean_ari <= 1.0

    def test_min_le_mean_le_max(self, X_small):
        result = compute_cluster_stability(X_small, n_clusters=3, n_runs=5)
        assert result.min_ari <= result.mean_ari <= result.max_ari

    def test_std_non_negative(self, X_small):
        result = compute_cluster_stability(X_small, n_clusters=3, n_runs=5)
        assert result.std_ari >= 0.0

    def test_interpretation_non_empty(self, X_small):
        result = compute_cluster_stability(X_small, n_clusters=3, n_runs=5)
        assert len(result.interpretation) > 0

    def test_high_ari_gives_stable_interpretation(self):
        """With enough separation, ARI should be high."""
        rng = np.random.default_rng(42)
        centers = rng.normal(0, 5, size=(3, 4))
        X = np.vstack([rng.normal(c, 0.3, size=(50, 4)) for c in centers])
        result = compute_cluster_stability(X, n_clusters=3, n_runs=5)
        assert result.mean_ari > 0.9

    def test_custom_seeds(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(80, 3))
        seeds = [0, 1, 2, 3, 4]
        result = compute_cluster_stability(X, n_clusters=3, n_runs=5, random_seeds=seeds)
        assert result.n_runs == 5

    def test_seed_length_mismatch_raises(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(80, 3))
        with pytest.raises(ValueError, match="random_seeds length"):
            compute_cluster_stability(X, n_clusters=3, n_runs=5, random_seeds=[0, 1])

    def test_algorithm_is_kmeans(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(80, 3))
        result = compute_cluster_stability(X, n_clusters=3, n_runs=3)
        assert result.algorithm == "kmeans"


# ===========================================================================
# generate_analytical_insights
# ===========================================================================
class TestGenerateAnalyticalInsights:
    def test_returns_list(self):
        df = _make_df()
        labels = _make_labels()
        profiles_df = build_cluster_profiles(df, labels)
        centers = pd.DataFrame({
            ANNUAL_INCOME: [30.0, 60.0],
            SPENDING_SCORE: [20.0, 55.0],
        }, index=[0, 1])
        result = generate_analytical_insights(profiles_df, centers, labels, df)
        assert isinstance(result, list)

    def test_empty_inputs_returns_empty(self):
        result = generate_analytical_insights(
            pd.DataFrame(), pd.DataFrame(), np.array([]), _make_df()
        )
        assert result == []

    def test_insights_are_analytical_insight_type(self):
        df = _make_df()
        labels = _make_labels()
        profiles_df = build_cluster_profiles(df, labels)
        centers = pd.DataFrame({
            ANNUAL_INCOME: [30.0, 60.0],
            SPENDING_SCORE: [20.0, 55.0],
        }, index=[0, 1])
        result = generate_analytical_insights(profiles_df, centers, labels, df)
        assert all(isinstance(i, AnalyticalInsight) for i in result)

    def test_largest_segment_insight(self):
        df = _make_df()
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        profiles_df = build_cluster_profiles(df, labels)
        centers = pd.DataFrame({
            ANNUAL_INCOME: [30.0, 60.0],
            SPENDING_SCORE: [20.0, 55.0],
        }, index=[0, 1])
        result = generate_analytical_insights(profiles_df, centers, labels, df)
        titles = [i.title for i in result]
        assert "Largest Segment" in titles

    def test_highest_spending_insight(self):
        df = _make_df()
        labels = _make_labels()
        profiles_df = build_cluster_profiles(df, labels)
        centers = pd.DataFrame({
            ANNUAL_INCOME: [30.0, 60.0],
            SPENDING_SCORE: [20.0, 55.0],
        }, index=[0, 1])
        result = generate_analytical_insights(profiles_df, centers, labels, df)
        titles = [i.title for i in result]
        assert "Highest-Spending Segment" in titles

    def test_highest_income_insight(self):
        df = _make_df()
        labels = _make_labels()
        profiles_df = build_cluster_profiles(df, labels)
        centers = pd.DataFrame({
            ANNUAL_INCOME: [30.0, 60.0],
            SPENDING_SCORE: [20.0, 55.0],
        }, index=[0, 1])
        result = generate_analytical_insights(profiles_df, centers, labels, df)
        titles = [i.title for i in result]
        assert "Highest-Income Segment" in titles

    def test_income_spending_relationship_insight(self):
        df = _make_df()
        labels = _make_labels()
        profiles_df = build_cluster_profiles(df, labels)
        centers = pd.DataFrame({
            ANNUAL_INCOME: [30.0, 60.0],
            SPENDING_SCORE: [20.0, 55.0],
        }, index=[0, 1])
        result = generate_analytical_insights(profiles_df, centers, labels, df)
        titles = [i.title for i in result]
        assert "Income vs Spending Relationship" in titles

    def test_most_distinctive_insight(self):
        df = _make_df()
        labels = _make_labels()
        profiles_df = build_cluster_profiles(df, labels)
        centers = pd.DataFrame({
            ANNUAL_INCOME: [30.0, 60.0],
            SPENDING_SCORE: [20.0, 55.0],
        }, index=[0, 1])
        result = generate_analytical_insights(profiles_df, centers, labels, df)
        titles = [i.title for i in result]
        assert "Most Distinctive Segment" in titles

    def test_insight_segment_ids_are_valid(self):
        df = _make_df()
        labels = _make_labels()
        profiles_df = build_cluster_profiles(df, labels)
        centers = pd.DataFrame({
            ANNUAL_INCOME: [30.0, 60.0],
            SPENDING_SCORE: [20.0, 55.0],
        }, index=[0, 1])
        result = generate_analytical_insights(profiles_df, centers, labels, df)
        for i in result:
            if i.segment_id is not None:
                assert i.segment_id in labels

    def test_insights_to_dataframe(self):
        insights = [
            AnalyticalInsight(title="T1", detail="D1", segment_id=0),
            AnalyticalInsight(title="T2", detail="D2"),
        ]
        df_out = insights_to_dataframe(insights)
        assert isinstance(df_out, pd.DataFrame)
        assert len(df_out) == 2
        assert "Insight" in df_out.columns
        assert "Detail" in df_out.columns
        assert "Segment" in df_out.columns


# ===========================================================================
# End-to-end with actual clustering
# ===========================================================================
class TestAnalyticsEndToEnd:
    """Run the full pipeline on real data and verify analytics output."""

    @pytest.fixture(scope="session")
    def full_df(self):
        from src.data_loader import load_data
        from src.utils import DEFAULT_DATA_PATH
        return load_data(DEFAULT_DATA_PATH)

    @pytest.fixture(scope="session")
    def clustering_result(self, full_df):
        from src.preprocessing import CustomerDataPreprocessor
        from src.clustering import run_kmeans
        from src.utils import FEATURE_COLUMNS
        pre = CustomerDataPreprocessor(numeric_features=FEATURE_COLUMNS, categorical_features=[])
        X, cleaned = pre.fit_transform(full_df)
        result = run_kmeans(X, n_clusters=5)
        return result, cleaned, pre

    def test_profiles_from_real_clustering(self, clustering_result):
        result, cleaned, pre = clustering_result
        profiles_df = build_cluster_profiles(cleaned, result.labels)
        assert len(profiles_df) == 5
        assert "count" in profiles_df.columns
        assert "percentage" in profiles_df.columns

    def test_segment_comparison_from_real_clustering(self, clustering_result):
        result, cleaned, pre = clustering_result
        comparisons = compare_segments_vs_overall(cleaned, result.labels)
        assert len(comparisons) == 5

    def test_feature_separation_from_real_clustering(self, clustering_result):
        result, cleaned, pre = clustering_result
        separation = compute_feature_separation(cleaned, result.labels)
        assert len(separation) == 3

    def test_stability_from_real_data(self, full_df):
        from src.preprocessing import CustomerDataPreprocessor
        from src.utils import FEATURE_COLUMNS
        pre = CustomerDataPreprocessor(numeric_features=FEATURE_COLUMNS, categorical_features=[])
        X, _ = pre.fit_transform(full_df)
        stab = compute_cluster_stability(X, n_clusters=5, n_runs=5)
        assert stab.n_clusters == 5
        assert 0.0 <= stab.mean_ari <= 1.0

    def test_insights_from_real_clustering(self, clustering_result):
        result, cleaned, pre = clustering_result
        profiles_df = build_cluster_profiles(cleaned, result.labels)
        num_pipe = pre.transformer_.named_transformers_["num"]
        scaler = num_pipe.named_steps["scaler"]
        orig_centers = pd.DataFrame(
            scaler.inverse_transform(result.cluster_centers),
            columns=[ANNUAL_INCOME, SPENDING_SCORE],
        )
        insights = generate_analytical_insights(
            profiles_df, orig_centers, result.labels, cleaned
        )
        titles = [i.title for i in insights]
        assert "Largest Segment" in titles
        assert "Highest-Spending Segment" in titles
        assert "Highest-Income Segment" in titles
        assert "Income vs Spending Relationship" in titles
