"""Tests for src.personas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.personas import (
    PERSONA_CATALOG,
    assign_personas,
    assign_personas_from_data,
    build_persona_profiles,
    classify_persona,
    compute_breakpoints,
)
from src.utils import ANNUAL_INCOME, SPENDING_SCORE


# ---------------------------------------------------------------------------
# Persona catalog
# ---------------------------------------------------------------------------
class TestPersonaCatalog:
    def test_catalog_contains_five_personas(self):
        assert len(PERSONA_CATALOG) == 5

    def test_keys_match_expected(self):
        expected = {"vip", "saver", "impulsive", "budget", "mainstream"}
        assert set(PERSONA_CATALOG.keys()) == expected

    def test_each_persona_has_required_fields(self):
        for persona in PERSONA_CATALOG.values():
            assert persona.key
            assert persona.name
            assert persona.description
            assert persona.strategy
            assert persona.color


# ---------------------------------------------------------------------------
# Breakpoints
# ---------------------------------------------------------------------------
class TestComputeBreakpoints:
    def test_returns_tuple_of_floats(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        low, high = compute_breakpoints(s)
        assert isinstance(low, float)
        assert isinstance(high, float)

    def test_quartile_values(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        low, high = compute_breakpoints(s)
        assert low == 20.0
        assert high == 40.0

    def test_logs_message(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        s = pd.Series([1.0, 2.0, 3.0])
        compute_breakpoints(s)
        assert "Computed breakpoints" in caplog.text


# ---------------------------------------------------------------------------
# classify_persona
# ---------------------------------------------------------------------------
class TestClassifyPersona:
    def test_high_high_returns_vip(self):
        p = classify_persona(income=100.0, spending=90.0, income_low=40.0, income_high=70.0, spending_low=30.0, spending_high=60.0)
        assert p.key == "vip"

    def test_low_low_returns_budget(self):
        p = classify_persona(income=20.0, spending=10.0, income_low=40.0, income_high=70.0, spending_low=30.0, spending_high=60.0)
        assert p.key == "budget"

    def test_high_low_returns_saver(self):
        p = classify_persona(income=100.0, spending=10.0, income_low=40.0, income_high=70.0, spending_low=30.0, spending_high=60.0)
        assert p.key == "saver"

    def test_low_high_returns_impulsive(self):
        p = classify_persona(income=20.0, spending=90.0, income_low=40.0, income_high=70.0, spending_low=30.0, spending_high=60.0)
        assert p.key == "impulsive"

    def test_mid_mid_returns_mainstream(self):
        p = classify_persona(income=55.0, spending=45.0, income_low=40.0, income_high=70.0, spending_low=30.0, spending_high=60.0)
        assert p.key == "mainstream"


# ---------------------------------------------------------------------------
# assign_personas
# ---------------------------------------------------------------------------
class TestAssignPersonas:
    def test_returns_dict_with_all_clusters(self):
        df = pd.DataFrame({
            ANNUAL_INCOME: [55.3, 86.5, 25.7, 88.2, 26.3],
            SPENDING_SCORE: [49.5, 82.1, 79.4, 17.1, 20.9],
        })
        result = assign_personas(df)
        assert set(result.keys()) == {0, 1, 2, 3, 4}

    def test_raises_on_missing_income(self):
        df = pd.DataFrame({SPENDING_SCORE: [49.5, 82.1]})
        with pytest.raises(KeyError, match="Annual Income"):
            assign_personas(df)

    def test_raises_on_missing_spending(self):
        df = pd.DataFrame({ANNUAL_INCOME: [55.3, 86.5]})
        with pytest.raises(KeyError, match="Spending Score"):
            assign_personas(df)

    def test_uses_custom_breakpoints(self):
        df = pd.DataFrame({
            ANNUAL_INCOME: [55.3, 86.5, 25.7, 88.2, 26.3],
            SPENDING_SCORE: [49.5, 82.1, 79.4, 17.1, 20.9],
        })
        bp = {ANNUAL_INCOME: (30.0, 80.0), SPENDING_SCORE: (20.0, 70.0)}
        result = assign_personas(df, breakpoints=bp)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# assign_personas_from_data
# ---------------------------------------------------------------------------
class TestAssignPersonasFromData:
    @pytest.fixture(scope="session")
    def full_df(self):
        from src.data_loader import load_data
        from src.utils import DEFAULT_DATA_PATH
        return load_data(DEFAULT_DATA_PATH)

    def test_returns_all_five_personas(self, full_df):
        centers = pd.DataFrame({
            ANNUAL_INCOME: [55.3, 86.5, 25.7, 88.2, 26.3],
            SPENDING_SCORE: [49.5, 82.1, 79.4, 17.1, 20.9],
        })
        result = assign_personas_from_data(centers, full_df)
        keys = {p.key for p in result.values()}
        assert keys == {"vip", "saver", "impulsive", "budget", "mainstream"}

    def test_raises_on_missing_income_column(self, full_df):
        bad = full_df.drop(columns=[ANNUAL_INCOME])
        centers = pd.DataFrame({SPENDING_SCORE: [1.0, 2.0]})
        with pytest.raises(KeyError, match="Annual Income"):
            assign_personas_from_data(centers, bad)

    def test_raises_on_missing_spending_column(self, full_df):
        bad = full_df.drop(columns=[SPENDING_SCORE])
        centers = pd.DataFrame({ANNUAL_INCOME: [1.0, 2.0]})
        with pytest.raises(KeyError, match="Spending Score"):
            assign_personas_from_data(centers, bad)


# ---------------------------------------------------------------------------
# build_persona_profiles
# ---------------------------------------------------------------------------
class TestBuildPersonaProfiles:
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
        return result, cleaned

    def test_returns_list_of_five_profiles(self, clustering_result):
        result, cleaned = clustering_result
        centers = pd.DataFrame(result.cluster_centers, columns=[ANNUAL_INCOME, SPENDING_SCORE])
        profiles = build_persona_profiles(cleaned, result.labels, cluster_centers=centers)
        assert len(profiles) == 5

    def test_every_cluster_has_exactly_one_persona(self, clustering_result):
        result, cleaned = clustering_result
        centers = pd.DataFrame(result.cluster_centers, columns=[ANNUAL_INCOME, SPENDING_SCORE])
        profiles = build_persona_profiles(cleaned, result.labels, cluster_centers=centers)
        assert len(profiles) == result.n_clusters

    def test_every_profile_has_non_empty_explanation(self, clustering_result):
        result, cleaned = clustering_result
        centers = pd.DataFrame(result.cluster_centers, columns=[ANNUAL_INCOME, SPENDING_SCORE])
        profiles = build_persona_profiles(cleaned, result.labels, cluster_centers=centers)
        for profile in profiles:
            assert profile.profile_summary
            assert len(profile.profile_summary.strip()) > 0

    def test_every_profile_has_non_empty_characteristics(self, clustering_result):
        result, cleaned = clustering_result
        centers = pd.DataFrame(result.cluster_centers, columns=[ANNUAL_INCOME, SPENDING_SCORE])
        profiles = build_persona_profiles(cleaned, result.labels, cluster_centers=centers)
        for profile in profiles:
            assert profile.key_characteristics
            assert len(profile.key_characteristics) > 0

    def test_customer_count_matches_actual(self, clustering_result):
        result, cleaned = clustering_result
        centers = pd.DataFrame(result.cluster_centers, columns=[ANNUAL_INCOME, SPENDING_SCORE])
        profiles = build_persona_profiles(cleaned, result.labels, cluster_centers=centers)
        total = sum(p.customer_count for p in profiles)
        assert total == len(cleaned)

    def test_percentage_sums_to_100(self, clustering_result):
        result, cleaned = clustering_result
        centers = pd.DataFrame(result.cluster_centers, columns=[ANNUAL_INCOME, SPENDING_SCORE])
        profiles = build_persona_profiles(cleaned, result.labels, cluster_centers=centers)
        total_pct = sum(p.percentage for p in profiles)
        assert abs(total_pct - 100.0) < 0.1

    def test_raises_on_length_mismatch(self, clustering_result):
        result, cleaned = clustering_result
        centers = pd.DataFrame(result.cluster_centers, columns=[ANNUAL_INCOME, SPENDING_SCORE])
        bad_labels = np.array([0, 1])
        with pytest.raises(ValueError, match="must match labels length"):
            build_persona_profiles(cleaned, bad_labels, cluster_centers=centers)

    def test_raises_when_no_centers_and_no_personas(self, clustering_result):
        result, cleaned = clustering_result
        with pytest.raises(ValueError, match="cluster_centers"):
            build_persona_profiles(cleaned, result.labels)
