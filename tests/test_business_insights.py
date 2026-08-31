"""Tests for src.business_insights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.business_insights import (
    PersonaInsight,
    build_cluster_profiles,
    generate_cluster_insights,
    generate_persona_insights,
    generate_report,
)
from src.personas import assign_personas_from_data, build_persona_profiles
from src.utils import ANNUAL_INCOME, AGE, GENRE, SPENDING_SCORE


# ---------------------------------------------------------------------------
# PersonaInsight dataclass
# ---------------------------------------------------------------------------
class TestPersonaInsight:
    def test_default_retention_is_empty_string(self):
        insight = PersonaInsight(
            persona_name="Test",
            business_interpretation="desc",
            potential_opportunity="opp",
            recommended_marketing_strategy="strat",
        )
        assert insight.retention_engagement_recommendation == ""

    def test_frozen_dataclass(self):
        insight = PersonaInsight(
            persona_name="Test",
            business_interpretation="desc",
            potential_opportunity="opp",
            recommended_marketing_strategy="strat",
        )
        with pytest.raises(AttributeError):
            insight.persona_name = "Other"


# ---------------------------------------------------------------------------
# build_cluster_profiles
# ---------------------------------------------------------------------------
class TestBuildClusterProfiles:
    def test_returns_dataframe(self):
        df = pd.DataFrame({
            ANNUAL_INCOME: [10.0, 20.0, 30.0, 40.0],
            AGE: [20.0, 30.0, 40.0, 50.0],
            SPENDING_SCORE: [1.0, 2.0, 3.0, 4.0],
            GENRE: ["Male", "Female", "Male", "Female"],
        })
        labels = np.array([0, 0, 1, 1])
        result = build_cluster_profiles(df, labels)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_correct_aggregations(self):
        df = pd.DataFrame({
            ANNUAL_INCOME: [10.0, 20.0, 30.0, 40.0],
            AGE: [20.0, 30.0, 40.0, 50.0],
            SPENDING_SCORE: [1.0, 2.0, 3.0, 4.0],
            GENRE: ["Male", "Female", "Male", "Female"],
        })
        labels = np.array([0, 0, 1, 1])
        result = build_cluster_profiles(df, labels)
        assert result.loc[0, "count"] == 2
        assert result.loc[0, f"{AGE}_mean"] == 25.0
        assert result.loc[1, f"{ANNUAL_INCOME}_mean"] == 35.0
        assert result.loc[0, f"{AGE}_median"] == 25.0
        assert result.loc[1, f"{SPENDING_SCORE}_min"] == 3.0

    def test_contains_top_genre(self):
        df = pd.DataFrame({
            ANNUAL_INCOME: [10.0, 20.0, 30.0, 40.0],
            AGE: [20.0, 30.0, 40.0, 50.0],
            SPENDING_SCORE: [1.0, 2.0, 3.0, 4.0],
            GENRE: ["Male", "Female", "Male", "Female"],
        })
        labels = np.array([0, 0, 1, 1])
        result = build_cluster_profiles(df, labels)
        assert "top_genre" in result.columns


# ---------------------------------------------------------------------------
# generate_cluster_insights
# ---------------------------------------------------------------------------
class TestGenerateClusterInsights:
    def test_returns_list_of_strings(self):
        profiles = pd.DataFrame({
            "count": [10],
            "percentage": [50.0],
            f"{AGE}_mean": [30.0],
            f"{ANNUAL_INCOME}_mean": [50.0],
            f"{SPENDING_SCORE}_mean": [40.0],
            "top_genre": ["Female"],
        }, index=[0])
        from src.personas import PERSONA_CATALOG
        personas = {0: PERSONA_CATALOG["mainstream"]}
        insights = generate_cluster_insights(profiles, personas)
        assert isinstance(insights, list)
        assert len(insights) == 1
        assert isinstance(insights[0], str)

    def test_includes_strategy(self):
        profiles = pd.DataFrame({
            "count": [10],
            "percentage": [50.0],
            f"{AGE}_mean": [30.0],
            f"{ANNUAL_INCOME}_mean": [50.0],
            f"{SPENDING_SCORE}_mean": [40.0],
            "top_genre": ["Female"],
        }, index=[0])
        from src.personas import PERSONA_CATALOG
        personas = {0: PERSONA_CATALOG["vip"]}
        insights = generate_cluster_insights(profiles, personas)
        assert "Strategy:" in insights[0]


# ---------------------------------------------------------------------------
# generate_persona_insights
# ---------------------------------------------------------------------------
class TestGeneratePersonaInsights:
    def test_returns_list_of_strings(self):
        from src.personas import PersonaProfile
        profiles = [
            PersonaProfile(
                cluster_id=0,
                persona_name="High-Value Customers",
                customer_count=50,
                percentage=25.0,
                profile_summary="Summary text.",
                key_characteristics=["high income", "high spending"],
            )
        ]
        insights = generate_persona_insights(profiles)
        assert isinstance(insights, list)
        assert len(insights) == 1
        assert isinstance(insights[0], str)

    def test_includes_opportunity(self):
        from src.personas import PersonaProfile
        profiles = [
            PersonaProfile(
                cluster_id=0,
                persona_name="High-Value Customers",
                customer_count=50,
                percentage=25.0,
                profile_summary="Summary text.",
                key_characteristics=["high income", "high spending"],
            )
        ]
        insights = generate_persona_insights(profiles)
        assert "Opportunity:" in insights[0]

    def test_includes_retention(self):
        from src.personas import PersonaProfile
        profiles = [
            PersonaProfile(
                cluster_id=0,
                persona_name="High-Value Customers",
                customer_count=50,
                percentage=25.0,
                profile_summary="Summary text.",
                key_characteristics=["high income", "high spending"],
            )
        ]
        insights = generate_persona_insights(profiles)
        assert "Retention:" in insights[0]

    def test_unknown_persona_gets_default_insights(self):
        from src.personas import PersonaProfile
        profiles = [
            PersonaProfile(
                cluster_id=0,
                persona_name="Unknown Persona",
                customer_count=10,
                percentage=5.0,
                profile_summary="Summary text.",
                key_characteristics=["moderate income"],
            )
        ]
        insights = generate_persona_insights(profiles)
        assert "Target with tailored promotions" in insights[0]

    def test_multiple_profiles_each_get_insights(self):
        from src.personas import PersonaProfile
        profiles = [
            PersonaProfile(
                cluster_id=0,
                persona_name="High-Value Customers",
                customer_count=50,
                percentage=25.0,
                profile_summary="Summary.",
                key_characteristics=["high income"],
            ),
            PersonaProfile(
                cluster_id=1,
                persona_name="Low-Engagement Customers",
                customer_count=30,
                percentage=15.0,
                profile_summary="Summary.",
                key_characteristics=["low income"],
            ),
        ]
        insights = generate_persona_insights(profiles)
        assert len(insights) == 2


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------
class TestGenerateReport:
    def test_returns_string(self):
        profiles = pd.DataFrame({
            "count": [10],
            "percentage": [50.0],
            f"{AGE}_mean": [30.0],
            f"{ANNUAL_INCOME}_mean": [50.0],
            f"{SPENDING_SCORE}_mean": [40.0],
            "top_genre": ["Female"],
        }, index=[0])
        from src.personas import PERSONA_CATALOG
        personas = {0: PERSONA_CATALOG["mainstream"]}
        report = generate_report(profiles, personas, optimal_k=5)
        assert isinstance(report, str)
        assert "Customer Segmentation" in report

    def test_includes_optimal_k(self):
        profiles = pd.DataFrame({
            "count": [10],
            "percentage": [50.0],
            f"{AGE}_mean": [30.0],
            f"{ANNUAL_INCOME}_mean": [50.0],
            f"{SPENDING_SCORE}_mean": [40.0],
            "top_genre": ["Female"],
        }, index=[0])
        from src.personas import PERSONA_CATALOG
        report = generate_report(profiles, {0: PERSONA_CATALOG["mainstream"]}, optimal_k=5)
        assert "5" in report

    def test_includes_silhouette_when_provided(self):
        profiles = pd.DataFrame({
            "count": [10],
            "percentage": [50.0],
            f"{AGE}_mean": [30.0],
            f"{ANNUAL_INCOME}_mean": [50.0],
            f"{SPENDING_SCORE}_mean": [40.0],
            "top_genre": ["Female"],
        }, index=[0])
        from src.personas import PERSONA_CATALOG
        report = generate_report(profiles, {0: PERSONA_CATALOG["mainstream"]}, optimal_k=5, silhouette_scores=[0.5, 0.6, 0.7])
        assert "0.7000" in report


# ---------------------------------------------------------------------------
# End-to-end validation with actual clustering output
# ---------------------------------------------------------------------------
class TestEndToEndValidation:
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

    def test_every_cluster_gets_exactly_one_persona(self, clustering_result):
        result, cleaned, pre = clustering_result
        num_pipe = pre.transformer_.named_transformers_["num"]
        scaler = num_pipe.named_steps["scaler"]
        orig_centers = pd.DataFrame(
            scaler.inverse_transform(result.cluster_centers),
            columns=[ANNUAL_INCOME, SPENDING_SCORE],
        )
        personas = assign_personas_from_data(orig_centers, cleaned)
        assert len(personas) == result.n_clusters

    def test_every_cluster_gets_non_empty_explanation(self, clustering_result):
        result, cleaned, pre = clustering_result
        num_pipe = pre.transformer_.named_transformers_["num"]
        scaler = num_pipe.named_steps["scaler"]
        orig_centers = pd.DataFrame(
            scaler.inverse_transform(result.cluster_centers),
            columns=[ANNUAL_INCOME, SPENDING_SCORE],
        )
        profiles = build_persona_profiles(cleaned, result.labels, cluster_centers=orig_centers)
        for profile in profiles:
            assert profile.profile_summary
            assert len(profile.profile_summary.strip()) > 0

    def test_every_cluster_gets_recommendation(self, clustering_result):
        result, cleaned, pre = clustering_result
        num_pipe = pre.transformer_.named_transformers_["num"]
        scaler = num_pipe.named_steps["scaler"]
        orig_centers = pd.DataFrame(
            scaler.inverse_transform(result.cluster_centers),
            columns=[ANNUAL_INCOME, SPENDING_SCORE],
        )
        profiles = build_persona_profiles(cleaned, result.labels, cluster_centers=orig_centers)
        insights = generate_persona_insights(profiles)
        assert len(insights) == result.n_clusters
        for insight in insights:
            assert "Strategy:" in insight or "Marketing Strategy:" in insight

    def test_all_five_personas_assigned(self, clustering_result):
        result, cleaned, pre = clustering_result
        num_pipe = pre.transformer_.named_transformers_["num"]
        scaler = num_pipe.named_steps["scaler"]
        orig_centers = pd.DataFrame(
            scaler.inverse_transform(result.cluster_centers),
            columns=[ANNUAL_INCOME, SPENDING_SCORE],
        )
        personas = assign_personas_from_data(orig_centers, cleaned)
        assigned_keys = {p.key for p in personas.values()}
        assert assigned_keys == {"vip", "saver", "impulsive", "budget", "mainstream"}

    def test_profile_output_format(self, clustering_result):
        result, cleaned, pre = clustering_result
        num_pipe = pre.transformer_.named_transformers_["num"]
        scaler = num_pipe.named_steps["scaler"]
        orig_centers = pd.DataFrame(
            scaler.inverse_transform(result.cluster_centers),
            columns=[ANNUAL_INCOME, SPENDING_SCORE],
        )
        profiles = build_persona_profiles(cleaned, result.labels, cluster_centers=orig_centers)
        for profile in profiles:
            assert isinstance(profile.cluster_id, int)
            assert isinstance(profile.persona_name, str)
            assert isinstance(profile.customer_count, int)
            assert isinstance(profile.percentage, float)
            assert isinstance(profile.profile_summary, str)
            assert isinstance(profile.key_characteristics, list)
