"""Business-insight generation built on top of personas and cluster stats.

This module consumes :mod:`personas` so persona text is never duplicated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .personas import Persona, PersonaProfile, build_persona_profiles, assign_personas
from .utils import ANNUAL_INCOME, AGE, GENRE, SPENDING_SCORE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonaInsight:
    """Actionable business insight for a single persona."""

    persona_name: str
    business_interpretation: str
    potential_opportunity: str
    recommended_marketing_strategy: str
    retention_engagement_recommendation: str = ""


_PERSONA_INSIGHTS: dict[str, PersonaInsight] = {
    "vip": PersonaInsight(
        persona_name="High-Value Customers",
        business_interpretation=(
            "This segment combines above-average income with above-average spending, "
            "indicating strong purchasing power and willingness to spend."
        ),
        potential_opportunity=(
            "Cross-sell premium complementary products and introduce loyalty tiers "
            "to increase share-of-wallet."
        ),
        recommended_marketing_strategy=(
            "Launch personalised premium offers, early-access campaigns, and "
            "concierge-style service to reinforce perceived value."
        ),
        retention_engagement_recommendation=(
            "Implement a VIP loyalty programme with points multipliers, exclusive events, "
            "and dedicated support to maintain high engagement and reduce churn risk."
        ),
    ),
    "saver": PersonaInsight(
        persona_name="Premium Savers",
        business_interpretation=(
            "This segment has high income but low spending, suggesting price "
            "sensitivity or low perceived value for current offerings."
        ),
        potential_opportunity=(
            "Convert untapped wealth into revenue by demonstrating premium value "
            "propositions and removing purchase friction."
        ),
        recommended_marketing_strategy=(
            "Use value-based messaging that links premium products to long-term savings "
            "or quality-of-life improvements; offer bundle deals and easy financing."
        ),
        retention_engagement_recommendation=(
            "Retain through education-rich content (e.g., product comparisons, ROI stories) "
            "and time-limited premium trial offers to build trust."
        ),
    ),
    "impulsive": PersonaInsight(
        persona_name="Budget-Conscious Spenders",
        business_interpretation=(
            "This segment has low income but high spending, indicating enthusiasm "
            "for shopping despite limited means - a potentially volatile group."
        ),
        potential_opportunity=(
            "Capture repeat purchases through affordable instalment plans, "
            "flash sales, and high-engagement social campaigns."
        ),
        recommended_marketing_strategy=(
            "Promote trend-driven, affordable products with social proof and "
            "limited-time discounts to match their spontaneous buying style."
        ),
        retention_engagement_recommendation=(
            "Offer responsible credit options, budget-planning tools, and a rewards "
            "programme that recognises frequent purchases without encouraging debt."
        ),
    ),
    "budget": PersonaInsight(
        persona_name="Low-Engagement Customers",
        business_interpretation=(
            "This segment has both low income and low spending, suggesting limited "
            "budget availability and low engagement with premium offerings."
        ),
        potential_opportunity=(
            "Grow lifetime value by introducing entry-level products and gradually "
            "upscaling as trust and disposable income increase."
        ),
        recommended_marketing_strategy=(
            "Focus on high-value, low-cost promotions, clearance events, and "
            "essential bundles that deliver clear cost savings."
        ),
        retention_engagement_recommendation=(
            "Maintain contact through email newsletters, referral incentives, and "
            "milestone-based discounts to keep the brand top-of-mind."
        ),
    ),
    "mainstream": PersonaInsight(
        persona_name="Growth Opportunity Customers",
        business_interpretation=(
            "This segment sits in the middle on income and spending, representing "
            "the largest share of typical mall traffic with balanced needs."
        ),
        potential_opportunity=(
            "Increase basket size through personalised recommendations and "
            "targeted promotions that move them toward higher-value tiers."
        ),
        recommended_marketing_strategy=(
            "Deploy standard volume promotions, seasonal campaigns, and "
            "personalised product recommendations based on browsing history."
        ),
        retention_engagement_recommendation=(
            "Nurture with loyalty points, birthday offers, and occasional "
            "upgrade incentives to gradually increase spending frequency and value."
        ),
    ),
}


def build_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Build a per-cluster profile dataframe.

    Parameters
    ----------
    df:
        Cleaned customer dataframe aligned row-for-row with *labels*.
    labels:
        Cluster label per row (from K-Means).
    """
    profile = df.copy()
    profile["Cluster"] = labels

    grouped = profile.groupby("Cluster", observed=True)

    summary = grouped.agg(
        count=(ANNUAL_INCOME, "size"),
        avg_age=(AGE, "mean"),
        avg_income=(ANNUAL_INCOME, "mean"),
        avg_spending=(SPENDING_SCORE, "mean"),
    ).round(1)

    summary["top_genre"] = grouped[GENRE].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else None
    )
    logger.info("Built profiles for %d clusters", len(summary))
    return summary


def generate_cluster_insights(
    profiles: pd.DataFrame,
    personas: dict[int, Persona],
) -> list[str]:
    """Return a list of human-readable, actionable insight strings."""
    insights: list[str] = []
    for cluster_id in sorted(personas.keys()):
        row = profiles.loc[cluster_id]
        persona = personas[cluster_id]
        insights.append(
            f"Cluster {cluster_id} - {persona.name} "
            f"({persona.description.strip()})\n"
            f"  Size: {int(row['count'])} customers | "
            f"Avg Income: {row['avg_income']}k | "
            f"Avg Spending: {row['avg_spending']} | "
            f"Top Genre: {row['top_genre']}\n"
            f"  Strategy: {persona.strategy}"
        )
    return insights


def generate_persona_insights(
    persona_profiles: list[PersonaProfile],
) -> list[str]:
    """Return detailed insight strings for each :class:`PersonaProfile`.

    Each insight contains a business interpretation, opportunity, recommended
    marketing strategy, and retention/engagement recommendation.
    """
    insights: list[str] = []
    for profile in persona_profiles:
        persona_key = None
        for key, persona in _PERSONA_INSIGHTS.items():
            if persona.persona_name == profile.persona_name:
                persona_key = key
                break

        if persona_key is None:
            insight = PersonaInsight(
                persona_name=profile.persona_name,
                business_interpretation=(
                    f"Cluster with {profile.customer_count} customers ({profile.percentage}%)."
                ),
                potential_opportunity="Target with tailored promotions and offers.",
                recommended_marketing_strategy="Use personalised marketing campaigns.",
                retention_engagement_recommendation="Monitor engagement and follow up regularly.",
            )
        else:
            insight = _PERSONA_INSIGHTS[persona_key]

        insights.append(
            f"Persona: {profile.persona_name} "
            f"(Cluster {profile.cluster_id}, {profile.customer_count} customers, "
            f"{profile.percentage}%)\n"
            f"  Profile: {profile.profile_summary}\n"
            f"  Characteristics: {', '.join(profile.key_characteristics)}\n"
            f"  Business Interpretation: {insight.business_interpretation}\n"
            f"  Opportunity: {insight.potential_opportunity}\n"
            f"  Marketing Strategy: {insight.recommended_marketing_strategy}\n"
            f"  Retention: {insight.retention_engagement_recommendation}"
        )
    return insights


def generate_report(
    profiles: pd.DataFrame,
    personas: dict[int, Persona],
    optimal_k: int,
    silhouette_scores: Optional[list[float]] = None,
) -> str:
    """Compose a full text report of results and insights."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Customer Segmentation - Report")
    lines.append("=" * 60)
    lines.append(f"Optimal number of clusters (k): {optimal_k}")
    if silhouette_scores is not None:
        best = max(silhouette_scores) if silhouette_scores else float("nan")
        lines.append(f"Best silhouette score: {best:.4f}")
    lines.append("")
    lines.append("Cluster Profiles:")
    lines.append(profiles.to_string())
    lines.append("")
    lines.append("Business Insights & Personas:")
    lines.append("-" * 60)
    lines.extend(generate_cluster_insights(profiles, personas))
    return "\n".join(lines)
