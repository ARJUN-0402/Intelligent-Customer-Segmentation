"""Cluster-to-persona mapping and persona definitions.

Personas are classification labels derived from each cluster's average
income and spending score. The rules are data-driven (quartile breakpoints)
yet deterministic and reusable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .utils import ANNUAL_INCOME, AGE, GENRE, SPENDING_SCORE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Persona:
    """A named, interpretable customer persona."""

    key: str
    name: str
    description: str
    strategy: str
    color: str


@dataclass(frozen=True)
class PersonaProfile:
    """Full persona output for a single cluster."""

    cluster_id: int
    persona_name: str
    customer_count: int
    percentage: float
    profile_summary: str
    key_characteristics: list[str] = field(default_factory=list)


PERSONA_CATALOG: dict[str, Persona] = {
    "vip": Persona(
        key="vip",
        name="High-Value Customers",
        description="High income and high spending - the ideal customer group.",
        strategy="Offer exclusive premium products, loyalty rewards, and "
        "VIP-only experiences to retain engagement.",
        color="tab:green",
    ),
    "saver": Persona(
        key="saver",
        name="Premium Savers",
        description="High income but low spending - wealthy but cautious.",
        strategy="Target with premium products paired with value messaging "
        "and convenience incentives to convert savings into spending.",
        color="tab:blue",
    ),
    "impulsive": Persona(
        key="impulsive",
        name="Budget-Conscious Spenders",
        description="Low income but high spending - enthusiastic but risky.",
        strategy="Promote affordable, high-engagement products and "
        "responsible credit or budget-planning tools.",
        color="tab:red",
    ),
    "budget": Persona(
        key="budget",
        name="Low-Engagement Customers",
        description="Low income and low spending - cautious with money.",
        strategy="Highlight value packs, discounts, and budget-friendly "
        "options to build trust and modest engagement.",
        color="tab:purple",
    ),
    "mainstream": Persona(
        key="mainstream",
        name="Growth Opportunity Customers",
        description="Average income and average spending - the broad middle.",
        strategy="Deploy standard marketing campaigns and volume promotions "
        "to sustain steady engagement across the segment.",
        color="tab:orange",
    ),
}


def compute_breakpoints(values: pd.Series) -> tuple[float, float]:
    """Return the (low, high) quartile thresholds for *values*."""
    low = float(values.quantile(0.25))
    high = float(values.quantile(0.75))
    logger.info(
        "Computed breakpoints for '%s': low=%.2f, high=%.2f", values.name, low, high
    )
    return low, high


def _band(value: float, low: float, high: float) -> str:
    """Classify *value* as ``"low"``, ``"mid"`` or ``"high"``."""
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "mid"


def classify_persona(
    income: float,
    spending: float,
    income_low: float,
    income_high: float,
    spending_low: float,
    spending_high: float,
) -> Persona:
    """Map an income/spending pair to the most fitting :class:`Persona`."""
    inc = _band(income, income_low, income_high)
    spd = _band(spending, spending_low, spending_high)
    key = _PERSONA_MATRIX.get((inc, spd), "mainstream")
    return PERSONA_CATALOG[key]


_PERSONA_MATRIX: dict[tuple[str, str], str] = {
    ("high", "high"): "vip",
    ("high", "low"): "saver",
    ("low", "high"): "impulsive",
    ("low", "low"): "budget",
    ("mid", "high"): "impulsive",
    ("high", "mid"): "saver",
    ("low", "mid"): "budget",
}


def assign_personas(
    cluster_centers: pd.DataFrame,
    breakpoints: Optional[dict[str, tuple[float, float]]] = None,
) -> dict[int, Persona]:
    """Assign a :class:`Persona` to every cluster.

    Parameters
    ----------
    cluster_centers:
        DataFrame with one row per cluster and columns including
        :data:`~src.utils.ANNUAL_INCOME` and
        :data:`~src.utils.SPENDING_SCORE` (original currency scale).
    breakpoints:
        Optional ``{"Annual Income (k$)": (low, high), ...}`` overrides.
        If ``None`` the quartiles are derived from *cluster_centers*.
    """
    if ANNUAL_INCOME not in cluster_centers.columns or SPENDING_SCORE not in cluster_centers.columns:
        raise KeyError(
            f"cluster_centers must contain '{ANNUAL_INCOME}' and '{SPENDING_SCORE}'"
        )

    if breakpoints is None:
        inc_low, inc_high = compute_breakpoints(cluster_centers[ANNUAL_INCOME])
        spd_low, spd_high = compute_breakpoints(cluster_centers[SPENDING_SCORE])
    else:
        inc_low, inc_high = breakpoints[ANNUAL_INCOME]
        spd_low, spd_high = breakpoints[SPENDING_SCORE]

    personas: dict[int, Persona] = {}
    for cluster_id, row in cluster_centers.iterrows():
        personas[int(cluster_id)] = classify_persona(
            float(row[ANNUAL_INCOME]),
            float(row[SPENDING_SCORE]),
            inc_low,
            inc_high,
            spd_low,
            spd_high,
        )
    return personas


def assign_personas_from_data(
    cluster_centers: pd.DataFrame,
    data: pd.DataFrame,
) -> dict[int, Persona]:
    """Assign personas using breakpoints derived from the full dataset.

    This ensures persona labels reflect actual customer characteristics rather
    than arbitrary cluster-centre quartiles.

    Parameters
    ----------
    cluster_centers:
        DataFrame with one row per cluster and columns including
        :data:`~src.utils.ANNUAL_INCOME` and
        :data:`~src.utils.SPENDING_SCORE`.
    data:
        Full cleaned customer dataframe used to compute global quartiles.
    """
    if ANNUAL_INCOME not in data.columns or SPENDING_SCORE not in data.columns:
        raise KeyError(
            f"data must contain '{ANNUAL_INCOME}' and '{SPENDING_SCORE}'"
        )
    inc_low, inc_high = compute_breakpoints(data[ANNUAL_INCOME])
    spd_low, spd_high = compute_breakpoints(data[SPENDING_SCORE])
    return assign_personas(
        cluster_centers,
        breakpoints={
            ANNUAL_INCOME: (inc_low, inc_high),
            SPENDING_SCORE: (spd_low, spd_high),
        },
    )


def build_persona_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_centers: Optional[pd.DataFrame] = None,
    personas: Optional[dict[int, Persona]] = None,
) -> list[PersonaProfile]:
    """Build a list of :class:`PersonaProfile` for every cluster.

    Parameters
    ----------
    df:
        Cleaned customer dataframe aligned row-for-row with *labels*.
    labels:
        Cluster label per row (from K-Means or other algorithm).
    cluster_centers:
        Optional DataFrame with one row per cluster and columns including
        :data:`~src.utils.ANNUAL_INCOME` and
        :data:`~src.utils.SPENDING_SCORE`. Required if *personas* is ``None``.
    personas:
        Optional mapping of cluster_id -> :class:`Persona` from
        :func:`assign_personas` or :func:`assign_personas_from_data`.
        If ``None``, personas are derived from *df* using
        :func:`assign_personas_from_data`.
    """
    if len(df) != len(labels):
        raise ValueError(
            f"DataFrame rows ({len(df)}) must match labels length ({len(labels)})."
        )

    if personas is None:
        if cluster_centers is None:
            raise ValueError(
                "Either 'personas' or 'cluster_centers' must be provided."
            )
        personas = assign_personas_from_data(cluster_centers, df)

    total = len(df)
    profile = df.copy()
    profile["Cluster"] = labels

    grouped = profile.groupby("Cluster", observed=True)
    counts = grouped.size()
    avg_age = grouped[AGE].mean()
    avg_income = grouped[ANNUAL_INCOME].mean()
    avg_spending = grouped[SPENDING_SCORE].mean()
    top_genre = grouped[GENRE].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"
    )

    profiles: list[PersonaProfile] = []
    for cluster_id in sorted(personas.keys()):
        persona = personas[cluster_id]
        count = int(counts.get(cluster_id, 0))
        pct = round(100.0 * count / total, 1) if total > 0 else 0.0
        age = float(avg_age.get(cluster_id, np.nan))
        inc = float(avg_income.get(cluster_id, np.nan))
        spd = float(avg_spending.get(cluster_id, np.nan))
        genre = str(top_genre.get(cluster_id, "Unknown"))

        summary = (
            f"{count} customers ({pct}% of total), "
            f"avg age {age:.1f}, avg income ${inc:.1f}k, "
            f"avg spending score {spd:.1f}, predominantly {genre}."
        )

        chars: list[str] = []
        if not np.isnan(inc) and inc > 70:
            chars.append("high income")
        elif not np.isnan(inc) and inc < 40:
            chars.append("low income")
        else:
            chars.append("moderate income")

        if not np.isnan(spd) and spd > 70:
            chars.append("high spending")
        elif not np.isnan(spd) and spd < 35:
            chars.append("low spending")
        else:
            chars.append("moderate spending")

        if not np.isnan(age) and age < 30:
            chars.append("younger demographic")
        elif not np.isnan(age) and age > 40:
            chars.append("older demographic")
        else:
            chars.append("middle-aged demographic")

        if genre and genre != "Unknown":
            chars.append(f"majority {genre.lower()}")

        profiles.append(
            PersonaProfile(
                cluster_id=int(cluster_id),
                persona_name=persona.name,
                customer_count=count,
                percentage=pct,
                profile_summary=summary,
                key_characteristics=chars,
            )
        )
    return profiles
