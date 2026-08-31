"""Analytical depth module for customer segmentation.

Provides clustering-oriented interpretability tools that replace any need for
supervised "feature importance" with methods appropriate to unsupervised learning:

* **Cluster profiles**   - count, percentage, mean, median, min, max per feature.
* **Segment comparison**  - per-segment vs overall-population standardised
                           differences (z-scores), expressed in plain language.
* **Feature separation**  - standardised between-cluster variation per feature
                           (ANOVA F-ratio), revealing which features drive the
                           segmentation most strongly.
* **Cluster stability**   - ARI across repeated K-Means runs with different
                           random seeds; reports a single interpretable score.
* **Analytical insights** - automatically derived observations (largest segment,
                           highest-spending, highest-income, strongest
                           income/spending correlation, most distinctive segment).

All functions operate on pandas DataFrames / numpy arrays returned by the
existing pipeline in ``src/``.  No function in this module modifies its inputs.

Methodology notes
-----------------
**Standardised differences (segment vs overall):**
    For each feature f and segment s::

        delta_s,f = (mean_s,f - overall_mean_f) / overall_std_f

    Positive values mean the segment is above average; negative means below.
    The absolute value indicates effect size (Cohen's d convention: |d|>0.8
    is "large").

**Feature separation (ANOVA F-ratio):**
    For each feature f the one-way ANOVA F-statistic is computed across all
    clusters.  A larger F-ratio means that feature varies more between clusters
    relative to within clusters, making it a stronger differentiator of the
    segmentation.  This is the standard clustering-oriented analogue of
    supervised feature importance.

**Cluster stability (Adjusted Rand Index):**
    K-Means is run *n_runs* times with different random seeds on the same data.
    Pairwise ARI scores are averaged.  A score of 1.0 means every run assigns
    identical labels; 0.0 means no agreement beyond chance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .utils import ANNUAL_INCOME, NUMERIC_COLUMNS, SPENDING_SCORE

logger = logging.getLogger(__name__)


# ===========================================================================
# Result containers
# ===========================================================================
@dataclass(frozen=True)
class ClusterProfile:
    """Comprehensive statistics for a single cluster."""

    cluster_id: int
    count: int
    percentage: float
    features: dict[str, dict[str, float]]  # feature -> {mean, median, min, max, std}


@dataclass(frozen=True)
class SegmentComparison:
    """Per-segment comparison against the overall population."""

    cluster_id: int
    feature_deltas: dict[str, float]  # feature -> standardised delta (Cohen's d)
    interpretation: str


@dataclass(frozen=True)
class FeatureSeparation:
    """Feature importance / separation analysis for clustering."""

    feature: str
    f_ratio: float
    p_value: float
    rank: int  # 1 = most differentiating


@dataclass(frozen=True)
class ClusterStabilityResult:
    """Stability summary for repeated clustering runs."""

    algorithm: str
    n_clusters: int
    n_runs: int
    mean_ari: float
    min_ari: float
    max_ari: float
    std_ari: float
    interpretation: str


@dataclass(frozen=True)
class AnalyticalInsight:
    """A single automatically generated insight."""

    title: str
    detail: str
    segment_id: Optional[int] = None


# ===========================================================================
# 2. Segment Comparison
# ===========================================================================
def compare_segments_vs_overall(
    df: pd.DataFrame,
    labels: np.ndarray,
    numeric_features: Optional[list[str]] = None,
) -> list[SegmentComparison]:
    """Compare each segment's averages against the overall population.

    Uses standardised mean differences (Cohen's d) to quantify how unusually
    high or low each segment is on every numeric feature.

    Parameters
    ----------
    df:
        Cleaned customer dataframe aligned row-for-row with *labels*.
    labels:
        Cluster label per row.
    numeric_features:
        Features to compare.  Defaults to :data:`~src.utils.NUMERIC_COLUMNS`.

    Returns
    -------
    list[SegmentComparison]
        One entry per unique cluster label.
    """
    if len(df) != len(labels):
        raise ValueError(
            f"DataFrame rows ({len(df)}) must match labels length ({len(labels)})."
        )

    features = numeric_features if numeric_features is not None else NUMERIC_COLUMNS
    available = [f for f in features if f in df.columns]
    if not available:
        raise ValueError(f"None of the requested features {features} are in the DataFrame.")

    data = df[available].copy()
    overall_mean = data.mean()
    overall_std = data.std(ddof=0)
    overall_std = overall_std.replace(0, 1e-9)

    comparisons: list[SegmentComparison] = []
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        segment_data = data[mask]
        segment_mean = segment_data.mean()

        deltas = {}
        for f in available:
            delta = (segment_mean[f] - overall_mean[f]) / overall_std[f]
            deltas[f] = round(float(delta), 3)

        parts = [f"{f}: {v:+.2f}σ" for f, v in deltas.items()]
        interp = f"Cluster {cluster_id} vs overall: " + ", ".join(parts)
        comparisons.append(
            SegmentComparison(
                cluster_id=int(cluster_id),
                feature_deltas=deltas,
                interpretation=interp,
            )
        )

    logger.info("Computed segment comparisons for %d clusters.", len(comparisons))
    return comparisons


def comparison_to_dataframe(comparisons: list[SegmentComparison]) -> pd.DataFrame:
    """Convert a list of :class:`SegmentComparison` to a display-ready DataFrame."""
    rows = []
    for c in comparisons:
        row = {"Cluster": c.cluster_id}
        row.update(c.feature_deltas)
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# 3. Feature Importance / Separation Analysis
# ===========================================================================
def compute_feature_separation(
    df: pd.DataFrame,
    labels: np.ndarray,
    numeric_features: Optional[list[str]] = None,
) -> list[FeatureSeparation]:
    """Compute feature-wise between-cluster variation (ANOVA F-ratio).

    This is the appropriate clustering-oriented interpretation of feature
    importance.  For each numeric feature, a one-way ANOVA F-statistic measures
    how much that feature varies *between* clusters relative to how much it
    varies *within* clusters.

    A larger F-ratio means the feature is a stronger differentiator of the
    segments — analogous to feature importance in supervised learning, but
    derived entirely from the cluster structure.

    Parameters
    ----------
    df:
        Cleaned customer dataframe aligned row-for-row with *labels*.
    labels:
        Cluster label per row.
    numeric_features:
        Features to analyse.  Defaults to :data:`~src.utils.NUMERIC_COLUMNS`.

    Returns
    -------
    list[FeatureSeparation]
        One entry per feature, sorted by F-ratio descending (rank 1 = most
        differentiating).
    """
    if len(df) != len(labels):
        raise ValueError(
            f"DataFrame rows ({len(df)}) must match labels length ({len(labels)})."
        )

    features = NUMERIC_COLUMNS if numeric_features is None else numeric_features
    available = [f for f in features if f in df.columns]
    if not available:
        raise ValueError(f"None of the requested features {features} are in the DataFrame.")

    unique_labels = np.unique(labels)
    n_groups = len(unique_labels)
    if n_groups < 2:
        logger.warning("Need at least 2 clusters for feature separation analysis.")
        return []

    results: list[FeatureSeparation] = []
    for feature in available:
        groups = [df[labels == lbl][feature].values for lbl in unique_labels]
        f_ratio, p_value = _anova_f(groups)
        results.append(
            FeatureSeparation(
                feature=feature,
                f_ratio=round(float(f_ratio), 4),
                p_value=round(float(p_value), 4),
                rank=0,
            )
        )

    results.sort(key=lambda r: r.f_ratio, reverse=True)
    for i, r in enumerate(results):
        r.__dict__["rank"] = i + 1

    logger.info("Feature separation (F-ratio): %s", {r.feature: r.f_ratio for r in results})
    return results


def _anova_f(groups: list[np.ndarray]) -> tuple[float, float]:
    """One-way ANOVA F-statistic.

    Parameters
    ----------
    groups:
        List of 1-D arrays, one per group.

    Returns
    -------
    (f_ratio, p_value)
        The F-statistic and its associated p-value (scipy-free implementation
        using the standard formula).
    """
    n_total = sum(len(g) for g in groups)
    k = len(groups)

    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

    df_between = k - 1
    df_within = n_total - k

    if df_within <= 0 or ss_within == 0:
        return float("inf"), 0.0

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    f_ratio = ms_between / ms_within

    try:
        from scipy import stats

        p_value = float(stats.f.sf(f_ratio, df_between, df_within))
    except ImportError:
        p_value = _f_pvalue_approx(f_ratio, df_between, df_within)

    return float(f_ratio), p_value


def _f_pvalue_approx(f_stat: float, df1: int, df2: int) -> float:
    """Fallback p-value approximation when scipy is unavailable.

    Uses the regularised incomplete beta function via mpmath if available,
    otherwise returns 0.0 for large F and 1.0 for F <= 1.
    """
    try:
        from mpmath import betainc

        x = df1 * f_stat / (df1 * f_stat + df2)
        p = float(betainc(df1 / 2, df2 / 2, x=0, y=x, regularized=True))
        return 1.0 - p
    except Exception:
        if f_stat > 1:
            return 0.0
        return 1.0


# ===========================================================================
# 4. Cluster Stability
# ===========================================================================
def compute_cluster_stability(
    X: np.ndarray,
    n_clusters: int = 5,
    n_runs: int = 10,
    random_seeds: Optional[list[int]] = None,
) -> ClusterStabilityResult:
    """Evaluate K-Means stability across multiple random seeds.

    Runs K-Means *n_runs* times with different random seeds, computes pairwise
    Adjusted Rand Index (ARI) between all runs, and returns a summary.

    ARI ranges from 0.0 (random agreement) to 1.0 (perfect agreement).
    Interpretation guide:

    * **≥ 0.90** — Highly stable; labels are reliable.
    * **0.75 – 0.90** — Stable; minor label-switching expected.
    * **< 0.75** — Unstable; the cluster structure may be weak or
      the data may not have well-separated groups.

    Parameters
    ----------
    X:
        Scaled feature matrix.
    n_clusters:
        Number of clusters for K-Means.
    n_runs:
        Number of independent runs.
    random_seeds:
        Optional explicit list of random seeds (length must equal *n_runs*).
        If ``None``, seeds 0, 1, ..., *n_runs*-1 are used.

    Returns
    -------
    ClusterStabilityResult
        Summary with mean/min/max/std ARI and a plain-language interpretation.
    """
    from .clustering import run_kmeans

    if random_seeds is not None and len(random_seeds) != n_runs:
        raise ValueError(
            f"random_seeds length ({len(random_seeds)}) must equal n_runs ({n_runs})."
        )

    seeds = random_seeds if random_seeds is not None else list(range(n_runs))
    label_sets: list[np.ndarray] = []

    for seed in seeds:
        result = run_kmeans(X, n_clusters=n_clusters, random_state=seed)
        label_sets.append(result.labels)

    pairwise_aris = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            ari = adjusted_rand_score(label_sets[i], label_sets[j])
            pairwise_aris.append(ari)

    if not pairwise_aris:
        mean_ari = min_ari = max_ari = std_ari = 0.0
    else:
        mean_ari = float(np.mean(pairwise_aris))
        min_ari = float(np.min(pairwise_aris))
        max_ari = float(np.max(pairwise_aris))
        std_ari = float(np.std(pairwise_aris))

    if mean_ari >= 0.90:
        interp = (
            "Highly stable (mean ARI {:.3f}). K-Means produces consistent "
            "cluster assignments across different random seeds; the segments "
            "are reliable for business use.".format(mean_ari)
        )
    elif mean_ari >= 0.75:
        interp = (
            "Mostly stable (mean ARI {:.3f}). Minor label-switching occurs "
            "between runs, but the overall segment structure is consistent.".format(mean_ari)
        )
    else:
        interp = (
            "Potentially unstable (mean ARI {:.3f}). Cluster assignments vary "
            "considerably across runs, suggesting weak or overlapping segment "
            "structure. Consider more distinct features or a different algorithm.".format(mean_ari)
        )

    logger.info(
        "Stability (k=%d, runs=%d): ARI mean=%.3f min=%.3f max=%.3f",
        n_clusters,
        n_runs,
        mean_ari,
        min_ari,
        max_ari,
    )
    return ClusterStabilityResult(
        algorithm="kmeans",
        n_clusters=n_clusters,
        n_runs=n_runs,
        mean_ari=mean_ari,
        min_ari=min_ari,
        max_ari=max_ari,
        std_ari=std_ari,
        interpretation=interp,
    )


# ===========================================================================
# 5. Analytical Insights
# ===========================================================================
def generate_analytical_insights(
    profiles_df: pd.DataFrame,
    centers_orig: pd.DataFrame,
    labels: np.ndarray,
    df: pd.DataFrame,
) -> list[AnalyticalInsight]:
    """Automatically generate data-driven insights about the segmentation.

    Every insight is derived from the actual data — no fabricated conclusions.

    Parameters
    ----------
    profiles_df:
        Output of :func:`src.business_insights.build_cluster_profiles`.
    centers_orig:
        Cluster centres on the original (unscaled) feature scale.
    labels:
        Cluster label per row.
    df:
        Cleaned customer dataframe (used for correlation and top-spender lookup).

    Returns
    -------
    list[AnalyticalInsight]
        Ordered list of insights covering largest segment, highest-spending,
        highest-income, strongest income/spending relationship, and most
        distinctive segment characteristics.
    """
    insights: list[AnalyticalInsight] = []
    income_col = ANNUAL_INCOME
    spending_col = SPENDING_SCORE

    if profiles_df.empty or centers_orig.empty:
        return insights

    # --- Helper: safe column lookup ---
    def _safe_mean(df_in: pd.DataFrame, cid: int, col: str) -> float:
        try:
            return float(df_in.loc[cid, f"{col}_mean"])
        except (KeyError, TypeError):
            try:
                return float(centers_orig.loc[cid, col])
            except (KeyError, TypeError):
                return float("nan")

    # 1. Largest segment
    if "count" in profiles_df.columns:
        largest_cid = int(profiles_df["count"].idxmax())
        largest_count = int(profiles_df.loc[largest_cid, "count"])
        largest_pct = float(profiles_df.loc[largest_cid, "percentage"])
        insights.append(
            AnalyticalInsight(
                title="Largest Segment",
                detail=(
                    f"Segment {largest_cid} is the largest with "
                    f"{largest_count:,} customers ({largest_pct:.1f}% of the base)."
                ),
                segment_id=largest_cid,
            )
        )

    # 2. Highest-spending segment
    spending_key = f"{spending_col}_mean"
    if spending_key in profiles_df.columns:
        spend_cid = int(profiles_df[spending_key].idxmax())
        spend_val = float(profiles_df.loc[spend_cid, spending_key])
        insights.append(
            AnalyticalInsight(
                title="Highest-Spending Segment",
                detail=(
                    f"Segment {spend_cid} has the highest average spending "
                    f"score at {spend_val:.1f}."
                ),
                segment_id=spend_cid,
            )
        )

    # 3. Highest-income segment
    income_key = f"{income_col}_mean"
    if income_key in profiles_df.columns:
        income_cid = int(profiles_df[income_key].idxmax())
        income_val = float(profiles_df.loc[income_cid, income_key])
        insights.append(
            AnalyticalInsight(
                title="Highest-Income Segment",
                detail=(
                    f"Segment {income_cid} has the highest average income "
                    f"at ${income_val:.1f}k."
                ),
                segment_id=income_cid,
            )
        )

    # 4. Strongest income/spending relationship (highest combined)
    if income_col in centers_orig.columns and spending_col in centers_orig.columns:
        centers_orig["_combined"] = (
            centers_orig[income_col] + centers_orig[spending_col]
        )
        best_cid = int(centers_orig["_combined"].idxmax())
        best_inc = float(centers_orig.loc[best_cid, income_col])
        best_spd = float(centers_orig.loc[best_cid, spending_col])
        del centers_orig["_combined"]
        insights.append(
            AnalyticalInsight(
                title="Strongest Income/Spending Relationship",
                detail=(
                    f"Segment {best_cid} shows the strongest combined income "
                    f"and spending: ${best_inc:.1f}k income, "
                    f"{best_spd:.1f} spending score."
                ),
                segment_id=best_cid,
            )
        )

    # 5. Most distinctive segment characteristics
    std_col = f"{income_col}_std"
    if std_col in profiles_df.columns:
        std_vals = profiles_df[std_col].dropna()
        if not std_vals.empty:
            most_distinct_cid = int(std_vals.idxmax())
            std_val = float(std_vals.loc[most_distinct_cid])
            insights.append(
                AnalyticalInsight(
                    title="Most Distinctive Segment",
                    detail=(
                        f"Segment {most_distinct_cid} has the widest internal "
                        f"income spread (std ${std_val:.1f}k), making it the most "
                        f"heterogeneous segment — avoid one-size-fits-all "
                        f"marketing for this group."
                    ),
                    segment_id=most_distinct_cid,
                )
            )

    # 6. Income vs spending correlation insight
    if income_col in df.columns and spending_col in df.columns:
        corr = df[income_col].corr(df[spending_col])
        corr_val = float(corr)
        direction = "positive" if corr_val > 0 else "negative"
        strength = (
            "strong" if abs(corr_val) > 0.5 else
            "moderate" if abs(corr_val) > 0.3 else
            "weak"
        )
        insights.append(
            AnalyticalInsight(
                title="Income vs Spending Relationship",
                detail=(
                    f"The overall correlation between income and spending score "
                    f"is {corr_val:+.3f} ({strength} {direction}). "
                    f"A {'higher' if corr_val > 0 else 'lower'} spending score "
                    f"tends to accompany {'higher' if corr_val > 0 else 'lower'} "
                    f"income across the customer base."
                ),
            )
        )

    logger.info("Generated %d analytical insights.", len(insights))
    return insights


def insights_to_dataframe(insights: list[AnalyticalInsight]) -> pd.DataFrame:
    """Convert a list of :class:`AnalyticalInsight` to a display-ready DataFrame."""
    rows = [
        {
            "Insight": i.title,
            "Detail": i.detail,
            "Segment": i.segment_id if i.segment_id is not None else "All",
        }
        for i in insights
    ]
    return pd.DataFrame(rows)
