"""Model evaluation: multi-metric comparison and automatic model selection.

Extends the K-Means-only elbow/silhouette helpers with:

* Silhouette Score
* Calinski-Harabasz Index
* Davies-Bouldin Index
* K-Means Inertia

Provides:

* :func:`evaluate_model` — compute all applicable metrics for a fitted model.
* :func:`compare_models` — evaluate a list of models and return a structured table.
* :func:`select_best_clustering_model` — rank candidates and recommend the best
  configuration for the current dataset.

Metrics are only computed when they are meaningful for the given algorithm /
cluster count. Inertia is exclusive to K-Means. Silhouette, Calinski-Harabasz,
and Davies-Bouldin require at least two clusters.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .clustering import ClusterResult
from .utils import (
    DEFAULT_RANDOM_STATE,
    FIGURES_DIR,
    MAX_K,
    MIN_K,
    N_INIT,
    ensure_directory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class MetricResult:
    """Single metric outcome.

    Attributes
    ----------
    name:
        Human-readable metric name.
    value:
        Computed score, or ``None`` if unavailable / invalid.
    available:
        Whether the metric could be computed for this model.
    note:
        Optional explanation when *value* is ``None``.
    """

    name: str
    value: Optional[float]
    available: bool
    note: str = ""


@dataclass
class ModelEvaluationResult:
    """Evaluation output for one clustering model.

    Attributes
    ----------
    algorithm:
        Algorithm identifier (e.g. ``'kmeans'``).
    n_clusters:
        Number of clusters found (excludes noise).
    metrics:
        Mapping of metric name to :class:`MetricResult`.
    model:
        The fitted estimator.
    labels:
        Cluster label per sample.
    """

    algorithm: str
    n_clusters: int
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    model: object = None
    labels: Optional[np.ndarray] = None


@dataclass
class ComparisonTable:
    """Structured comparison of multiple models.

    Attributes
    ----------
    rows:
        One dict per model with keys:
        ``Algorithm``, ``Clusters``, ``Silhouette Score``,
        ``Calinski-Harabasz Score``, ``Davies-Bouldin Score``, ``Inertia``.
    recommendation:
        The algorithm name of the recommended model, or ``None`` if the
        table is empty.
    ranking:
        Ordered list of ``(algorithm, score)`` tuples where *score* is the
        normalized composite score used for ranking.
    """

    rows: list[dict[str, object]] = field(default_factory=list)
    recommendation: Optional[str] = None
    ranking: list[tuple[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> MetricResult:
    """Compute silhouette score if at least two valid clusters exist."""
    unique = np.unique(labels)
    n_valid = len(unique[unique != -1])
    if n_valid < 2:
        return MetricResult(
            name="Silhouette Score",
            value=None,
            available=False,
            note="Requires >= 2 clusters (excluding noise).",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = float(silhouette_score(X, labels))
        return MetricResult(name="Silhouette Score", value=score, available=True)
    except Exception as exc:
        return MetricResult(
            name="Silhouette Score",
            value=None,
            available=False,
            note=f"Computation failed: {exc}",
        )


def _safe_calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> MetricResult:
    """Compute Calinski-Harabasz index if at least two valid clusters exist."""
    unique = np.unique(labels)
    n_valid = len(unique[unique != -1])
    if n_valid < 2:
        return MetricResult(
            name="Calinski-Harabasz Score",
            value=None,
            available=False,
            note="Requires >= 2 clusters (excluding noise).",
        )
    try:
        score = float(calinski_harabasz_score(X, labels))
        return MetricResult(name="Calinski-Harabasz Score", value=score, available=True)
    except Exception as exc:
        return MetricResult(
            name="Calinski-Harabasz Score",
            value=None,
            available=False,
            note=f"Computation failed: {exc}",
        )


def _safe_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> MetricResult:
    """Compute Davies-Bouldin index if at least two valid clusters exist."""
    unique = np.unique(labels)
    n_valid = len(unique[unique != -1])
    if n_valid < 2:
        return MetricResult(
            name="Davies-Bouldin Score",
            value=None,
            available=False,
            note="Requires >= 2 clusters (excluding noise).",
        )
    try:
        score = float(davies_bouldin_score(X, labels))
        return MetricResult(name="Davies-Bouldin Score", value=score, available=True)
    except Exception as exc:
        return MetricResult(
            name="Davies-Bouldin Score",
            value=None,
            available=False,
            note=f"Computation failed: {exc}",
        )


def _safe_inertia(result: ClusterResult) -> MetricResult:
    """Return inertia if the model exposes it (K-Means only)."""
    if result.inertia is not None:
        return MetricResult(name="Inertia", value=float(result.inertia), available=True)
    return MetricResult(
        name="Inertia",
        value=None,
        available=False,
        note="Only available for K-Means.",
    )


# ---------------------------------------------------------------------------
# Core evaluation API
# ---------------------------------------------------------------------------
def evaluate_model(X: np.ndarray, result: ClusterResult) -> ModelEvaluationResult:
    """Compute every metric applicable to *result*.

    Parameters
    ----------
    X:
        Scaled feature matrix used for fitting.
    result:
        Output of :func:`src.clustering.run_clustering` (or any compatible
        :class:`~src.clustering.ClusterResult`).

    Returns
    -------
    ModelEvaluationResult
        Contains all available metrics with explicit availability flags.
    """
    metrics: dict[str, MetricResult] = {
        "Silhouette Score": _safe_silhouette(X, result.labels),
        "Calinski-Harabasz Score": _safe_calinski_harabasz(X, result.labels),
        "Davies-Bouldin Score": _safe_davies_bouldin(X, result.labels),
        "Inertia": _safe_inertia(result),
    }

    for m in metrics.values():
        if m.available:
            logger.info(
                "%s (%s, k=%d): %.4f", m.name, result.algorithm, result.n_clusters, m.value
            )
        else:
            logger.debug(
                "%s (%s, k=%d) unavailable: %s",
                m.name,
                result.algorithm,
                result.n_clusters,
                m.note,
            )

    return ModelEvaluationResult(
        algorithm=result.algorithm,
        n_clusters=result.n_clusters,
        metrics=metrics,
        model=result.model,
        labels=result.labels.copy(),
    )


def compare_models(
    X: np.ndarray,
    results: Sequence[ClusterResult],
) -> ComparisonTable:
    """Evaluate multiple clustering results and build a comparison table.

    Parameters
    ----------
    X:
        Scaled feature matrix.
    results:
        Sequence of fitted clustering results.

    Returns
    -------
    ComparisonTable
        Structured rows suitable for DataFrame conversion or Streamlit
        display. Each row contains the algorithm name, cluster count, and
        every metric that could be computed (``None`` when unavailable).
    """
    evaluations = [evaluate_model(X, r) for r in results]
    evaluations = [
        ev
        for ev in evaluations
        if any(m.available for m in ev.metrics.values())
    ]
    rows: list[dict[str, object]] = []
    for ev in evaluations:
        row: dict[str, object] = {
            "Algorithm": ev.algorithm,
            "Clusters": ev.n_clusters,
        }
        for m in ev.metrics.values():
            row[m.name] = m.value
        rows.append(row)

    ranking = _rank_models(evaluations)
    recommendation = ranking[0][0] if ranking else None

    return ComparisonTable(rows=rows, recommendation=recommendation, ranking=ranking)


def _rank_models(
    evaluations: Sequence[ModelEvaluationResult],
) -> list[tuple[str, float]]:
    """Rank models by normalised composite score (higher = better).

    **Ranking methodology (dataset-specific):**

    1. For each metric, collect values across all models that have that
       metric available.
    2. Min-max normalise each metric to ``[0, 1]``.
       * Silhouette Score — higher is better.
       * Calinski-Harabasz Score — higher is better.
       * Davies-Bouldin Score — lower is better (reversed).
       * Inertia — lower is better (reversed).
    3. If all models share the same value for a metric (zero range), that
       metric contributes 0 to avoid division-by-zero artefacts.
    4. Per model, average the normalised scores of **all available metrics**.
       Models with fewer available metrics are not penalised for missing
       metrics; the average is taken over whatever is present.
    5. Models with no available metrics receive a score of 0.0 and are
       ranked last.
    6. Sort descending by composite score. Ties are broken alphabetically
       by algorithm name for determinism.

    This ranking is **dataset-specific**: the relative normalised scores
    depend entirely on the distribution of metrics observed for the input
    data and candidate configurations. It does not imply universal
    optimality.
    """
    metric_names = [
        "Silhouette Score",
        "Calinski-Harabasz Score",
        "Davies-Bouldin Score",
        "Inertia",
    ]
    # direction: True = higher is better, False = lower is better
    directions = {
        "Silhouette Score": True,
        "Calinski-Harabasz Score": True,
        "Davies-Bouldin Score": False,
        "Inertia": False,
    }

    # Collect values per metric
    values: dict[str, list[float]] = {m: [] for m in metric_names}
    available_mask: dict[str, list[bool]] = {m: [] for m in metric_names}
    for ev in evaluations:
        for m in metric_names:
            if m in ev.metrics and ev.metrics[m].available and ev.metrics[m].value is not None:
                values[m].append(float(ev.metrics[m].value))
                available_mask[m].append(True)
            else:
                values[m].append(0.0)
                available_mask[m].append(False)

    # Compute normalised scores
    norm_scores: dict[str, list[float]] = {m: [] for m in metric_names}
    for m in metric_names:
        valid_vals = [v for v, ok in zip(values[m], available_mask[m]) if ok]
        if not valid_vals:
            # No model has this metric → all zeros
            norm_scores[m] = [0.0] * len(evaluations)
            continue
        vmin, vmax = min(valid_vals), max(valid_vals)
        rng = vmax - vmin
        for v, ok in zip(values[m], available_mask[m]):
            if not ok:
                norm_scores[m].append(0.0)
            elif rng == 0:
                norm_scores[m].append(1.0)
            else:
                norm = (v - vmin) / rng
                if not directions[m]:
                    norm = 1.0 - norm
                norm_scores[m].append(norm)

    # Composite score: average over available metrics per model
    composite: list[tuple[str, float]] = []
    for i, ev in enumerate(evaluations):
        scores = [
            norm_scores[m][i]
            for m in metric_names
            if available_mask[m][i]
        ]
        avg = sum(scores) / len(scores) if scores else 0.0
        composite.append((ev.algorithm, avg))

    # Sort descending by score, tie-break alphabetically
    composite.sort(key=lambda x: (-x[1], x[0]))
    return composite


def select_best_clustering_model(
    X: np.ndarray,
    candidates: Sequence[ClusterResult],
) -> tuple[ModelEvaluationResult, ComparisonTable]:
    """Evaluate *candidates* and recommend the best model for *X*.

    The recommendation is **dataset-specific**: it reflects the relative
    performance of the supplied configurations on the provided data. It
    does not claim universal optimality.

    Parameters
    ----------
    X:
        Scaled feature matrix used for fitting the candidate models.
    candidates:
        Sequence of fitted :class:`~src.clustering.ClusterResult` objects
        to compare.

    Returns
    -------
    best:
        The evaluation result of the highest-ranked model.
    table:
        Full comparison table including all candidates and their metrics.

    Raises
    ------
    ValueError
        If *candidates* is empty.
    """
    if not candidates:
        raise ValueError("Cannot select best model from an empty candidate list.")

    table = compare_models(X, candidates)
    if table.recommendation is None:
        raise RuntimeError(
            "No valid clustering metrics could be computed for any candidate."
        )

    for c in candidates:
        ev = evaluate_model(X, c)
        if ev.algorithm == table.recommendation:
            return ev, table

    # Fallback (should not happen with consistent data)
    raise RuntimeError("Recommended model not found in candidates.")


# ---------------------------------------------------------------------------
# Dataset-level evaluation runner (consumable by Streamlit / app.py)
# ---------------------------------------------------------------------------
def run_full_evaluation(
    X: np.ndarray,
    algorithms: Optional[dict[str, dict]] = None,
) -> ComparisonTable:
    """Run a predefined suite of clustering algorithms and return results.

    Parameters
    ----------
    X:
        Scaled feature matrix.
    algorithms:
        Optional mapping of ``algorithm_name -> kwargs`` forwarded to
        :func:`src.clustering.run_clustering`. Defaults to a curated set
        of configurations for K-Means, Agglomerative, DBSCAN, and GMM.

    Returns
    -------
    ComparisonTable
        Structured results for downstream consumption.
    """
    from src.clustering import run_clustering

    if algorithms is None:
        algorithms = {
            "kmeans_k2": {"algorithm": "kmeans", "n_clusters": 2},
            "kmeans_k3": {"algorithm": "kmeans", "n_clusters": 3},
            "kmeans_k4": {"algorithm": "kmeans", "n_clusters": 4},
            "kmeans_k5": {"algorithm": "kmeans", "n_clusters": 5},
            "agglomerative_ward_k5": {
                "algorithm": "agglomerative",
                "n_clusters": 5,
                "linkage": "ward",
            },
            "dbscan_eps5": {"algorithm": "dbscan", "eps": 5, "min_samples": 4},
            "gmm_k5": {"algorithm": "gmm", "n_components": 5},
        }

    results: list[ClusterResult] = []
    for name, kwargs in algorithms.items():
        try:
            res = run_clustering(X, **kwargs)
            if res.valid:
                results.append(res)
                logger.info("Evaluated %s: %d clusters", name, res.n_clusters)
            else:
                logger.warning("Skipping %s: no clusters found.", name)
        except Exception as exc:
            logger.error("Failed to run %s: %s", name, exc)

    if not results:
        raise RuntimeError("All candidate algorithms failed to produce valid clusters.")

    return compare_models(X, results)


def save_evaluation_results(table: ComparisonTable, path: Optional[Path] = None) -> Path:
    """Persist the comparison table as CSV for downstream consumption.

    Parameters
    ----------
    table:
        Comparison table to persist.
    path:
        Destination path. Defaults to ``outputs/evaluation_results.csv``.

    Returns
    -------
    Path
        The file path that was written.
    """
    dest = path or FIGURES_DIR.parent / "evaluation_results.csv"
    ensure_directory(dest.parent)
    df = pd.DataFrame(table.rows)
    df.to_csv(dest, index=False)
    logger.info("Saved evaluation results to %s", dest)
    return dest


# ---------------------------------------------------------------------------
# Legacy helpers (Phase 3 — retained for backward compatibility)
# ---------------------------------------------------------------------------
@dataclass
class ElbowResult:
    """Inertia values across a range of ``k``."""

    k_values: list[int] = field(default_factory=list)
    inertias: list[float] = field(default_factory=list)


@dataclass
class SilhouetteResult:
    """Silhouette scores across a range of ``k``."""

    k_values: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    optimal_k: int = 0


def _k_range(k_min: int = MIN_K, k_max: int = MAX_K) -> range:
    return range(k_min, k_max + 1)


def compute_elbow_curve(
    X_scaled: np.ndarray,
    k_range: Optional[Sequence[int]] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = N_INIT,
) -> ElbowResult:
    """Compute K-Means inertia for each ``k`` to form the elbow curve.

    .. deprecated::
        Use :func:`src.clustering.evaluate_k_range` instead.
    """
    import warnings

    warnings.warn(
        "compute_elbow_curve is deprecated; use evaluate_k_range from src.clustering.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.clustering import build_kmeans

    ks = list(k_range) if k_range is not None else list(_k_range())
    inertias: list[float] = []
    for k in ks:
        model = build_kmeans(n_clusters=k, random_state=random_state, n_init=n_init)
        model.fit(X_scaled)
        inertias.append(float(model.inertia_))
    logger.info("Elbow inertias computed for k=%s", ks)
    return ElbowResult(k_values=ks, inertias=inertias)


def compute_silhouette_scores(
    X_scaled: np.ndarray,
    k_range: Optional[Sequence[int]] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = N_INIT,
) -> SilhouetteResult:
    """Compute silhouette scores for each ``k``.

    .. deprecated::
        Use :func:`src.clustering.evaluate_k_range` instead.
    """
    import warnings

    warnings.warn(
        "compute_silhouette_scores is deprecated; use evaluate_k_range from src.clustering.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.clustering import build_kmeans

    ks = list(k_range) if k_range is not None else list(_k_range())
    scores: list[float] = []
    for k in ks:
        model = build_kmeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X_scaled)
        score = float(silhouette_score(X_scaled, labels))
        scores.append(score)
        logger.info("Silhouette score for k=%d: %.4f", k, score)
    optimal_k = ks[int(np.argmax(scores))]
    return SilhouetteResult(
        k_values=ks, scores=scores, optimal_k=optimal_k
    )


def find_optimal_k(silhouette: SilhouetteResult) -> int:
    """Return the ``k`` with the highest silhouette score.

    .. deprecated::
        Use :func:`src.clustering.evaluate_k_range` instead.
    """
    import warnings

    warnings.warn(
        "find_optimal_k is deprecated; use evaluate_k_range from src.clustering.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not silhouette.scores:
        raise ValueError("SilhouetteResult contains no scores.")
    optimal = int(
        silhouette.k_values[int(np.argmax(silhouette.scores))]
    )
    logger.info("Optimal k selected via silhouette: %d", optimal)
    return optimal


def plot_elbow_curve(
    result: ElbowResult,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot and save the elbow curve.

    .. deprecated::
        Use :func:`src.clustering.evaluate_k_range` for data and plot manually.
    """
    import warnings

    warnings.warn(
        "plot_elbow_curve is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.k_values, result.inertias, "bo-", linewidth=2)
    ax.set_title("Elbow Method - Inertia vs. Number of Clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_xticks(result.k_values)
    dest = save_path or FIGURES_DIR / "elbow_method.png"
    ensure_directory(dest.parent)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved elbow plot to %s", dest)
    return dest


def plot_silhouette_scores(
    result: SilhouetteResult,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot and save the silhouette-score curve.

    .. deprecated::
        Use :func:`src.clustering.evaluate_k_range` for data and plot manually.
    """
    import warnings

    warnings.warn(
        "plot_silhouette_scores is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.k_values, result.scores, "go-", linewidth=2)
    ax.axvline(
        result.optimal_k, color="red", linestyle="--", label=f"Optimal k={result.optimal_k}"
    )
    ax.set_title("Silhouette Score vs. Number of Clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(result.k_values)
    ax.legend()
    dest = save_path or FIGURES_DIR / "silhouette_scores.png"
    ensure_directory(dest.parent)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved silhouette plot to %s", dest)
    return dest
