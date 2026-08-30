"""Unified clustering engine supporting K-Means, Agglomerative, DBSCAN, and GMM.

All algorithms expose a consistent interface via :class:`ClusterResult` and the
:func:`run_clustering` dispatcher. The engine accepts preprocessed data, trains
the selected algorithm, returns cluster labels, and exposes model information.
No Streamlit or UI code belongs here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from .utils import (
    ANNUAL_INCOME,
    DEFAULT_N_CLUSTERS,
    DEFAULT_RANDOM_STATE,
    FIGURES_DIR,
    LABEL_MAP,
    MAX_K,
    MIN_K,
    N_INIT,
    SPENDING_SCORE,
    ensure_directory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ClusterResult:
    """Container for clustering output, agnostic to algorithm.

    Attributes
    ----------
    labels:
        Cluster label per sample (``-1`` marks DBSCAN noise).
    model:
        The fitted scikit-learn estimator.
    n_clusters:
        Number of clusters found (excludes noise).
    algorithm:
        Identifier string for the algorithm used.
    cluster_centers:
        Per-cluster mean coordinates (``None`` if unavailable).
    inertia:
        K-Means within-cluster sum of squares (``None`` for other algorithms).
    noise_points:
        Count of points labelled as noise (DBSCAN only).
    feature_names:
        Names of the features used for clustering.
    bic:
        Bayesian Information Criterion (GMM only).
    aic:
        Akaike Information Criterion (GMM only).
    """

    labels: np.ndarray
    model: object
    n_clusters: int
    algorithm: str
    cluster_centers: Optional[np.ndarray] = None
    inertia: Optional[float] = None
    noise_points: int = 0
    feature_names: list[str] = field(default_factory=list)
    bic: Optional[float] = None
    aic: Optional[float] = None

    @property
    def valid(self) -> bool:
        """True if at least one real cluster was found."""
        return self.n_clusters > 0

    @property
    def unique_labels(self) -> np.ndarray:
        """Sorted unique labels (may include ``-1`` for noise)."""
        return np.unique(self.labels)


# Backward-compatible alias for the previous K-Means-only result type.
ClusteringResult = ClusterResult


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_X(X: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D.")
    if X.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples to cluster, got {X.shape[0]}.")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN; preprocess before clustering.")


def _validate_k(n_clusters: int, n_samples: int) -> None:
    if not isinstance(n_clusters, int) or n_clusters < 2:
        raise ValueError(f"n_clusters must be an integer >= 2, got {n_clusters!r}.")
    if n_clusters >= n_samples:
        raise ValueError(
            f"n_clusters ({n_clusters}) must be less than n_samples ({n_samples})."
        )


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
def build_kmeans(
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = N_INIT,
    init: str = "k-means++",
) -> KMeans:
    """Return a configured :class:`~sklearn.cluster.KMeans` estimator."""
    return KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        random_state=random_state,
    )


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = N_INIT,
) -> ClusterResult:
    """Fit K-Means on *X* and return a :class:`ClusterResult`."""
    _validate_X(X)
    _validate_k(n_clusters, X.shape[0])

    model = build_kmeans(
        n_clusters=n_clusters, random_state=random_state, n_init=n_init
    )
    labels = model.fit_predict(X)
    logger.info(
        "K-Means k=%d -> %s",
        n_clusters,
        dict(zip(*np.unique(labels, return_counts=True))),
    )
    return ClusterResult(
        labels=labels,
        model=model,
        n_clusters=n_clusters,
        algorithm="kmeans",
        cluster_centers=model.cluster_centers_,
        inertia=float(model.inertia_),
    )


# ---------------------------------------------------------------------------
# Agglomerative Clustering
# ---------------------------------------------------------------------------
def build_agglomerative(
    n_clusters: int = DEFAULT_N_CLUSTERS,
    linkage: str = "ward",
    metric: Optional[str] = None,
) -> AgglomerativeClustering:
    """Return a configured :class:`~sklearn.cluster.AgglomerativeClustering`."""
    kwargs: dict = {"n_clusters": n_clusters, "linkage": linkage}
    if metric is not None:
        kwargs["metric"] = metric
    return AgglomerativeClustering(**kwargs)


def run_agglomerative(
    X: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    linkage: str = "ward",
    metric: Optional[str] = None,
) -> ClusterResult:
    """Fit Agglomerative Clustering on *X* and return a :class:`ClusterResult`."""
    _validate_X(X)
    _validate_k(n_clusters, X.shape[0])

    valid_linkages = ("ward", "complete", "average", "single")
    if linkage not in valid_linkages:
        raise ValueError(
            f"linkage must be one of {valid_linkages}, got {linkage!r}."
        )
    if linkage == "ward" and metric is not None and metric != "euclidean":
        raise ValueError(
            "ward linkage only supports the euclidean metric; "
            f"got metric={metric!r}."
        )

    model = build_agglomerative(
        n_clusters=n_clusters, linkage=linkage, metric=metric
    )
    labels = model.fit_predict(X)
    centers = _compute_centers(X, labels, n_clusters)
    logger.info(
        "Agglomerative linkage=%s k=%d -> %s",
        linkage,
        n_clusters,
        dict(zip(*np.unique(labels, return_counts=True))),
    )
    return ClusterResult(
        labels=labels,
        model=model,
        n_clusters=n_clusters,
        algorithm=f"agglomerative_{linkage}",
        cluster_centers=centers,
    )


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------
def build_dbscan(
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> DBSCAN:
    """Return a configured :class:`~sklearn.cluster.DBSCAN` estimator."""
    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric)


def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> ClusterResult:
    """Fit DBSCAN on *X* and return a :class:`ClusterResult`."""
    _validate_X(X)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps!r}.")
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples!r}.")

    model = build_dbscan(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X)
    n_noise = int(np.sum(labels == -1))
    unique = set(labels.tolist()) - {-1}
    n_clusters = len(unique)
    centers = _compute_centers(X, labels, n_clusters) if n_clusters > 0 else None
    logger.info(
        "DBSCAN eps=%.2f min_samples=%d -> %d clusters, %d noise",
        eps,
        min_samples,
        n_clusters,
        n_noise,
    )
    return ClusterResult(
        labels=labels,
        model=model,
        n_clusters=n_clusters,
        algorithm="dbscan",
        cluster_centers=centers,
        noise_points=n_noise,
    )


# ---------------------------------------------------------------------------
# Gaussian Mixture Model
# ---------------------------------------------------------------------------
def build_gmm(
    n_components: int = DEFAULT_N_CLUSTERS,
    covariance_type: str = "full",
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = 1,
) -> GaussianMixture:
    """Return a configured :class:`~sklearn.mixture.GaussianMixture`."""
    return GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
    )


def run_gmm(
    X: np.ndarray,
    n_components: int = DEFAULT_N_CLUSTERS,
    covariance_type: str = "full",
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = 1,
) -> ClusterResult:
    """Fit a Gaussian Mixture Model on *X* and return a :class:`ClusterResult`."""
    _validate_X(X)
    _validate_k(n_components, X.shape[0])

    valid_cov = ("full", "tied", "diag", "spherical")
    if covariance_type not in valid_cov:
        raise ValueError(
            f"covariance_type must be one of {valid_cov}, got {covariance_type!r}."
        )

    model = build_gmm(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
    )
    model.fit(X)
    labels = model.predict(X)
    logger.info(
        "GMM components=%d cov=%s -> %s",
        n_components,
        covariance_type,
        dict(zip(*np.unique(labels, return_counts=True))),
    )
    return ClusterResult(
        labels=labels,
        model=model,
        n_clusters=n_components,
        algorithm="gmm",
        cluster_centers=model.means_,
        bic=float(model.bic(X)),
        aic=float(model.aic(X)),
    )


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------
ALGORITHMS = {
    "kmeans": run_kmeans,
    "agglomerative": run_agglomerative,
    "dbscan": run_dbscan,
    "gmm": run_gmm,
}


def run_clustering(
    X: np.ndarray,
    algorithm: str = "kmeans",
    **kwargs,
) -> ClusterResult:
    """Fit the selected algorithm on *X* and return a :class:`ClusterResult`.

    Parameters
    ----------
    X:
        Preprocessed feature matrix (scaled).
    algorithm:
        One of ``'kmeans'``, ``'agglomerative'``, ``'dbscan'``, ``'gmm'``.
    **kwargs:
        Forwarded to the underlying algorithm function.

    Returns
    -------
    ClusterResult
    """
    key = algorithm.lower().replace("-", "_").replace("_", "")
    if key not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. "
            f"Choose from: {', '.join(ALGORITHMS)}."
        )
    return ALGORITHMS[key](X, **kwargs)


# ---------------------------------------------------------------------------
# K-Means optimization: elbow + silhouette analysis across a range of K
# ---------------------------------------------------------------------------
@dataclass
class KOptimizationResult:
    """Output of :func:`evaluate_k_range`."""

    k_values: list[int]
    inertias: list[float]
    silhouette_scores: list[float]
    optimal_k: int


def evaluate_k_range(
    X: np.ndarray,
    k_min: int = MIN_K,
    k_max: int = MAX_K,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = N_INIT,
) -> KOptimizationResult:
    """Run K-Means across a range of *k* and compute elbow + silhouette data.

    Returns a :class:`KOptimizationResult` containing the inertia and silhouette
    score for each *k*, plus the optimal *k* (highest silhouette score).
    """
    _validate_X(X)
    if k_min < 2:
        raise ValueError(f"k_min must be >= 2, got {k_min}.")
    if k_max < k_min:
        raise ValueError(f"k_max ({k_max}) must be >= k_min ({k_min}).")
    if k_max >= X.shape[0]:
        raise ValueError(
            f"k_max ({k_max}) must be less than n_samples ({X.shape[0]})."
        )

    ks = list(range(k_min, k_max + 1))
    inertias: list[float] = []
    scores: list[float] = []
    for k in ks:
        model = build_kmeans(
            n_clusters=k, random_state=random_state, n_init=n_init
        )
        labels = model.fit_predict(X)
        inertias.append(float(model.inertia_))
        scores.append(float(silhouette_score(X, labels)))
    optimal_k = ks[int(np.argmax(scores))]
    logger.info("K optimization: k=%s, optimal_k=%d", ks, optimal_k)
    return KOptimizationResult(
        k_values=ks,
        inertias=inertias,
        silhouette_scores=scores,
        optimal_k=optimal_k,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_centers(
    X: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Compute per-cluster mean centres from labels (ignoring ``-1`` noise)."""
    if n_clusters <= 0:
        return np.empty((0, X.shape[1]))
    centers = np.zeros((n_clusters, X.shape[1]))
    for c in range(n_clusters):
        mask = labels == c
        if np.any(mask):
            centers[c] = X[mask].mean(axis=0)
        else:
            centers[c] = np.nan
    return centers


def labels_to_series(
    labels: np.ndarray, index: Optional[pd.Index] = None
) -> pd.Series:
    """Convert cluster labels into a named :class:`pandas.Series`."""
    return pd.Series(labels, index=index, name="Cluster")


def plot_clusters(
    X_scaled: np.ndarray,
    result: ClusterResult,
    feature_names: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
) -> Path:
    """Scatter-plot the first two (scaled) features coloured by cluster.

    Saves the figure and returns the destination path.
    """
    names = feature_names or result.feature_names or ["Feature 1", "Feature 2"]
    x_label = LABEL_MAP.get(names[0], names[0]) if names else "Feature 1"
    y_label = LABEL_MAP.get(names[1], names[1]) if len(names) > 1 else "Feature 2"

    fig, ax = plt.subplots(figsize=(12, 8))
    unique = result.unique_labels
    n_colors = len(unique)
    palette = plt.cm.Set1(np.linspace(0, 1, max(n_colors, 1)))

    for i, label in enumerate(sorted(unique)):
        mask = result.labels == label
        if label == -1:
            ax.scatter(
                X_scaled[mask, 0],
                X_scaled[mask, 1],
                s=50,
                c="gray",
                marker="x",
                label="Noise",
                edgecolors="black",
                linewidth=0.3,
            )
        else:
            ax.scatter(
                X_scaled[mask, 0],
                X_scaled[mask, 1],
                s=100,
                c=[palette[i]],
                label=f"Cluster {int(label) + 1}",
                edgecolors="black",
                linewidth=0.5,
            )

    if result.cluster_centers is not None:
        valid = result.cluster_centers[
            ~np.isnan(result.cluster_centers).any(axis=1)
        ]
        if len(valid):
            ax.scatter(
                valid[:, 0],
                valid[:, 1],
                s=300,
                c="yellow",
                edgecolors="black",
                linewidth=1.0,
                label="Centroids",
            )

    ax.set_title(f"Clusters of customers ({result.algorithm})")
    ax.set_xlabel(f"{x_label} (scaled)")
    ax.set_ylabel(f"{y_label} (scaled)")
    ax.legend()

    dest = save_path or FIGURES_DIR / "customer_clusters.png"
    ensure_directory(dest.parent)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cluster scatter plot to %s", dest)
    return dest
