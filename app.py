"""Premium Customer Intelligence dashboard for Intelligent Customer Segmentation.

The dashboard is a thin presentation layer: all ML logic (loading, validation,
preprocessing, clustering, evaluation, persona assignment, business insights)
is delegated to the reusable modules in ``src/``. No modelling logic is
duplicated here. Presentation primitives (cards, charts, layout helpers) live in
``src/ui`` so styling is defined once and reused everywhere.

Run with::

    streamlit run app.py

Navigation (sidebar):
    01 Overview          02 Customer Analytics   03 Segmentation Lab
    04 Model Evaluation  05 Segment Explorer      06 Customer Prediction
    07 Export

The prediction model is trained once and persisted with ``joblib``; submitting
the prediction form does not retrain it.
"""

from __future__ import annotations

import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.business_insights import (
    _PERSONA_INSIGHTS,
    build_cluster_profiles,
    generate_cluster_insights,
)
from src.clustering import evaluate_k_range, run_clustering
from src.data_loader import load_data
from src.evaluation import run_full_evaluation
from src.personas import assign_personas_from_data, build_persona_profiles
from src.preprocessing import CustomerDataPreprocessor
from src.ui import cards, charts, components, styles
from src.ui.styles import (
    AMBER, ERROR, PRIMARY, SUCCESS, TEAL,
    WARNING, PURPLE,
    PERSONA_COLORS,
    PALETTE,
)
from src.utils import (
    AGE,
    ANNUAL_INCOME,
    CUSTOMER_ID,
    DEFAULT_DATA_PATH,
    DEFAULT_EPS,
    DEFAULT_LINKAGE,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_RANDOM_STATE,
    FEATURE_COLUMNS,
    GENRE,
    MAX_K,
    MIN_K,
    MODEL_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    SPENDING_SCORE,
    ensure_directory,
    setup_logging,
)

logger = setup_logging()


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------
ALGORITHMS: dict[str, str] = {
    "kmeans": "K-Means",
    "agglomerative": "Agglomerative",
    "dbscan": "DBSCAN",
    "gmm": "Gaussian Mixture",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "algorithm": "kmeans",
    "n_clusters": 5,
    "linkage": DEFAULT_LINKAGE,
    "eps": DEFAULT_EPS,
    "min_samples": DEFAULT_MIN_SAMPLES,
}

# Business-friendly column display names.
DISPLAY: dict[str, str] = {
    AGE: "Age",
    ANNUAL_INCOME: "Annual Income (k$)",
    SPENDING_SCORE: "Spending Score (1-100)",
    GENRE: "Gender",
}

# Sidebar / page navigation (numbered journey).
NAV: list[tuple[str, str]] = [
    ("overview", "01 Overview"),
    ("analytics", "02 Customer Analytics"),
    ("lab", "03 Segmentation Lab"),
    ("evaluation", "04 Model Evaluation"),
    ("explorer", "05 Segment Explorer"),
    ("prediction", "06 Customer Prediction"),
    ("export", "07 Export"),
]
NAV_LABELS: dict[str, str] = {k: v for k, v in NAV}

# Metric glossary for the evaluation screen.
METRIC_GLOSSARY: list[tuple[str, str, str]] = [
    (
        "Silhouette Score",
        "Higher is better",
        "Measures how tightly grouped and how far apart the segments are. "
        "It compares each customer's distance to its own segment centre with its "
        "distance to the next nearest centre. Ranges from -1 (overlapping) to "
        "1 (perfectly distinct); values above ~0.5 are good. Needs at least "
        "two segments.",
    ),
    (
        "Davies-Bouldin Index",
        "Lower is better",
        "The average 'similarity' of each segment to its most similar neighbour, "
        "where similarity combines spread and separation. A lower value means "
        "segments are compact and well separated; 0 would be ideal.",
    ),
    (
        "Calinski-Harabasz Index",
        "Higher is better",
        "The ratio of separation between segments to spread within segments. A "
        "higher value indicates denser, better-separated segments. There is no "
        "fixed upper bound, so it is interpreted relative to other configurations.",
    ),
    (
        "Inertia (WCSS)",
        "Lower is better (K-Means only)",
        "The sum of squared distances from each customer to its segment centre. "
        "Lower means customers sit closer to their centre. It always falls as k "
        "increases, so it is used for the 'elbow' shape rather than comparing "
        "different values of k directly.",
    ),
]


def _suppress_warnings():
    """Context manager that suppresses sklearn/convergence warnings."""
    return warnings.catch_warnings()


def alg_display(alg: str) -> str:
    """Human-readable algorithm name, expanding linkage/covariance suffixes."""
    parts = alg.split("_", 1)
    label = ALGORITHMS.get(parts[0], parts[0].replace("_", " ").title())
    if len(parts) > 1:
        return f"{label} ({parts[1]})"
    return label


def cfg_label(cfg: dict[str, Any]) -> str:
    """Human-readable description of a clustering configuration."""
    alg = ALGORITHMS.get(cfg["algorithm"], cfg["algorithm"])
    if cfg["algorithm"] == "dbscan":
        return f"{alg} (eps={cfg['eps']}, min_samples={cfg['min_samples']})"
    if cfg["algorithm"] == "agglomerative":
        return f"{alg} (k={cfg['n_clusters']}, linkage={cfg['linkage']})"
    return f"{alg} (k={cfg['n_clusters']})"


# ===========================================================================
# Cached data access
# ===========================================================================
@st.cache_resource(show_spinner="Loading dataset...")
def get_base() -> dict[str, Any]:
    """Load + preprocess the dataset once per session.

    Returns a dict with either ``error`` (message) or the raw dataframe, the
    scaled feature matrix ``X``, the cleaned dataframe, and the fitted
    preprocessor.
    """
    try:
        raw = load_data(DEFAULT_DATA_PATH)
        logger.info(
            "Dataset loaded successfully: %d rows, %d columns",
            len(raw), len(raw.columns),
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        logger.error("Failed to load dataset: %s", exc)
        return {"error": str(exc)}

    pre = CustomerDataPreprocessor(
        numeric_features=FEATURE_COLUMNS,
        categorical_features=[],
        drop_duplicates=True,
    )
    with _suppress_warnings():
        warnings.simplefilter("ignore")
        X, cleaned = pre.fit_transform(raw)
    logger.info(
        "Preprocessing complete: %d samples, %d features", X.shape[0], X.shape[1]
    )
    return {"raw": raw, "X": X, "cleaned": cleaned, "pre": pre, "error": None}


def _build_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map the UI configuration into kwargs for ``run_clustering``."""
    alg = cfg["algorithm"]
    if alg == "kmeans":
        return {"n_clusters": cfg["n_clusters"], "random_state": DEFAULT_RANDOM_STATE}
    if alg == "agglomerative":
        return {"n_clusters": cfg["n_clusters"], "linkage": cfg["linkage"]}
    if alg == "dbscan":
        return {"eps": cfg["eps"], "min_samples": cfg["min_samples"]}
    if alg == "gmm":
        return {"n_components": cfg["n_clusters"], "random_state": DEFAULT_RANDOM_STATE}
    return {}


def _cfg_key(cfg: dict[str, Any]) -> tuple:
    return (
        cfg["algorithm"],
        cfg["n_clusters"],
        cfg["linkage"],
        round(float(cfg["eps"]), 6),
        cfg["min_samples"],
    )


def compute_segmentation(base: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the active clustering configuration and derive personas/profiles/analytics."""
    from src.analytics import (
        compare_segments_vs_overall,
        compute_cluster_stability,
        compute_feature_separation,
        generate_analytical_insights,
    )

    X = base["X"]
    cleaned = base["cleaned"]
    pre = base["pre"]

    with _suppress_warnings():
        warnings.simplefilter("ignore")
        try:
            result = run_clustering(X, algorithm=cfg["algorithm"], **_build_kwargs(cfg))
        except Exception as exc:  # noqa: BLE001
            logger.error("Clustering failed for %s: %s", cfg["algorithm"], exc)
            return {"error": str(exc), "algorithm": cfg["algorithm"]}

    result.feature_names = pre.output_feature_names
    logger.info(
        "Clustering complete: algorithm=%s, n_clusters=%d, valid=%s",
        result.algorithm, result.n_clusters, result.valid,
    )

    centers_orig: pd.DataFrame | None = None
    if result.cluster_centers is not None:
        try:
            centers_orig = pre.inverse_transform_centers(result.cluster_centers)
            logger.info(
                "Inverse-transformed %d cluster centres to original scale",
                len(centers_orig),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to inverse-transform centres: %s", exc)
            centers_orig = None

    personas: dict[int, Any] = {}
    if centers_orig is not None and len(centers_orig):
        try:
            personas = assign_personas_from_data(centers_orig, cleaned)
            logger.info("Assigned %d personas to clusters", len(personas))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Persona assignment failed: %s", exc)
            personas = {}

    from src.evaluation import evaluate_model

    with _suppress_warnings():
        warnings.simplefilter("ignore")
        eval_res = evaluate_model(X, result)
    metrics_map = eval_res.metrics
    sil = metrics_map["Silhouette Score"].value
    db = metrics_map["Davies-Bouldin Score"].value
    ch = metrics_map["Calinski-Harabasz Score"].value

    profiles: list[Any] = []
    profiles_df: pd.DataFrame | None = None
    if personas:
        profiles = build_persona_profiles(
            cleaned, result.labels, cluster_centers=centers_orig, personas=personas
        )
        try:
            profiles_df = build_cluster_profiles(cleaned, result.labels)
        except Exception:  # noqa: BLE001
            profiles_df = None

    # Analytics
    segment_comparison: list[Any] = []
    feature_separation: list[Any] = []
    stability: Any = None
    analytical_insights: list[Any] = []

    if profiles_df is not None and not profiles_df.empty:
        try:
            segment_comparison = compare_segments_vs_overall(cleaned, result.labels)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Segment comparison failed: %s", exc)
        try:
            feature_separation = compute_feature_separation(cleaned, result.labels)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Feature separation failed: %s", exc)
        if cfg["algorithm"] == "kmeans":
            try:
                stability = compute_cluster_stability(
                    X, n_clusters=result.n_clusters
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cluster stability analysis failed: %s", exc)
        try:
            analytical_insights = generate_analytical_insights(
                profiles_df,
                centers_orig if centers_orig is not None else pd.DataFrame(),
                result.labels,
                cleaned,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Analytical insights generation failed: %s", exc)

    return {
        "error": None,
        "labels": result.labels,
        "n_clusters": result.n_clusters,
        "algorithm": result.algorithm,
        "centers_orig": centers_orig,
        "personas": personas,
        "profiles": profiles,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "noise": result.noise_points,
        "valid": result.valid,
        "inertia": result.inertia,
        "profiles_df": profiles_df,
        "segment_comparison": segment_comparison,
        "feature_separation": feature_separation,
        "stability": stability,
        "analytical_insights": analytical_insights,
    }


def get_active(base: dict[str, Any]) -> dict[str, Any]:
    """Return the segmentation for the currently selected configuration.

    Results are cached in ``session_state`` and only recomputed when the
    configuration changes, so unrelated reruns stay cheap.
    """
    cfg = st.session_state.cfg
    key = _cfg_key(cfg)
    if (
        st.session_state.get("active_key") != key
        or st.session_state.get("active") is None
    ):
        st.session_state.active = compute_segmentation(base, cfg)
        st.session_state.active_key = key
    return st.session_state.active


# ===========================================================================
# Cached evaluation runs (avoids recomputation across tab visits / reruns)
# ===========================================================================
@st.cache_data(show_spinner=False, max_entries=8)
def _eval_table(X: Any) -> Any:
    """Cached full model comparison table."""
    with _suppress_warnings():
        warnings.simplefilter("ignore")
        return run_full_evaluation(X)


@st.cache_data(show_spinner=False, max_entries=8)
def _k_analysis(X: Any) -> Any:
    """Cached K-Means elbow + silhouette analysis."""
    with _suppress_warnings():
        warnings.simplefilter("ignore")
        return evaluate_k_range(X)


# ===========================================================================
# Prediction model (persisted with joblib - never retrained per submit)
# ===========================================================================
def train_bundle(base: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Train and return a portable model bundle for new-customer prediction."""
    pre = base["pre"]
    with _suppress_warnings():
        warnings.simplefilter("ignore")
        result = run_clustering(
            base["X"], algorithm=cfg["algorithm"], **_build_kwargs(cfg)
        )
    centers_orig = None
    scaled_centers = result.cluster_centers
    if scaled_centers is not None:
        try:
            centers_orig = pre.inverse_transform_centers(scaled_centers)
        except Exception:  # noqa: BLE001
            centers_orig = None
    personas = {}
    if centers_orig is not None and len(centers_orig):
        personas = assign_personas_from_data(centers_orig, base["cleaned"])
    bundle = {
        "pre": pre,
        "scaled_centers": scaled_centers,
        "centers_orig": centers_orig,
        "personas": personas,
        "algorithm": cfg["algorithm"],
        "config": dict(cfg),
        "model_version": MODEL_VERSION,
    }
    logger.info(
        "Trained prediction bundle: algorithm=%s, n_clusters=%d, personas=%d",
        cfg["algorithm"], result.n_clusters, len(personas),
    )
    return bundle


def _validate_bundle(bundle: Any) -> str | None:
    """Validate a loaded prediction bundle.

    Returns ``None`` if valid, otherwise a human-readable reason describing the
    first failure. Validates artifact structure, version, preprocessing config,
    feature names, algorithm, cluster count, and persona mapping.
    """
    if not isinstance(bundle, dict):
        return f"bundle is not a dict (got {type(bundle).__name__})"
    required = {"pre", "scaled_centers", "centers_orig", "personas",
               "algorithm", "config", "model_version"}
    missing = required - set(bundle.keys())
    if missing:
        return f"missing keys: {sorted(missing)}"
    if bundle.get("model_version") != MODEL_VERSION:
        return (
            f"model_version mismatch: artifact={bundle.get('model_version')!r}, "
            f"current={MODEL_VERSION!r}"
        )
    pre = bundle.get("pre")
    if pre is None or getattr(pre, "transformer_", None) is None:
        return "preprocessor is not fitted"
    try:
        feat_names = list(pre.output_feature_names)
    except Exception as exc:  # noqa: BLE001
        return f"cannot read preprocessor feature names: {exc}"
    expected = list(FEATURE_COLUMNS)
    if feat_names != expected:
        return (
            f"feature names mismatch: artifact={feat_names}, current={expected}"
        )
    if bundle.get("algorithm") not in ALGORITHMS:
        return f"unknown algorithm in bundle: {bundle.get('algorithm')!r}"
    centers = bundle.get("scaled_centers")
    if centers is None or len(centers) == 0:
        return "bundle has no cluster centres"
    n_centers = int(len(centers))
    personas = bundle.get("personas") or {}
    if n_centers != len(personas):
        return (
            f"cluster/persona count mismatch: centers={n_centers}, "
            f"personas={len(personas)}"
        )
    return None


@st.cache_resource(show_spinner="Loading prediction model...")
def load_prediction_model() -> dict[str, Any] | None:
    """Load the persisted prediction model, training + saving it if absent."""
    base = get_base()
    if base.get("error"):
        return None
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load model artifact %s: %s", MODEL_PATH, exc)
            bundle = None
        else:
            reason = _validate_bundle(bundle)
            if reason is None:
                logger.info(
                    "Loaded prediction model from %s (version %s)",
                    MODEL_PATH.name, bundle.get("model_version"),
                )
                return bundle
            logger.warning(
                "Discarding incompatible model artifact %s: %s",
                MODEL_PATH.name, reason,
            )
            bundle = None
    else:
        logger.warning(
            "No model artifact found at %s; training new bundle at runtime.",
            MODEL_PATH,
        )
        bundle = None

    if bundle is None:
        bundle = train_bundle(base, DEFAULT_CONFIG)
        ensure_directory(MODELS_DIR)
        joblib.dump(bundle, MODEL_PATH)
        logger.info("Saved new prediction model to %s", MODEL_PATH.name)
    return bundle


def predict_new(
    bundle: dict[str, Any], age: int, genre: str, income: float, spending: float
) -> dict[str, Any] | None:
    """Map a new customer to a cluster + persona without retraining."""
    if not bundle or not bundle.get("personas"):
        logger.warning("Prediction attempted with empty bundle or no personas.")
        return None
    row = pd.DataFrame(
        [
            {
                CUSTOMER_ID: -1,
                GENRE: genre,
                AGE: age,
                ANNUAL_INCOME: income,
                SPENDING_SCORE: spending,
            }
        ]
    )
    try:
        Xs = bundle["pre"].transform(row)
    except Exception as exc:
        logger.error("Prediction transform failed: %s", exc)
        return None
    centers = bundle["scaled_centers"]
    if centers is None or len(centers) == 0:
        logger.warning("Prediction attempted with no cluster centres.")
        return None
    valid = ~np.isnan(centers).any(axis=1)
    if not np.any(valid):
        logger.warning("All cluster centres contain NaN; cannot predict.")
        return None
    c = centers[valid]
    dists = np.linalg.norm(c - Xs, axis=1)
    local = int(np.argmin(dists))
    cluster_id = int(np.where(valid)[0][local])
    persona = bundle["personas"].get(cluster_id)
    return {
        "cluster_id": cluster_id,
        "persona": persona,
        "distance": float(dists[local]),
    }


# ===========================================================================
# Presentation helpers (delegating to the ui package + small local helpers)
# ===========================================================================
def cluster_colors(active: dict[str, Any]) -> dict[int, str]:
    """Map each cluster id to its persona colour for visual consistency."""
    personas = active.get("personas") or {}
    return {
        cid: PERSONA_COLORS.get(p.key, "#94a3b8") for cid, p in personas.items()
    }


def _segment_size_df(active: dict[str, Any]) -> pd.DataFrame:
    """Build a segment-size dataframe with persona names for charting."""
    personas = active.get("personas") or {}
    labels = active.get("labels")
    if labels is None:
        return pd.DataFrame(columns=["Segment", "Persona", "Customers"])
    counts = pd.Series(labels).value_counts().sort_index()
    rows = []
    for cid, cnt in counts.items():
        if int(cid) == -1:
            name = "Noise"
        elif personas and int(cid) in personas:
            name = personas[int(cid)].name
        else:
            name = f"Segment {cid}"
        rows.append({"Segment": int(cid), "Persona": name, "Customers": int(cnt)})
    return pd.DataFrame(rows)


def _palette_for_labels(labels: list[int], personas: dict[int, Any]) -> list[str]:
    """Return persona colours in the order of *labels*."""
    colors = []
    for cid in labels:
        p = personas.get(cid) if personas else None
        colors.append(PERSONA_COLORS.get(p.key, "#94a3b8") if p else "#94a3b8")
    return colors


def derive_headline(active: dict[str, Any]) -> tuple[str, str]:
    """Return (headline, plain-language interpretation) for the active result.

    The headline is derived from the segments actually present so it stays
    correct when the configuration changes.
    """
    personas = active.get("personas") or {}
    profiles = active.get("profiles") or []
    centers = active.get("centers_orig")

    if not personas or not profiles:
        return (
            "No segments are available for the current configuration.",
            "Adjust the algorithm and parameters in the Segmentation Lab, then "
            "apply a configuration that produces at least two valid clusters.",
        )

    by_count = sorted(profiles, key=lambda p: p.customer_count, reverse=True)
    largest = by_count[0]
    largest_persona = personas.get(largest.cluster_id)

    # Highest-value opportunity: a High-Value (vip) segment if present,
    # otherwise the segment with the greatest combined income + spending.
    vip = next(
        (
            p for p in profiles
            if personas.get(p.cluster_id, None)
            and personas[p.cluster_id].key == "vip"
        ),
        None,
    )
    if vip is not None:
        value_seg = vip
    else:
        def _score(p: Any) -> float:
            if centers is None or p.cluster_id not in centers.index:
                return 0.0
            row = centers.loc[p.cluster_id]
            return float(row.get(ANNUAL_INCOME, 0)) + float(row.get(SPENDING_SCORE, 0))
        value_seg = max(profiles, key=_score)

    value_persona = personas.get(value_seg.cluster_id)
    inc = (
        float(centers.loc[value_seg.cluster_id, ANNUAL_INCOME])
        if centers is not None else float("nan")
    )
    spd = (
        float(centers.loc[value_seg.cluster_id, SPENDING_SCORE])
        if centers is not None else float("nan")
    )

    seg_name = (
        largest_persona.name if largest_persona else f"Segment {largest.cluster_id}"
    )
    val_name = (
        value_persona.name
        if value_persona else f"Segment {value_seg.cluster_id}"
    )
    headline = (
        f"<b>{seg_name}</b> is the largest group at "
        f"<b>{largest.percentage}%</b> of the base "
        f"({largest.customer_count:,} customers). "
        f"<b>{val_name}</b> is the highest-value opportunity, averaging "
        f"${inc:.1f}k income and a spending score of {spd:.0f}."
    )

    total = (
        len(active.get("labels", [])) if active.get("labels") is not None else 0
    )
    interpretation = (
        f"Customer bases are rarely uniform. The segmentation splits the "
        f"{total:,} customers into distinct groups so each team can act on the "
        f"right audience. The largest segment tells you where most of your "
        f"volume sits; the highest-value segment tells you where a small, "
        f"well-targeted premium offer can return the most revenue. Use the "
        f"Segment Explorer to read each group's profile and recommended action."
    )
    return headline, interpretation


def _sidebar_controls(cfg: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """Render configuration controls in the sidebar.

    Returns the (possibly modified) configuration. Changes are only committed
    to ``session_state.cfg`` when the user clicks *Apply*.
    """
    n_samples = len(base["cleaned"])
    max_valid_k = max(MIN_K, min(MAX_K, max(1, n_samples // 5)))
    slider_max = max(cfg["n_clusters"], max_valid_k)

    st.sidebar.subheader("Configuration")
    st.sidebar.caption("Technical controls. Apply to update every view.")

    algorithm = st.sidebar.selectbox(
        "Algorithm",
        options=list(ALGORITHMS.keys()),
        format_func=lambda a: ALGORITHMS[a],
        index=list(ALGORITHMS.keys()).index(cfg["algorithm"]),
    )

    params: dict[str, Any] = {"algorithm": algorithm}
    if algorithm in ("kmeans", "agglomerative", "gmm"):
        params["n_clusters"] = st.sidebar.slider(
            "Number of segments (k)",
            min_value=MIN_K,
            max_value=slider_max,
            value=min(max(cfg["n_clusters"], MIN_K), slider_max),
            help=(
                f"Technically valid for {n_samples} samples: 2-{max_valid_k}. "
                "Analytically recommended: see Model Evaluation "
                "(typically 5-6 for this dataset)."
            ),
        )
        st.sidebar.caption(
            f"Valid range: 2-{max_valid_k} for {n_samples} customers."
        )
    if algorithm == "agglomerative":
        params["linkage"] = st.sidebar.selectbox(
            "Linkage method",
            ["ward", "complete", "average", "single"],
            index=["ward", "complete", "average", "single"].index(cfg["linkage"]),
            help="How distance between clusters is measured when merging them.",
        )
    if algorithm == "dbscan":
        params["eps"] = st.sidebar.number_input(
            "Neighbourhood radius (eps)",
            min_value=0.05, max_value=5.0,
            value=float(cfg["eps"]), step=0.05,
            help="Maximum distance between points in the same dense region.",
        )
        params["min_samples"] = st.sidebar.number_input(
            "Minimum samples",
            min_value=1, max_value=50,
            value=int(cfg["min_samples"]), step=1,
            help="Fewest points needed to form a dense region.",
        )

    # Carry over unchanged fields.
    params.setdefault("n_clusters", cfg["n_clusters"])
    params.setdefault("linkage", cfg["linkage"])
    params.setdefault("eps", cfg["eps"])
    params.setdefault("min_samples", cfg["min_samples"])

    st.sidebar.markdown("")
    applied = st.sidebar.button(
        "Apply Configuration", type="primary",
        help="Recompute segments for this configuration.",
    )
    if applied:
        st.session_state.cfg = params
        st.session_state.active = None
        st.session_state.active_key = None
        st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("Reload data", help="Clear cache and reload the dataset."):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.active = None
        st.session_state.active_key = None
        st.session_state.last_prediction = None
        st.rerun()

    return st.session_state.cfg


# ===========================================================================
# Section 1 - Executive Overview
# ===========================================================================
def section_overview(base: dict[str, Any], active: dict[str, Any]) -> None:
    components.section_header(
        "Customer Intelligence Overview",
        "Discover behavioural patterns, understand customer segments, and turn "
        "clustering results into actionable insights.",
    )

    raw = base["raw"]
    cfg = st.session_state.cfg

    if active.get("error"):
        components.error_state(
            "The selected configuration failed.", active.get("error", "")
        )
        return

    # ---- Hero + headline ----
    st.markdown(
        f"""
        <div class="hero-box">
            <div class="hero-box-tag">Most important business insight</div>
            <div class="hero-box-text">{derive_headline(active)[0]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("What does this mean?"):
        st.markdown(derive_headline(active)[1])
        st.markdown(
            "- **Segments** are groups of customers with similar income and "
            "spending behaviour, so campaigns can be tailored instead of "
            "one-size-fits-all.\n"
            "- **Silhouette** (0 to 1) measures how clearly separated the "
            "segments are — higher means tighter, more distinct groups.\n"
            "- **Average income / spending** set the financial scale of each "
            "group and guide how premium an offer should be."
        )

    # ---- KPI row (2x2 grid for readability) ----
    st.markdown("##### Key Metrics", help="")
    k1, k2 = st.columns(2)
    with k1:
        cards.kpi_card("Total Customers", f"{len(raw):,}", "After deduplication", PRIMARY)
        avg_income = float(raw[ANNUAL_INCOME].mean())
        cards.kpi_card(
            "Average Income", styles.fmt_income(avg_income),
            "Across all customers", WARNING,
        )
    with k2:
        n_clusters = active.get("n_clusters", 0)
        sub = (
            f"{active.get('noise', 0)} noise points" if cfg["algorithm"] == "dbscan"
            else "Distinct groups found"
        )
        cards.kpi_card("Segments", str(n_clusters), sub, SUCCESS)
        avg_spend = float(raw[SPENDING_SCORE].mean())
        cards.kpi_card(
            "Average Spending", f"{avg_spend:,.1f}",
            "Score (1-100)", PURPLE,
        )

    # ---- Model configuration card ----
    st.markdown("###### Active Configuration")
    m1, m2 = st.columns([2, 1])
    with m1:
        cards.kpi_card(
            "Selected Model", alg_display(cfg["algorithm"]),
            "Configurable in the sidebar", TEAL,
        )
    with m2:
        sil = active.get("silhouette")
        cards.kpi_card(
            "Clustering Quality", cfg_label(cfg),
            f"Silhouette: {styles.fmt_metric(sil, 3)}",
            AMBER,
        )

    # ---- Segment distribution ----
    st.markdown("##### Segment Distribution")
    st.caption("Customers split by the active segmentation.")
    seg_df = _segment_size_df(active)
    if not seg_df.empty:
        personas = active.get("personas") or {}
        color_map = {}
        for cid, _cnt in pd.Series(active["labels"]).value_counts().sort_index().items():
            p = personas.get(int(cid))
            color_map[int(cid)] = PERSONA_COLORS.get(p.key, "#94a3b8") if p else "#94a3b8"
        fig = px.bar(
            seg_df, x="Persona", y="Customers",
            color="Segment",
            color_discrete_map={k: v for k, v in color_map.items()},
            title="",
        )
        fig.update_traces(
            hovertemplate="%{x}<br>Customers: %{y}<extra></extra>",
            marker_line_color="white", marker_line_width=1,
        )
        fig = charts.style_chart(
            fig, xlabel="Segment", ylabel="Customers", height=380,
        )
        st.plotly_chart(fig, width="stretch")

    # ---- Income vs Spending scatter (primary view) ----
    st.markdown("##### Income vs Spending")
    if active.get("labels") is not None and active.get("valid"):
        plot_df = raw[[AGE, ANNUAL_INCOME, SPENDING_SCORE, GENRE]].copy()
        plot_df["Segment"] = active["labels"]
        centers_orig = active.get("centers_orig")
        has_centers = (
            centers_orig is not None and len(centers_orig)
            and ANNUAL_INCOME in centers_orig.columns
        )
        fig = charts.scatter_segments(
            plot_df, ANNUAL_INCOME, SPENDING_SCORE, "Segment",
            "Customers by segment (income vs spending)",
            color_map=cluster_colors(active),
            hover_data=[AGE, GENRE],
            hover_labels=["Age", "Gender"],
            centers_x=(centers_orig[ANNUAL_INCOME] if has_centers else None),
            centers_y=(centers_orig[SPENDING_SCORE] if has_centers else None),
            xlabel="Annual Income (k$)", ylabel="Spending Score (1-100)",
            height=420,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        components.info_box(
            "No valid segments have been produced for the current configuration. "
            "Choose an algorithm and click **Apply Configuration** in the sidebar.",
        )

    # ---- Executive insights ----
    st.markdown("##### Executive Insights")
    st.caption("Data-driven observations from the active segmentation.")
    insights = active.get("analytical_insights") or []
    if insights:
        personas = active.get("personas") or {}
        cols = st.columns(min(len(insights), 4))
        for i, ins in enumerate(insights[:4]):
            with cols[i % len(cols)]:
                color = PRIMARY
                if ins.segment_id is not None:
                    p = personas.get(int(ins.segment_id))
                    if p:
                        color = PERSONA_COLORS.get(p.key, PRIMARY)
                cards.insight_card(ins.title, ins.detail, color=color)
    else:
        components.info_box(
            "Analytical insights are not available for this configuration."
        )


# ===========================================================================
# Section 2 - Customer Analytics
# ===========================================================================
def section_analytics(base: dict[str, Any]) -> None:
    components.section_header(
        "Customer Analytics",
        "Interactive, exploratory views of the underlying customer base.",
    )

    raw = base["raw"]

    st.markdown("##### Demographics")
    d1, d2 = st.columns(2)
    with d1:
        fig = px.histogram(
            raw, x=AGE, nbins=20, color_discrete_sequence=[PRIMARY],
            title="Age Distribution",
        )
        fig.update_traces(hovertemplate="Age: %{x}<br>Customers: %{y}<extra></extra>")
        fig = charts.style_chart(fig, xlabel="Age (years)", ylabel="Customers")
        st.plotly_chart(fig, width="stretch")
    with d2:
        counts = raw[GENRE].value_counts().reset_index()
        counts.columns = [GENRE, "Count"]
        fig = px.histogram(
            raw, x=GENRE, nbins=2, color_discrete_sequence=PALETTE,
            title="Customer Composition by Gender",
        )
        fig.update_traces(
            hovertemplate="%{x}<br>Customers: %{y}<extra></extra>",
        )
        fig = charts.style_chart(fig, xlabel="Gender", ylabel="Customers")
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Financial Profile")
    f1, f2 = st.columns(2)
    with f1:
        fig = px.histogram(
            raw, x=ANNUAL_INCOME, nbins=20, color_discrete_sequence=[WARNING],
            title="Annual Income Distribution",
        )
        fig.update_traces(hovertemplate="Income: %{x} k$<br>Customers: %{y}<extra></extra>")
        fig = charts.style_chart(fig, xlabel="Annual Income (k$)", ylabel="Customers")
        st.plotly_chart(fig, width="stretch")
    with f2:
        fig = px.histogram(
            raw, x=SPENDING_SCORE, nbins=20, color_discrete_sequence=[PURPLE],
            title="Spending Score Distribution",
        )
        fig.update_traces(hovertemplate="Score: %{x}<br>Customers: %{y}<extra></extra>")
        fig = charts.style_chart(
            fig, xlabel="Spending Score (1-100)", ylabel="Customers"
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Relationships")
    st.caption("How the core numeric features inter-relate.")
    r1, r2 = st.columns(2)
    with r1:
        corr = raw[[AGE, ANNUAL_INCOME, SPENDING_SCORE]].corr().round(2)
        corr.columns = ["Age", "Income", "Spending"]
        corr.index = ["Age", "Income", "Spending"]
        fig = charts.correlation_heatmap(corr, title="Feature Correlation")
        st.plotly_chart(fig, width="stretch")
    with r2:
        rel_mode = st.radio(
            "Colour points by",
            ["Income vs Spending by Gender", "Income vs Spending by Segment"],
            horizontal=True,
            label_visibility="collapsed",
        )
        active = st.session_state.active
        if rel_mode.endswith("by Segment"):
            if active.get("labels") is not None and active.get("valid"):
                plot_df = raw[[AGE, ANNUAL_INCOME, SPENDING_SCORE, GENRE]].copy()
                plot_df["Segment"] = active["labels"]
                fig = charts.scatter_segments(
                    plot_df, ANNUAL_INCOME, SPENDING_SCORE, "Segment",
                    "Income vs Spending by Segment",
                    color_map=cluster_colors(active),
                    hover_data=[AGE, GENRE],
                    hover_labels=["Age", "Gender"],
                    xlabel="Annual Income (k$)",
                    ylabel="Spending Score (1-100)", height=380,
                )
            else:
                fig = charts.scatter_segments(
                    raw, ANNUAL_INCOME, SPENDING_SCORE, GENRE,
                    "Annual Income vs Spending Score",
                    hover_data=[AGE], hover_labels=["Age"],
                    xlabel="Annual Income (k$)",
                    ylabel="Spending Score (1-100)", height=380,
                )
        else:
            fig = charts.scatter_segments(
                raw, ANNUAL_INCOME, SPENDING_SCORE, GENRE,
                "Income vs Spending by Gender",
                hover_data=[AGE], hover_labels=["Age"],
                color_map={
                    g: PALETTE[i % len(PALETTE)]
                    for i, g in enumerate(sorted(raw[GENRE].unique()))
                },
                xlabel="Annual Income (k$)",
                ylabel="Spending Score (1-100)", height=380,
            )
        st.plotly_chart(fig, width="stretch")

    with st.expander("3-D segment view (Age, Income, Spending)"):
        st.caption(
            "Three-dimensional view of all numeric features. Rotate by dragging."
        )
        active = st.session_state.active
        raw_3d = raw[[AGE, ANNUAL_INCOME, SPENDING_SCORE]].copy()
        raw_3d["Segment"] = active["labels"] if active.get("labels") is not None else -1
        raw_3d["SegmentLabel"] = raw_3d["Segment"].apply(
            lambda s: "Noise" if s == -1 else f"Segment {s}"
        )
        fig3d = px.scatter_3d(
            raw_3d, x=AGE, y=ANNUAL_INCOME, z=SPENDING_SCORE,
            color="SegmentLabel", title="Customer Segments in 3D",
            color_discrete_sequence=PALETTE, opacity=0.85,
        )
        fig3d.update_traces(
            marker=dict(size=5, line=dict(width=0.3, color="white"))
        )
        fig3d = charts.style_chart(fig3d, height=500)
        fig3d.update_layout(
            scene=dict(
                xaxis_title="Age",
                yaxis_title="Annual Income (k$)",
                zaxis_title="Spending Score (1-100)",
            ),
        )
        st.plotly_chart(fig3d, width="stretch")


# ===========================================================================
# Section 3 - Segmentation Lab
# ===========================================================================
_ALGO_BLURBS: dict[str, str] = {
    "kmeans": (
        "K-Means partitions customers into a fixed number of groups by placing "
        "each customer in the nearest centre, then moving the centres to best "
        "fit their group. It is fast and easy to explain, but assumes roughly "
        "round, similarly sized groups."
    ),
    "agglomerative": (
        "Agglomerative clustering starts with every customer as its own group "
        "and repeatedly merges the closest pairs. The linkage method controls "
        "how 'closeness' is measured, which shapes the final segment shapes."
    ),
    "dbscan": (
        "DBSCAN finds dense regions of customers and leaves sparse areas as "
        "noise instead of forcing every customer into a segment. It does not "
        "need a fixed number of segments, but is sensitive to eps and "
        "min_samples on this two-feature data."
    ),
    "gmm": (
        "Gaussian Mixture Modelling fits overlapping bell-shaped distributions "
        "to the customers and assigns each a probability of belonging to each "
        "group. It captures softer, overlapping segments better than K-Means."
    ),
}


def _active_metrics_df(active: dict[str, Any]) -> pd.DataFrame:
    """Build a small metrics table for the active configuration."""
    rows = []
    if active.get("silhouette") is not None:
        rows.append(("Silhouette Score", f"{active['silhouette']:.4f}"))
    if active.get("inertia") is not None:
        rows.append(("Inertia (WCSS)", f"{active['inertia']:.2f}"))
    rows.append(("Segments (excl. noise)", str(int(active.get("n_clusters", 0)))))
    if active.get("algorithm", "") == "dbscan" or active.get("noise", 0):
        rows.append(("Noise points", str(int(active.get("noise", 0)))))
    if active.get("davies_bouldin") is not None:
        rows.append(("Davies-Bouldin", f"{active['davies_bouldin']:.4f}"))
    if active.get("calinski_harabasz") is not None:
        rows.append(("Calinski-Harabasz", f"{active['calinski_harabasz']:.2f}"))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def section_lab(base: dict[str, Any], active: dict[str, Any]) -> None:
    components.section_header(
        "Segmentation Lab",
        "Review the active configuration, the resulting segments, and what the "
        "configuration means for the business.",
    )

    cfg = st.session_state.cfg

    if active.get("error"):
        components.error_state(
            "The selected configuration failed.", active.get("error", "")
        )
        return

    if not active["valid"]:
        if cfg["algorithm"] == "dbscan":
            components.error_state(
                "DBSCAN produced no valid clusters.",
                "All points were marked as noise. Increase eps or decrease "
                "min_samples so denser regions form clusters. Adjust the "
                "parameters in the sidebar.",
            )
        else:
            components.error_state(
                "No valid clusters were produced for this configuration."
            )
        return

    # ------------------------------------------------------------------ #
    # Layer 1 — Configuration summary
    # ------------------------------------------------------------------ #
    st.markdown("##### Model Configuration")
    st.caption(
        "The live configuration is controlled from the sidebar. This summary "
        "reflects the active model and its parameters."
    )
    cfg_lines = [f"**Algorithm:** {alg_display(cfg['algorithm'])}"]
    if cfg["algorithm"] == "dbscan":
        cfg_lines.append(
            f"**eps:** {cfg['eps']} — neighbourhood radius"
        )
        cfg_lines.append(
            f"**min_samples:** {cfg['min_samples']} — minimum region size"
        )
    else:
        cfg_lines.append(f"**Segments (k):** {cfg['n_clusters']}")
        if cfg["algorithm"] == "agglomerative":
            cfg_lines.append(f"**Linkage:** {cfg['linkage']}")
    cfg_html = "".join(
        f"<div class='config-line'>{line}</div>"
        for line in cfg_lines
    )
    st.markdown(
        f"""
        <div class="config-card">
            {cfg_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # Layer 2 — Cluster results
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### Cluster Results")

    labels = active["labels"]
    centers_orig = active.get("centers_orig")

    plot_df = base["raw"][[AGE, ANNUAL_INCOME, SPENDING_SCORE, GENRE]].copy()
    plot_df["Segment"] = labels
    has_centers = (
        centers_orig is not None and len(centers_orig)
        and ANNUAL_INCOME in centers_orig.columns
    )
    fig = charts.scatter_segments(
        plot_df, ANNUAL_INCOME, SPENDING_SCORE, "Segment",
        f"Customer Segments ({cfg_label(cfg)})",
        color_map=cluster_colors(active),
        hover_data=[AGE, GENRE],
        hover_labels=["Age", "Gender"],
        centers_x=(centers_orig[ANNUAL_INCOME] if has_centers else None),
        centers_y=(centers_orig[SPENDING_SCORE] if has_centers else None),
        xlabel="Annual Income (k$)", ylabel="Spending Score (1-100)",
        height=440,
    )
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("###### Customers per Segment")
        counts = pd.Series(labels).value_counts().sort_index()
        size_df = pd.DataFrame(
            {"Segment": counts.index, "Customers": counts.values}
        )
        color_map = {}
        for cid in counts.index:
            p = (active.get("personas") or {}).get(int(cid))
            color_map[int(cid)] = (
                PERSONA_COLORS.get(p.key, "#94a3b8") if p else "#94a3b8"
            )
        bar_fig = charts.bar_segments(
            size_df, "Segment", "Customers",
            color_map=color_map,
            xlabel="Segment", ylabel="Customers", height=320,
        )
        st.plotly_chart(bar_fig, width="stretch")
    with c2:
        st.markdown("###### Model Metrics")
        cards.styled_metric_table(_active_metrics_df(active))

    if centers_orig is not None and len(centers_orig):
        st.markdown("###### Segment Centres (original scale)")
        center_df = centers_orig.copy()
        center_df.index = [f"Segment {i}" for i in center_df.index]
        center_df = center_df.round(1)
        cards.styled_metric_table(
            center_df.reset_index().rename(columns={"index": "Segment"})
        )

    # ------------------------------------------------------------------ #
    # Layer 3 — Interpretation
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### Interpretation")
    st.caption("What the selected configuration means in business terms.")
    st.markdown(_ALGO_BLURBS.get(cfg["algorithm"], ""))

    param_lines: list[str] = []
    if cfg["algorithm"] in ("kmeans", "agglomerative", "gmm"):
        param_lines.append(
            f"- **Segments (k):** {cfg['n_clusters']} — the number of distinct "
            "groups the model will create."
        )
    if cfg["algorithm"] == "agglomerative":
        param_lines.append(
            f"- **Linkage:** {cfg['linkage']} — the rule used when merging groups."
        )
    if cfg["algorithm"] == "dbscan":
        param_lines.append(
            f"- **eps:** {cfg['eps']} — how close points must be to belong together."
        )
        param_lines.append(
            f"- **min_samples:** {cfg['min_samples']} — smallest a group can be to count."
        )
    if param_lines:
        st.markdown("\n".join(param_lines))

    sil = active.get("silhouette")
    if sil is not None:
        if sil >= 0.7:
            sil_text = "strong, well-separated segments"
        elif sil >= 0.5:
            sil_text = "reasonably separated segments"
        elif sil >= 0.25:
            sil_text = "weakly separated segments — groups overlap somewhat"
        else:
            sil_text = "poorly separated segments — groups overlap heavily"
        sil_read = (
            f"A silhouette of **{sil:.3f}** indicates {sil_text}. "
            "Values range from -1 (overlapping) to 1 (perfectly distinct)."
        )
    else:
        sil_read = "Silhouette is not available for this configuration."
    components.info_box(sil_read, kind="info")

    st.divider()
    st.markdown("###### Deploy this configuration")
    st.caption(
        "Save the active configuration as the deployed prediction model. The "
        "prediction form then reuses it without retraining."
    )
    if st.button("Retrain & Save Prediction Model"):
        bundle = train_bundle(base, cfg)
        ensure_directory(MODELS_DIR)
        joblib.dump(bundle, MODEL_PATH)
        st.cache_resource.clear()
        logger.info("Prediction model manually retrained and saved by user.")
        st.success(f"Prediction model saved to {MODEL_PATH.name}.")


# ===========================================================================
# Section 4 - Model Evaluation
# ===========================================================================
def section_comparison(base: dict[str, Any]) -> None:
    components.section_header(
        "Model Evaluation",
        "Compare candidate algorithms against multi-metric quality and choose "
        "an appropriate number of segments.",
    )

    X = base["X"]
    cfg = st.session_state.cfg
    active = st.session_state.active

    with st.spinner("Evaluating candidate algorithms..."):
        table = _eval_table(X)

    # ------------------------------------------------------------------ #
    # Active model metric cards
    # ------------------------------------------------------------------ #
    st.markdown("##### Active Configuration Metrics")
    st.caption(f"Quality metrics for the deployed {alg_display(cfg['algorithm'])} model.")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        cards.metric_card(
            "Silhouette Score", f"{styles.fmt_metric(active.get('silhouette'), 3)}",
            "Higher is better",
        )
    with m2:
        ch = active.get("calinski_harabasz")
        cards.metric_card(
            "Calinski-Harabasz", f"{styles.fmt_metric(ch, 1)}",
            "Higher is better", "Requires ≥ 2 segments",
        )
    with m3:
        db = active.get("davies_bouldin")
        cards.metric_card(
            "Davies-Bouldin", f"{styles.fmt_metric(db, 3)}",
            "Lower is better",
        )
    with m4:
        inertia = active.get("inertia")
        note = "K-Means only" if inertia is None else ""
        cards.metric_card(
            "Inertia (WCSS)", f"{styles.fmt_metric(inertia, 2)}",
            "Lower is better", note,
        )

    # ------------------------------------------------------------------ #
    # Model comparison table
    # ------------------------------------------------------------------ #
    st.markdown("###### Algorithm Comparison")
    comp_df = pd.DataFrame(table.rows).copy()
    raw_alg = comp_df["Algorithm"].copy()
    disp_df = comp_df.copy()
    disp_df["Algorithm"] = disp_df["Algorithm"].map(alg_display)

    metric_cols = [
        c for c in [
            "Silhouette Score", "Calinski-Harabasz Score",
            "Davies-Bouldin Score", "Inertia",
        ]
        if c in comp_df.columns and pd.api.types.is_numeric_dtype(comp_df[c])
    ]
    if metric_cols:
        styler = disp_df.style.format({c: "{:.3f}" for c in metric_cols})
        styler = styler.bar(subset=metric_cols, color="#e2e8f0", height=1.0)
        st.dataframe(styler, width="stretch", hide_index=True)
    else:
        cards.styled_metric_table(disp_df)

    with st.expander("How to read these metrics"):
        st.caption(
            "Each metric summarises a different aspect of segment quality. "
            "Higher-is-better and lower-is-better are noted for every metric "
            "so the table can be interpreted without a statistical background."
        )
        for name, direction, explanation in METRIC_GLOSSARY:
            st.markdown(f"**{name}** — *{direction}*")
            st.markdown(explanation)

    # ------------------------------------------------------------------ #
    # Recommendation box
    # ------------------------------------------------------------------ #
    st.divider()
    recommendation = table.recommendation
    if recommendation:
        rec_rows = comp_df[raw_alg == recommendation]
        rec_k = ""
        if not rec_rows.empty and "Clusters" in rec_rows.columns:
            try:
                best_idx = pd.to_numeric(
                    rec_rows["Clusters"], errors="coerce"
                ).idxmax()
                rec_k = f" · {int(rec_rows.loc[best_idx, 'Clusters'])} clusters"
            except Exception:
                rec_k = ""
        try:
            best_score = next(
                s for r, s in table.ranking if r == recommendation
            )
        except StopIteration:
            best_score = None

        why_parts = [
            "Best balance across Silhouette, Calinski-Harabasz, Davies-Bouldin "
            "and Inertia for this dataset."
        ]
        if best_score is not None:
            why_parts.append(
                f"Composite quality score: {best_score:.2f}."
            )
        components.recommendation_box(
            "Recommended Configuration",
            f"{alg_display(recommendation)}{rec_k}",
            items=why_parts,
            accent=PRIMARY,
        )
    else:
        components.info_box("No candidate produced valid metrics.")

    # ------------------------------------------------------------------ #
    # K selection + metric comparison charts
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### K Selection Diagnostics")
    st.caption(
        "K-Means evaluated across a range of k. The silhouette peak and the "
        "inertia elbow both inform the segment count."
    )
    with st.spinner("Computing k-range diagnostics..."):
        with _suppress_warnings():
            warnings.simplefilter("ignore")
            kopt = _k_analysis(X)

    fig = charts.elbow_silhouette_chart(
        kopt.k_values, kopt.inertias, kopt.silhouette_scores, kopt.optimal_k,
    )
    st.plotly_chart(fig, width="stretch")
    components.info_box(
        f"Based on silhouette analysis, the optimal k is **{kopt.optimal_k}** "
        f"(peak silhouette {max(kopt.silhouette_scores):.3f}). "
        f"Recommended algorithm: {alg_display(table.recommendation or 'n/a')}.",
        kind="info",
    )

    # Metric comparison bar chart (silhouette across configs).
    st.markdown("###### Metric Comparison")
    chart_df = disp_df.copy()
    if "Silhouette Score" in chart_df.columns:
        fig = charts.metric_bar(
            chart_df, "Silhouette Score", "Algorithm",
            ascending=False, title="Silhouette Score by Configuration",
        )
        st.plotly_chart(fig, width="stretch")

    # Cluster stability (K-Means)
    st.divider()
    st.markdown("##### Cluster Stability (K-Means)")
    st.caption(
        "K-Means is run 10 times with different random seeds. Pairwise "
        "Adjusted Rand Index (ARI) measures label agreement. 1.0 = perfect, "
        "0.0 = random."
    )
    with st.spinner("Evaluating cluster stability..."):
        with _suppress_warnings():
            warnings.simplefilter("ignore")
            from src.analytics import compute_cluster_stability
            try:
                stab = compute_cluster_stability(X, n_clusters=5, n_runs=10)
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    cards.kpi_card("Mean ARI", f"{stab.mean_ari:.3f}", "Across 10 runs", SUCCESS)
                with sc2:
                    cards.kpi_card("Min ARI", f"{stab.min_ari:.3f}", "Worst agreement", ERROR)
                with sc3:
                    cards.kpi_card("Max ARI", f"{stab.max_ari:.3f}", "Best agreement", PRIMARY)
                with sc4:
                    cards.kpi_card("Std ARI", f"{stab.std_ari:.3f}", "Variability", PURPLE)
                components.info_box(stab.interpretation)
            except Exception as exc:
                components.error_state("Stability analysis failed.", str(exc))


# ===========================================================================
# Section 5 - Segment Explorer
# ===========================================================================
def _segment_stats(base: dict[str, Any], labels: np.ndarray) -> pd.DataFrame:
    """Return per-cluster aggregates (income, spending, age, top genre)."""
    enriched = base["cleaned"].copy()
    enriched["Cluster"] = labels
    grp = enriched.groupby("Cluster", observed=True)
    stats = pd.DataFrame({
        "avg_income": grp[ANNUAL_INCOME].mean(),
        "avg_spending": grp[SPENDING_SCORE].mean(),
        "avg_age": grp[AGE].mean(),
        "top_genre": grp[GENRE].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"
        ),
    })
    return stats


def section_explorer(base: dict[str, Any], active: dict[str, Any]) -> None:
    components.section_header(
        "Segment Explorer",
        "Meet each customer persona and read its profile and recommended action.",
    )

    if active.get("error") or not active.get("valid"):
        components.error_state("No valid segments available for the current configuration.")
        return

    personas = active["personas"]
    profiles_by_id = {p.cluster_id: p for p in active["profiles"]}
    cluster_ids = sorted(personas.keys())
    stats = _segment_stats(base, active["labels"])

    # ------------------------------------------------------------------ #
    # Persona overview cards
    # ------------------------------------------------------------------ #
    st.markdown("##### Customer Personas")
    st.caption("At a glance: who each segment is and how many customers it holds.")
    cards_sorted = sorted(
        cluster_ids,
        key=lambda c: (profiles_by_id.get(c).customer_count
                       if profiles_by_id.get(c) else 0),
        reverse=True,
    )

    if "explorer_segment" not in st.session_state:
        st.session_state.explorer_segment = cards_sorted[0]
    selected = st.session_state.explorer_segment
    cols = st.columns(min(len(cards_sorted), 3))
    for idx, cid in enumerate(cards_sorted):
        persona = personas[cid]
        prof = profiles_by_id.get(cid)
        color = PERSONA_COLORS.get(persona.key, "#94a3b8")
        count = prof.customer_count if prof else 0
        pct = prof.percentage if prof else 0.0
        is_sel = cid == selected
        with cols[idx % len(cols)]:
            cards.persona_card(
                persona_name=persona.name, persona_key=persona.key,
                segment_id=cid, customer_count=count, percentage=pct,
                color=color, is_selected=is_sel,
            )

    # ------------------------------------------------------------------ #
    # Detailed drill-down for one segment
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### Segment Detail")
    selection = st.selectbox(
        "Select a segment / persona",
        options=cluster_ids,
        format_func=lambda cid: f"Segment {cid} — {personas[cid].name}",
        key="explorer_segment",
        label_visibility="collapsed",
    )

    persona = personas[selection]
    prof = profiles_by_id.get(selection)
    insight = _PERSONA_INSIGHTS.get(persona.key)
    color = PERSONA_COLORS.get(persona.key, "#94a3b8")

    st.markdown(
        f"""
        <div class="surface-box-bar" style="border-left-color:{color};margin-bottom:14px;">
            <div class="persona-header-title">{_esc(persona.name)}</div>
            <div class="persona-header-sub">Segment {selection}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(persona.description)

    if prof is not None:
        avg_income = float(stats.loc[selection, "avg_income"])
        avg_spending = float(stats.loc[selection, "avg_spending"])
        avg_age = float(stats.loc[selection, "avg_age"])
        top_genre = str(stats.loc[selection, "top_genre"])

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            cards.kpi_card(
                "Customers", f"{prof.customer_count:,}",
                f"{prof.percentage}% of base", color,
            )
        with k2:
            cards.kpi_card(
                "Avg Income", styles.fmt_income(avg_income),
                "per customer", WARNING,
            )
        with k3:
            cards.kpi_card(
                "Avg Spending", f"{avg_spending:,.1f}",
                "score (1-100)", PURPLE,
            )
        with k4:
            cards.kpi_card(
                "Avg Age", f"{avg_age:,.0f}",
                f"top gender: {top_genre}", TEAL,
            )

        st.markdown("###### Profile")
        st.write(prof.profile_summary)
        if prof.key_characteristics:
            st.markdown("**Key characteristics:** " + ", ".join(prof.key_characteristics))

        st.markdown("###### Recommended Action")
        components.recommendation_box(
            persona.name, persona.strategy, accent=color,
        )

    if insight is not None:
        st.markdown("###### Strategy & Recommendations")
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("**Business interpretation**")
            st.write(insight.business_interpretation)
            st.markdown("**Potential opportunity**")
            st.write(insight.potential_opportunity)
        with b2:
            st.markdown("**Marketing strategy**")
            st.write(insight.recommended_marketing_strategy)
            st.markdown("**Retention & engagement**")
            st.write(insight.retention_engagement_recommendation)

    # ------------------------------------------------------------------ #
    # Segment vs overall population
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### Segment vs Overall Population")
    st.caption(
        "How each segment differs from the overall customer base. Values are "
        "standardised differences (Cohen's d): positive = above average, "
        "negative = below average."
    )
    comp_data = active.get("segment_comparison")
    if comp_data:
        from src.analytics import comparison_to_dataframe
        comp_df = comparison_to_dataframe(comp_data)
        comp_display = comp_df.copy()
        for col in comp_display.columns:
            if col != "Cluster":
                comp_display[col] = comp_display[col].apply(
                    lambda v: f"{v:+.2f}" if isinstance(v, (int, float)) else v
                )
        cards.styled_metric_table(comp_display)
    else:
        components.info_box(
            "Segment comparison is not available for this configuration."
        )

    # ------------------------------------------------------------------ #
    # Feature separation
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### Feature Separation Analysis")
    st.caption(
        "ANOVA F-ratios showing which features most strongly differentiate the "
        "segments. Higher F-ratio = more important for distinguishing between "
        "clusters."
    )
    sep_data = active.get("feature_separation")
    if sep_data:
        sep_rows = [
            {"Feature": r.feature, "F-Ratio": r.f_ratio, "Rank": r.rank}
            for r in sorted(sep_data, key=lambda x: x.rank)
        ]
        sep_df = pd.DataFrame(sep_rows)
        cards.styled_metric_table(sep_df)
    else:
        components.info_box(
            "Feature separation analysis is not available for this configuration."
        )

    # ------------------------------------------------------------------ #
    # Customer table for the selected segment
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### Customers in this Segment")
    query = st.text_input(
        "Search customers (by customer id, age, income or spending)",
        value="", label_visibility="collapsed",
        placeholder="Type to filter…",
    )
    seg_labels = active["labels"]
    cust = base["raw"].copy()
    cust["Segment"] = seg_labels
    mask = cust["Segment"] == selection
    seg_cust = cust[mask].copy()
    if query:
        q = query.lower()
        seg_cust = seg_cust[
            seg_cust.apply(
                lambda r: q in str(r[CUSTOMER_ID]).lower()
                or q in str(r[AGE]).lower()
                or q in str(r[ANNUAL_INCOME]).lower()
                or q in str(r[SPENDING_SCORE]).lower()
                or q in str(r[GENRE]).lower(),
                axis=1,
            )
        ]
    if seg_cust.empty:
        components.empty_state(
            "No customers match the current filter for this segment.",
            hint="Clear the search box to see all customers in the segment.",
        )
    else:
        st.dataframe(
            seg_cust[[CUSTOMER_ID, GENRE, AGE, ANNUAL_INCOME, SPENDING_SCORE]],
            width="stretch", hide_index=True,
        )


# ===========================================================================
# Section 6 - New Customer Prediction
# ===========================================================================
def section_predict(base: dict[str, Any]) -> None:
    components.section_header(
        "New Customer Prediction",
        "Enter a customer's attributes to assign them to a segment and receive "
        "a tailored business recommendation.",
    )

    bundle = load_prediction_model()
    if bundle is None:
        components.error_state(
            "Prediction model could not be loaded (dataset missing).",
        )
        return
    if not bundle.get("personas"):
        components.error_state(
            "The loaded prediction model has no usable segments.",
            "Retrain it from the Segmentation Lab with a valid configuration.",
        )
        return

    cfg_used = bundle.get("config", {})
    st.caption(
        f"Using deployed model: **{cfg_label(cfg_used)}**. The model is loaded "
        "once and reused — submitting the form does not retrain it."
    )

    col_in, col_out = st.columns([1, 1])
    with col_in:
        with st.form("prediction_form"):
            st.markdown("###### Customer Information")
            age = st.number_input(
                "Age", min_value=10, max_value=100, value=35, step=1
            )
            genre = st.selectbox(
                "Gender", options=["Male", "Female"], index=0
            )
            income = st.number_input(
                "Annual Income (k$)", min_value=1.0, max_value=200.0,
                value=60.0, step=1.0,
            )
            spending = st.number_input(
                "Spending Score (1-100)", min_value=1, max_value=100,
                value=50, step=1,
            )
            submitted = st.form_submit_button(
                "Predict Segment", type="primary",
            )

        if submitted:
            problems = []
            if not (10 <= age <= 100):
                problems.append("Age must be between 10 and 100.")
            if not (1 <= income <= 200):
                problems.append("Annual income must be between 1 and 200 (k$).")
            if not (1 <= spending <= 100):
                problems.append("Spending score must be between 1 and 100.")
            if problems:
                for p in problems:
                    st.warning(p)
            else:
                with st.spinner("Assigning to segment..."):
                    pred = predict_new(
                        bundle, int(age), genre, float(income), float(spending)
                    )
                if pred is None or pred["persona"] is None:
                    components.error_state(
                        "Could not assign this customer to a segment.",
                        "The model has no valid cluster centres. Retrain it from "
                        "the Segmentation Lab.",
                    )
                else:
                    st.session_state.last_prediction = {
                        "pred": pred, "age": age, "genre": genre,
                        "income": income, "spending": spending,
                    }

    with col_out:
        st.markdown("###### Prediction")
        pred_state = st.session_state.get("last_prediction")
        if not pred_state:
            components.empty_state(
                "No prediction yet.",
                hint="Enter a customer's details on the left and choose "
                     "**Predict Segment**.",
            )
        else:
            pred = pred_state["pred"]
            persona = pred["persona"]
            insight = _PERSONA_INSIGHTS.get(persona.key)
            color = PERSONA_COLORS.get(persona.key, "#94a3b8")

            # Large, prominent result
            st.markdown(
                f"""
                <div class="surface-box" style="--accent:{color};">
                    <div class="surface-box-tag">Assigned segment</div>
                    <div class="result-title">Segment {pred['cluster_id']}</div>
                    <div class="result-subtitle">{_esc(persona.name)}</div>
                    <div class="kpi-sub">Distance to nearest centre: {pred['distance']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("###### Explanation")
            st.write(persona.description)
            if insight is not None:
                st.markdown("###### Recommended Strategy")
                components.recommendation_box(
                    persona.name, persona.strategy, accent=color,
                )
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown("**Opportunity**")
                    st.write(insight.potential_opportunity)
                    st.markdown("**Retention & engagement**")
                    st.write(insight.retention_engagement_recommendation)
                with r2:
                    st.markdown("**Marketing strategy**")
                    st.write(insight.recommended_marketing_strategy)
                    st.markdown("**Interpretation**")
                    st.write(insight.business_interpretation)


# ===========================================================================
# Section 7 - Export
# ===========================================================================
def section_export(base: dict[str, Any], active: dict[str, Any]) -> None:
    components.section_header(
        "Export",
        "Download analysis artefacts as CSV for downstream reporting and BI tools.",
    )

    st.markdown("##### Export Centre")
    st.caption(
        "Each export contains a single, self-describing dataset ready for "
        "spreadsheet or BI ingestion."
    )

    # 1. Segmented customers
    cleaned = base["cleaned"].copy()
    if active.get("labels") is not None:
        cleaned["Segment"] = active["labels"]
        seg_csv = cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Customer segmentation results",
            data=seg_csv,
            file_name="customer_segmentation_results.csv",
            mime="text/csv",
            help="Every customer with their assigned segment for the active configuration.",
        )
        st.caption("All customers and their assigned segment (CSV).")

    if active.get("profiles"):
        rows = []
        for p in active["profiles"]:
            persona = active["personas"].get(p.cluster_id)
            rows.append(
                {
                    "Segment": p.cluster_id,
                    "Persona": p.persona_name,
                    "Customers": p.customer_count,
                    "Percentage": p.percentage,
                    "Profile": p.profile_summary,
                    "Key Characteristics": ", ".join(p.key_characteristics),
                    "Strategy": persona.strategy if persona else "",
                }
            )
        summary_csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Segment summary",
            data=summary_csv,
            file_name="segment_summary.csv",
            mime="text/csv",
            help="One row per segment with size, profile and recommended strategy.",
        )
        st.caption("Per-segment size, profile and strategy (CSV).")

    # 3. Model comparison
    with st.spinner("Building model comparison..."):
        with _suppress_warnings():
            warnings.simplefilter("ignore")
            table = _eval_table(base["X"])
    comp_df = pd.DataFrame(table.rows).copy()
    comp_df["Algorithm"] = comp_df["Algorithm"].map(lambda a: alg_display(a))
    comp_csv = comp_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Model evaluation",
        data=comp_csv,
        file_name="model_evaluation.csv",
        mime="text/csv",
        help="Metric scores for every candidate algorithm and configuration.",
    )
    st.caption("Silhouette, Calinski-Harabasz, Davies-Bouldin and inertia per model (CSV).")

    # 4. Cluster insights report
    st.divider()
    st.markdown("##### Cluster Insights Report")
    if active.get("profiles") and active.get("personas"):
        profiles_df = build_cluster_profiles(base["cleaned"], active["labels"])
        insights = generate_cluster_insights(profiles_df, active["personas"])
        report = "\n\n".join(insights)
        st.download_button(
            "Customer segment insights",
            data=report.encode("utf-8"),
            file_name="cluster_insights_report.txt",
            mime="text/plain",
            help="Plain-text narrative of every segment, its size and strategy.",
        )
        st.caption("Human-readable narrative of all segments (TXT).")
        with st.expander("Preview insights"):
            st.text(report)
    else:
        components.empty_state(
            "No segments available to export.",
            hint="Apply a valid configuration in the sidebar first.",
        )


# ===========================================================================
# App shell
# ===========================================================================
def _init_session_state() -> None:
    """Initialise session_state keys once per session."""
    if "cfg" not in st.session_state:
        st.session_state.cfg = dict(DEFAULT_CONFIG)
    if "active" not in st.session_state:
        st.session_state.active = None
    if "active_key" not in st.session_state:
        st.session_state.active_key = None
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "page" not in st.session_state:
        st.session_state.page = "overview"


def _sidebar(base: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Render the sidebar: branding, navigation, configuration, dataset status."""
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### Intelligent Customer Segmentation")
    st.sidebar.caption("Customer Intelligence Platform")

    st.sidebar.radio(
        "Navigate to a section",
        options=[k for k, _ in NAV],
        format_func=lambda k: NAV_LABELS[k],
        key="page",
        label_visibility="collapsed",
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    _sidebar_controls(cfg, base)

    st.sidebar.markdown("---")
    components.dataset_status(
        raw_rows=len(base["raw"]),
        raw_cols=len(base["raw"].columns),
        ok=True,
        version=f"model v{MODEL_VERSION}",
    )
    return st.session_state.page


def main() -> None:
    st.set_page_config(
        page_title="Customer Intelligence dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )
    styles.inject_global_styles()

    _init_session_state()

    base = get_base()
    if base.get("error"):
        logger.error("Dataset loading failed: %s", base["error"])
        components.error_state("Could not load the dataset.", base["error"])
        st.stop()
        return

    cfg = st.session_state.cfg
    logger.info("App initialised with algorithm=%s", cfg["algorithm"])

    page = _sidebar(base, cfg)

    active = get_active(base)

    # Compact status bar summarising the active model.
    n_seg = active.get("n_clusters", 0) if not active.get("error") else 0
    components.app_header(
        app_name="INTELLIGENT CUSTOMER SEGMENTATION",
        tagline="Customer Intelligence & Behavioral Analytics",
        algorithm=alg_display(cfg["algorithm"]),
        n_segments=n_seg,
        n_customers=len(base["raw"]),
        active_label=cfg_label(cfg),
    )

    if page == "overview":
        section_overview(base, active)
    elif page == "analytics":
        section_analytics(base)
    elif page == "lab":
        section_lab(base, active)
    elif page == "evaluation":
        section_comparison(base)
    elif page == "explorer":
        section_explorer(base, active)
    elif page == "prediction":
        section_predict(base)
    elif page == "export":
        section_export(base, active)


if __name__ == "__main__":
    main()
