"""Professional Streamlit dashboard for Intelligent Customer Segmentation.

The dashboard is a thin presentation layer: all ML logic (loading, validation,
preprocessing, clustering, evaluation, persona assignment, business insights) is
delegated to the reusable modules in ``src/``. Nothing in the modelling pipeline
is duplicated here.

Run with::

    streamlit run app.py

Sections
--------
1. Executive Overview      - KPI cards for the active configuration
2. Customer Analytics      - interactive Plotly distributions / correlations
3. Clustering Lab          - choose algorithm + parameters, visualise result
4. Model Comparison        - algorithms, metrics, elbow + silhouette analysis
5. Segment Explorer        - inspect a persona / cluster in detail
6. New Customer Prediction - form -> preprocessing -> trained model -> persona
7. Export                  - download CSV artefacts

The prediction model is trained once and persisted with ``joblib``; submitting
the form does not retrain it.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    MODEL_FILENAME,
    MODEL_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    SPENDING_SCORE,
    ensure_directory,
    setup_logging,
)

logger = setup_logging()


def _suppress_warnings():
    """Context manager that suppresses sklearn/convergence warnings."""
    return warnings.catch_warnings()


# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------
ALGORITHMS = {
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
DISPLAY = {
    AGE: "Age",
    ANNUAL_INCOME: "Annual Income (k$)",
    SPENDING_SCORE: "Spending Score (1-100)",
    GENRE: "Gender",
}

# Restrained, professional palette (blue primary + selective accents).
PALETTE = [
    "#2563eb", "#16a34a", "#dc2626", "#7c3aed",
    "#ea580c", "#0891b2", "#ca8a04", "#db2777",
]

# Stable persona identity colours (keyed by persona.key from src/personas.py).
PERSONA_COLORS = {
    "vip": "#16a34a",
    "saver": "#2563eb",
    "impulsive": "#dc2626",
    "budget": "#7c3aed",
    "mainstream": "#ea580c",
}

PRIMARY = "#2563eb"
INK = "#0f172a"
MUTED = "#64748b"
BORDER = "#e2e8f0"
SURFACE = "#f8fafc"

CHART_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

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
        logger.info("Dataset loaded successfully: %d rows, %d columns", len(raw), len(raw.columns))
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
    logger.info("Preprocessing complete: %d samples, %d features", X.shape[0], X.shape[1])
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


def cfg_label(cfg: dict[str, Any]) -> str:
    """Human-readable description of a clustering configuration."""
    alg = ALGORITHMS.get(cfg["algorithm"], cfg["algorithm"])
    if cfg["algorithm"] == "dbscan":
        return f"{alg} (eps={cfg['eps']}, min_samples={cfg['min_samples']})"
    if cfg["algorithm"] == "agglomerative":
        return f"{alg} (k={cfg['n_clusters']}, linkage={cfg['linkage']})"
    return f"{alg} (k={cfg['n_clusters']})"


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
        generate_analytical_insights,
        compute_cluster_stability,
        compute_feature_separation,
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

    centers_orig: Optional[pd.DataFrame] = None
    if result.cluster_centers is not None:
        try:
            centers_orig = pre.inverse_transform_centers(result.cluster_centers)
            logger.info("Inverse-transformed %d cluster centres to original scale", len(centers_orig))
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
    sil = eval_res.metrics["Silhouette Score"].value

    profiles: list[Any] = []
    profiles_df: Optional[pd.DataFrame] = None
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
    stability: Optional[Any] = None
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
                stability = compute_cluster_stability(X, n_clusters=result.n_clusters)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cluster stability analysis failed: %s", exc)
        try:
            analytical_insights = generate_analytical_insights(
                profiles_df, centers_orig if centers_orig is not None else pd.DataFrame(),
                result.labels, cleaned,
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


def _validate_bundle(bundle: Any) -> Optional[str]:
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
        return (f"model_version mismatch: artifact={bundle.get('model_version')!r}, "
                f"current={MODEL_VERSION!r}")
    pre = bundle.get("pre")
    if pre is None or getattr(pre, "transformer_", None) is None:
        return "preprocessor is not fitted"
    try:
        feat_names = list(pre.output_feature_names)
    except Exception as exc:  # noqa: BLE001
        return f"cannot read preprocessor feature names: {exc}"
    expected = list(FEATURE_COLUMNS)
    if feat_names != expected:
        return (f"feature names mismatch: artifact={feat_names}, "
                f"current={expected}")
    if bundle.get("algorithm") not in ALGORITHMS:
        return f"unknown algorithm in bundle: {bundle.get('algorithm')!r}"
    centers = bundle.get("scaled_centers")
    if centers is None or len(centers) == 0:
        return "bundle has no cluster centres"
    n_centers = int(len(centers))
    personas = bundle.get("personas") or {}
    if n_centers != len(personas):
        return (f"cluster/ persona count mismatch: "
                f"centers={n_centers}, personas={len(personas)}")
    return None


@st.cache_resource(show_spinner="Loading prediction model...")
def load_prediction_model() -> Optional[dict[str, Any]]:
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
) -> Optional[dict[str, Any]]:
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
# Presentation helpers
# ===========================================================================
def cluster_colors(active: dict[str, Any]) -> dict[int, str]:
    """Map each cluster id to its persona colour for visual consistency."""
    personas = active.get("personas") or {}
    return {
        cid: PERSONA_COLORS.get(p.key, "#94a3b8") for cid, p in personas.items()
    }


def fmt_income(value: float) -> str:
    """Format an income value in thousands of dollars."""
    return f"${value:,.1f}k"


def fmt_metric(value: Optional[float], digits: int = 3) -> str:
    """Format a numeric metric, falling back to a dash for missing values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def kpi_card(
    label: str,
    value: str,
    sub: str = "",
    color: str = PRIMARY,
) -> None:
    """Render a single KPI card via HTML for a polished, business look."""
    html = f"""
    <div style="background:#ffffff;border:1px solid {BORDER};
        border-left:4px solid {color};border-radius:10px;padding:16px 18px;
        height:100%;box-shadow:0 1px 2px rgba(15,23,42,0.04);">
      <div style="font-size:12px;color:{MUTED};font-weight:600;
        text-transform:uppercase;letter-spacing:0.05em;">{label}</div>
      <div style="font-size:26px;color:{INK};font-weight:700;
        margin-top:8px;line-height:1.1;">{value}</div>
      <div style="font-size:12px;color:{MUTED};margin-top:6px;">{sub}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def section_header(title: str, caption: str = "") -> None:
    """Render a consistent section title with optional supporting caption."""
    st.markdown(f"<h2 style='color:{INK};margin-bottom:2px;'>{title}</h2>",
                unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def style_chart(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 360,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> go.Figure:
    """Apply the shared, restrained analytics theme to a Plotly figure."""
    if title is not None:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left"))
    fig.update_layout(
        template="plotly_white",
        font=dict(family=CHART_FONT, size=13, color=INK),
        height=height,
        margin=dict(t=54, b=44, l=54, r=24),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=PALETTE,
        hoverlabel=dict(font_size=12, font_family=CHART_FONT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    base_axes = dict(
        gridcolor="#eef2f7",
        zerolinecolor="#e2e8f0",
        linecolor="#cbd5e1",
        title_font=dict(size=12, color=MUTED),
        tickfont=dict(size=11, color=MUTED),
    )
    x_axes = {**base_axes}
    y_axes = {**base_axes}
    if xlabel is not None:
        x_axes["title"] = xlabel
    if ylabel is not None:
        y_axes["title"] = ylabel
    fig.update_xaxes(**x_axes)
    fig.update_yaxes(**y_axes)
    return fig


def styled_metric_table(df: pd.DataFrame) -> None:
    """Render a dataframe with light styling (no emoji clutter)."""
    st.dataframe(df, width="stretch", hide_index=True)


def show_error(msg: str, hint: str = "") -> None:
    st.error(msg)
    if hint:
        st.caption(hint)


# ===========================================================================
# Section 1 - Executive Overview
# ===========================================================================
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
            "Adjust the algorithm and parameters in the Clustering Lab, then "
            "apply a configuration that produces at least two valid clusters.",
        )

    by_count = sorted(profiles, key=lambda p: p.customer_count, reverse=True)
    largest = by_count[0]
    largest_persona = personas.get(largest.cluster_id)

    # Highest-value opportunity: a High-Value (vip) segment if present,
    # otherwise the segment with the greatest combined income + spending.
    vip = next((p for p in profiles if personas.get(p.cluster_id, None)
                and personas[p.cluster_id].key == "vip"), None)
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
    inc = float(centers.loc[value_seg.cluster_id, ANNUAL_INCOME]) if centers is not None else float("nan")
    spd = float(centers.loc[value_seg.cluster_id, SPENDING_SCORE]) if centers is not None else float("nan")

    headline = (
        f"The largest group is **{largest_persona.name if largest_persona else 'Segment ' + str(largest.cluster_id)}** "
        f"at {largest.percentage}% of the base "
        f"({largest.customer_count:,} customers). "
        f"**{value_persona.name if value_persona else 'Segment ' + str(value_seg.cluster_id)}** "
        f"is the highest-value opportunity, averaging {fmt_income(inc)} income "
        f"and a spending score of {spd:.0f}."
    )

    total = len(active.get("labels", [])) if active.get("labels") is not None else 0
    interpretation = (
        f"Customer bases are rarely uniform. The segmentation splits the "
        f"{total:,} customers into distinct groups so each "
        f"team can act on the right audience. The largest segment tells you where "
        f"most of your volume sits; the highest-value segment tells you where a "
        f"small, well-targeted premium offer can return the most revenue. "
        f"Use the Segment Explorer to read each group's profile and the recommended action."
    )
    return headline, interpretation


def section_overview(base: dict[str, Any], active: dict[str, Any]) -> None:
    section_header(
        "Executive Overview",
        "A snapshot of the customer base and the active segmentation configuration.",
    )

    raw = base["raw"]
    cfg = st.session_state.cfg

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Customers", f"{len(raw):,}", "After deduplication", PRIMARY)
    with col2:
        n_clusters = active.get("n_clusters", 0) if not active.get("error") else 0
        kpi_card(
            "Segments",
            str(n_clusters),
            f"{active.get('noise', 0)} noise points" if cfg["algorithm"] == "dbscan" else "Distinct groups found",
            "#16a34a",
        )
    with col3:
        avg_income = raw[ANNUAL_INCOME].mean()
        kpi_card("Average Income", fmt_income(avg_income), "Across all customers", "#ea580c")
    with col4:
        avg_spend = raw[SPENDING_SCORE].mean()
        kpi_card("Average Spending", f"{avg_spend:,.1f}", "Score (1-100)", "#7c3aed")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        kpi_card(
            "Selected Model",
            ALGORITHMS.get(cfg["algorithm"], cfg["algorithm"]),
            "Configurable in the Clustering Lab",
            "#0891b2",
        )
    with c2:
        sil = active.get("silhouette")
        kpi_card(
            "Clustering Configuration",
            cfg_label(cfg),
            f"Silhouette: {fmt_metric(sil, 3)}"
            if not active.get("error")
            else "Silhouette: n/a",
            "#ca8a04",
        )

    st.divider()
    headline, interpretation = derive_headline(active)
    st.markdown(
        f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};
            border-left:4px solid {PRIMARY};border-radius:10px;
            padding:18px 20px;">
          <div style="font-size:12px;font-weight:700;color:{MUTED};
            text-transform:uppercase;letter-spacing:0.05em;">
            Most important business insight</div>
          <div style="font-size:15px;color:{INK};margin-top:8px;line-height:1.5;">
            {headline}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("What does this mean?"):
        st.markdown(interpretation)
        st.markdown(
            "- **Segments** are groups of customers with similar income and "
            "spending behaviour, so campaigns can be tailored instead of "
            "one-size-fits-all.\n"
            "- **Silhouette** (0 to 1) measures how clearly separated the "
            "segments are — higher means tighter, more distinct groups.\n"
            "- **Average income / spending** set the financial scale of each "
            "group and guide how premium an offer should be."
        )

    st.divider()
    st.markdown("##### Key Analytical Insights")
    st.caption("Data-driven observations derived from the active segmentation.")
    insights_data = active.get("analytical_insights")
    if insights_data:
        for insight in insights_data:
            st.markdown(
                f"**{insight.title}:** {insight.detail}"
            )
    else:
        st.info("Analytical insights are not available for this configuration.")

    if active.get("error"):
        st.divider()
        show_error("The selected configuration failed.", active["error"])


# ===========================================================================
# Section 2 - Customer Analytics
# ===========================================================================
def section_analytics(base: dict[str, Any]) -> None:
    section_header(
        "Customer Analytics",
        "Interactive, exploratory views of the underlying customer base.",
    )

    raw = base["raw"]

    st.markdown("##### Distributions")
    d1, d2, d3 = st.columns(3)
    with d1:
        fig = px.histogram(
            raw, x=AGE, nbins=20, color_discrete_sequence=["#2563eb"],
            title="Age Distribution",
        )
        fig.update_traces(hovertemplate="Age: %{x}<br>Customers: %{y}<extra></extra>")
        fig = style_chart(fig, xlabel="Age (years)", ylabel="Customers")
        st.plotly_chart(fig, width="stretch")
    with d2:
        fig = px.histogram(
            raw, x=ANNUAL_INCOME, nbins=20, color_discrete_sequence=["#ea580c"],
            title="Annual Income Distribution",
        )
        fig.update_traces(hovertemplate="Income: %{x} k$<br>Customers: %{y}<extra></extra>")
        fig = style_chart(fig, xlabel="Annual Income (k$)", ylabel="Customers")
        st.plotly_chart(fig, width="stretch")
    with d3:
        fig = px.histogram(
            raw, x=SPENDING_SCORE, nbins=20, color_discrete_sequence=["#7c3aed"],
            title="Spending Score Distribution",
        )
        fig.update_traces(hovertemplate="Score: %{x}<br>Customers: %{y}<extra></extra>")
        fig = style_chart(fig, xlabel="Spending Score (1-100)", ylabel="Customers")
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Composition & Relationships")
    r1, r2 = st.columns(2)
    with r1:
        counts = raw[GENRE].value_counts().reset_index()
        counts.columns = [GENRE, "Count"]
        fig = px.pie(
            counts, names=GENRE, values="Count", title="Customer Composition by Gender",
            color_discrete_sequence=PALETTE, hole=0.45,
        )
        fig.update_traces(
            textinfo="percent+label",
            hovertemplate="%{label}<br>Customers: %{value}<br>Share: %{percent}<extra></extra>",
        )
        fig = style_chart(fig, height=380)
        fig.update_layout(showlegend=True, legend=dict(orientation="h",
                              yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
        st.plotly_chart(fig, width="stretch")
    with r2:
        corr = raw[[AGE, ANNUAL_INCOME, SPENDING_SCORE]].corr().round(2)
        corr.columns = ["Age", "Income", "Spending"]
        corr.index = ["Age", "Income", "Spending"]
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            title="Feature Correlation",
        )
        fig.update_traces(hovertemplate="%{y} vs %{x}<br>Correlation: %{z}<extra></extra>")
        fig.update_coloraxes(colorbar_title="Correlation")
        fig = style_chart(fig, height=380)
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Feature Relationships")
    rel_mode = st.radio(
        "Colour points by",
        ["Income vs Spending by Gender", "Income vs Spending by Segment"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if rel_mode.startswith("Income vs Spending by Gender"):
        fig = px.scatter(
            raw, x=ANNUAL_INCOME, y=SPENDING_SCORE, color=GENRE,
            hover_data=[AGE, GENRE], title="Annual Income vs Spending Score",
            color_discrete_sequence=PALETTE,
        )
    else:
        active = st.session_state.active
        cleaned = base["cleaned"].copy()
        if active.get("labels") is not None:
            cleaned["Segment"] = active["labels"]
            fig = px.scatter(
                cleaned, x=ANNUAL_INCOME, y=SPENDING_SCORE, color="Segment",
                hover_data=[AGE, GENRE], title="Income vs Spending by Segment",
                color_discrete_map=cluster_colors(active),
            )
        else:
            fig = px.scatter(
                raw, x=ANNUAL_INCOME, y=SPENDING_SCORE, hover_data=[AGE, GENRE],
                title="Annual Income vs Spending Score",
            )
    fig.update_traces(
        marker=dict(size=9, opacity=0.8, line=dict(width=0.5, color="white")),
        hovertemplate=(
            "Income: %{x} k$<br>Spending: %{y}<br>Age: %{customdata[0]}<br>"
            "Gender: %{customdata[1]}<extra></extra>"
        ),
    )
    fig = style_chart(fig, xlabel="Annual Income (k$)", ylabel="Spending Score (1-100)")
    st.plotly_chart(fig, width="stretch")

    # 3D scatter (Age, Income, Spending) coloured by segment when available.
    st.markdown("##### 3D Cluster View")
    st.caption(
        "Three-dimensional view of all numeric features. "
        "Rotate by dragging the plot."
    )
    active = st.session_state.active
    raw_3d = base["raw"][[AGE, ANNUAL_INCOME, SPENDING_SCORE]].copy()
    raw_3d["Segment"] = active["labels"] if active.get("labels") is not None else -1
    raw_3d["SegmentLabel"] = raw_3d["Segment"].apply(
        lambda s: "No segment" if s == -1 else f"Segment {s}"
    )
    fig3d = px.scatter_3d(
        raw_3d,
        x=AGE,
        y=ANNUAL_INCOME,
        z=SPENDING_SCORE,
        color="SegmentLabel",
        title="Customer Segments in 3D (Age, Income, Spending)",
        color_discrete_sequence=PALETTE,
        opacity=0.85,
    )
    fig3d.update_traces(
        marker=dict(size=5, line=dict(width=0.3, color="white")),
    )
    fig3d = style_chart(fig3d, height=500)
    fig3d.update_layout(
        scene=dict(
            xaxis_title="Age",
            yaxis_title="Annual Income (k$)",
            zaxis_title="Spending Score (1-100)",
        ),
    )
    st.plotly_chart(fig3d, width="stretch")


# ===========================================================================
# Section 3 - Clustering Lab
# ===========================================================================
def section_lab(base: dict[str, Any]) -> None:
    section_header(
        "Clustering Lab",
        "Configure an algorithm, review the resulting segments, then interpret "
        "what the configuration means for the business.",
    )

    cfg = st.session_state.cfg

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    st.markdown("##### Configuration")
    st.caption("Choose an algorithm and its parameters, then apply the configuration.")

    algorithm = st.selectbox(
        "Algorithm",
        options=list(ALGORITHMS.keys()),
        format_func=lambda a: ALGORITHMS[a],
        index=list(ALGORITHMS.keys()).index(cfg["algorithm"]),
    )

    params: dict[str, Any] = {"algorithm": algorithm}
    if algorithm in ("kmeans", "agglomerative", "gmm"):
        n_samples = len(base["cleaned"])
        max_valid_k = max(MIN_K, min(MAX_K, max(1, n_samples // 5)))
        slider_max = max(cfg["n_clusters"], max_valid_k)
        params["n_clusters"] = st.slider(
            "Number of segments (k)",
            min_value=MIN_K,
            max_value=slider_max,
            value=min(max(cfg["n_clusters"], MIN_K), slider_max),
            help=(
                f"Technically valid for {n_samples} samples: 2-{max_valid_k}. "
                f"Analytically recommended: see the silhouette analysis in "
                f"Model Comparison (typically 5-6 for this dataset)."
            ),
        )
        st.caption(
            f"Valid range for {n_samples} customers: **{MIN_K}-{max_valid_k}**. "
            "A larger k does not automatically give better segments; check the "
            "silhouette score and elbow in **Model Comparison**."
        )
    if algorithm == "agglomerative":
        params["linkage"] = st.selectbox(
            "Linkage method", ["ward", "complete", "average", "single"],
            index=["ward", "complete", "average", "single"].index(cfg["linkage"]),
            help="How distance between clusters is measured when merging them.",
        )
    if algorithm == "dbscan":
        c1, c2 = st.columns(2)
        with c1:
            params["eps"] = st.number_input(
                "Neighbourhood radius (eps)", min_value=0.05, max_value=5.0,
                value=float(cfg["eps"]), step=0.05,
                help="Maximum distance between points in the same dense region.",
            )
        with c2:
            params["min_samples"] = st.number_input(
                "Minimum samples", min_value=1, max_value=50,
                value=int(cfg["min_samples"]), step=1,
                help="Fewest points needed to form a dense region (a cluster).",
            )

    # Carry over unchanged fields.
    params.setdefault("n_clusters", cfg["n_clusters"])
    params.setdefault("linkage", cfg["linkage"])
    params.setdefault("eps", cfg["eps"])
    params.setdefault("min_samples", cfg["min_samples"])

    applied = st.button("Apply Configuration", type="primary",
                        help="Recompute segments for this configuration.")
    if applied:
        st.session_state.cfg = params
        st.session_state.active = None
        st.session_state.active_key = None
        st.rerun()

    active = st.session_state.active

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    if active.get("error"):
        st.divider()
        show_error("Clustering failed for this configuration.", active["error"])
        return

    if not active["valid"]:
        st.divider()
        if st.session_state.cfg["algorithm"] == "dbscan":
            show_error(
                "DBSCAN produced no valid clusters.",
                "All points were marked as noise. Increase eps or decrease "
                "min_samples so denser regions form clusters.",
            )
        else:
            show_error("No valid clusters were produced for this configuration.")
        return

    st.divider()
    st.markdown("##### Results")
    labels = active["labels"]
    n_clusters = active["n_clusters"]
    centers_orig = active["centers_orig"]

    cleaned = base["cleaned"].copy()
    cleaned["Segment"] = labels

    fig = px.scatter(
        cleaned, x=ANNUAL_INCOME, y=SPENDING_SCORE, color="Segment",
        hover_data=[AGE, GENRE], title=f"Customer Segments ({cfg_label(cfg)})",
        color_discrete_map=cluster_colors(active),
    )
    fig.update_traces(
        marker=dict(size=9, opacity=0.8, line=dict(width=0.5, color="white")),
        hovertemplate=(
            "Income: %{x} k$<br>Spending: %{y}<br>Age: %{customdata[0]}<br>"
            "Gender: %{customdata[1]}<extra></extra>"
        ),
    )
    if centers_orig is not None and len(centers_orig):
        fig.add_trace(
            go.Scatter(
                x=centers_orig[ANNUAL_INCOME],
                y=centers_orig[SPENDING_SCORE],
                mode="markers",
                marker=dict(symbol="diamond", size=16, color="#0f172a",
                            line=dict(width=2, color="white")),
                name="Segment centre",
                hovertemplate="Centre<br>Income: %{x} k$<br>Spending: %{y}<extra></extra>",
            )
        )
    fig = style_chart(fig, xlabel="Annual Income (k$)", ylabel="Spending Score (1-100)")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("###### Customers per Segment")
        counts = pd.Series(labels).value_counts().sort_index()
        size_df = pd.DataFrame({"Segment": counts.index, "Customers": counts.values})
        fig = px.bar(
            size_df, x="Segment", y="Customers", title="",
            color="Segment", color_discrete_map=cluster_colors(active),
        )
        fig.update_traces(
            hovertemplate="Segment %{x}<br>Customers: %{y}<extra></extra>",
            marker_line_color="white", marker_line_width=1,
        )
        fig = style_chart(fig, height=320, xlabel="Segment", ylabel="Customers")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("###### Model Metrics")
        metrics: dict[str, Any] = {}
        if active.get("silhouette") is not None:
            metrics["Silhouette Score"] = fmt_metric(active["silhouette"], 4)
        if active.get("inertia") is not None:
            metrics["Inertia (WCSS)"] = fmt_metric(active["inertia"], 2)
        metrics["Segments (excl. noise)"] = str(int(n_clusters))
        if st.session_state.cfg["algorithm"] == "dbscan":
            metrics["Noise points"] = str(int(active.get("noise", 0)))
        mdf = pd.DataFrame(
            {"Metric": list(metrics.keys()),
             "Value": [str(v) for v in metrics.values()]}
        )
        styled_metric_table(mdf)

    if centers_orig is not None and len(centers_orig):
        st.markdown("###### Segment Centres (original scale)")
        center_df = centers_orig.copy()
        center_df.index = [f"Segment {i}" for i in center_df.index]
        center_df = center_df.round(1)
        styled_metric_table(center_df.reset_index().rename(columns={"index": "Segment"}))

    st.divider()
    st.markdown("###### Deploy this configuration")
    st.caption(
        "Save the active configuration as the deployed prediction model. "
        "The prediction form then reuses it without retraining."
    )
    if st.button("Retrain & Save Prediction Model"):
        bundle = train_bundle(base, st.session_state.cfg)
        ensure_directory(MODELS_DIR)
        joblib.dump(bundle, MODEL_PATH)
        st.cache_resource.clear()
        logger.info("Prediction model manually retrained and saved by user.")
        st.success(f"Prediction model saved to {MODEL_PATH.name}.")

    # ------------------------------------------------------------------ #
    # Interpretation
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### Interpretation")
    st.caption("What the selected configuration means in business terms.")

    algo = ALGORITHMS.get(cfg["algorithm"], cfg["algorithm"])
    algo_blurb = {
        "kmeans": (
            "K-Means partitions customers into a fixed number of groups by "
            "placing each customer in the nearest centre, then moving the "
            "centres to best fit their group. It is fast and easy to explain, "
            "but assumes roughly round, similarly sized groups."
        ),
        "agglomerative": (
            "Agglomerative clustering starts with every customer as its own "
            "group and repeatedly merges the closest pairs. The linkage method "
            "controls how 'closeness' is measured, which shapes the final shape "
            "of the segments."
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
    }.get(cfg["algorithm"], "")

    param_lines = []
    if cfg["algorithm"] in ("kmeans", "agglomerative", "gmm"):
        param_lines.append(f"**Segments (k):** {cfg['n_clusters']} — the number "
                           f"of distinct groups the model will create.")
    if cfg["algorithm"] == "agglomerative":
        param_lines.append(f"**Linkage:** {cfg['linkage']} — the rule used to "
                           f"decide which groups merge first.")
    if cfg["algorithm"] == "dbscan":
        param_lines.append(f"**eps:** {cfg['eps']} — how close points must be to "
                           f"belong to the same dense region.")
        param_lines.append(f"**min_samples:** {cfg['min_samples']} — the smallest "
                           f"a dense region can be to count as a segment.")

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
        sil_read = (f"A silhouette of **{sil:.3f}** indicates {sil_text}. "
                    f"Values range from -1 (overlapping) to 1 (perfectly distinct).")
    else:
        sil_read = "Silhouette is not available for this configuration."

    st.markdown(algo_blurb)
    if param_lines:
        st.markdown("\n".join(f"- {line}" for line in param_lines))
    st.info(sil_read)


# ===========================================================================
# Section 4 - Model Comparison
# ===========================================================================
METRIC_GLOSSARY = [
    (
        "Silhouette Score",
        "Higher is better",
        "Measures how tightly grouped and how far apart the segments are. It "
        "compares each customer's distance to its own segment centre with its "
        "distance to the next nearest centre. Ranges from -1 (overlapping) to "
        "1 (perfectly distinct); values above ~0.5 are good. Needs at least two "
        "segments.",
    ),
    (
        "Davies-Bouldin Index",
        "Lower is better",
        "The average 'similarity' of each segment to its most similar neighbouring "
        "segment, where similarity combines spread and separation. A lower value "
        "means segments are compact and well separated; 0 would be ideal.",
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


def section_comparison(base: dict[str, Any]) -> None:
    section_header(
        "Model Comparison",
        "Compare candidate algorithms and choose an appropriate number of segments.",
    )

    X = base["X"]
    with st.spinner("Evaluating candidate algorithms..."):
        with _suppress_warnings():
            warnings.simplefilter("ignore")
            table = run_full_evaluation(X)

    st.markdown("##### Algorithm Comparison")
    comp_df = pd.DataFrame(table.rows)
    if "Algorithm" in comp_df.columns:
        comp_df["Algorithm"] = comp_df["Algorithm"].map(
            lambda a: ALGORITHMS.get(a, str(a))
        )
    for col in comp_df.columns:
        if col != "Algorithm":
            comp_df[col] = comp_df[col].apply(
                lambda v: fmt_metric(v, 3) if isinstance(v, (int, float, float)) and not isinstance(v, bool) else v
            )
    styled_metric_table(comp_df)

    recommendation = table.recommendation
    st.success(
        f"Recommended configuration: **{ALGORITHMS.get(recommendation, recommendation)}** "
        f"(highest composite score across the available metrics)."
        if recommendation
        else "No candidate produced valid metrics."
    )

    with st.expander("How to read these metrics"):
        st.caption(
            "Each metric summarises a different aspect of segment quality. "
            "Higher-is-better and lower-is-better are noted for every metric so "
            "the table can be interpreted without statistical background."
        )
        for name, direction, explanation in METRIC_GLOSSARY:
            st.markdown(f"**{name}** — *{direction}*")
            st.markdown(explanation)

    st.divider()
    st.markdown("##### Cluster Stability (K-Means)")
    st.caption(
        "K-Means is run 10 times with different random seeds. "
        "Pairwise Adjusted Rand Index (ARI) measures label agreement. "
        "1.0 = perfect agreement, 0.0 = random."
    )
    with st.spinner("Evaluating cluster stability..."):
        with _suppress_warnings():
            warnings.simplefilter("ignore")
            from src.analytics import compute_cluster_stability
            try:
                stab = compute_cluster_stability(X, n_clusters=5, n_runs=10)
                stab_cols = st.columns(4)
                with stab_cols[0]:
                    kpi_card("Mean ARI", f"{stab.mean_ari:.3f}", "Across 10 runs", "#16a34a")
                with stab_cols[1]:
                    kpi_card("Min ARI", f"{stab.min_ari:.3f}", "Worst agreement", "#dc2626")
                with stab_cols[2]:
                    kpi_card("Max ARI", f"{stab.max_ari:.3f}", "Best agreement", "#2563eb")
                with stab_cols[3]:
                    kpi_card("Std ARI", f"{stab.std_ari:.3f}", "Variability", "#7c3aed")
                st.info(stab.interpretation)
            except Exception as exc:
                st.warning(f"Stability analysis failed: {exc}")

    st.divider()
    st.markdown("##### K-Means Elbow & Silhouette Analysis")
    with st.spinner("Computing k-range diagnostics..."):
        with _suppress_warnings():
            warnings.simplefilter("ignore")
            kopt = evaluate_k_range(X)

    kdf = pd.DataFrame(
        {
            "k": kopt.k_values,
            "Inertia": kopt.inertias,
            "Silhouette": kopt.silhouette_scores,
        }
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kdf["k"], y=kdf["Inertia"], mode="lines+markers",
                             name="Inertia (WCSS)", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=kdf["k"], y=kdf["Silhouette"], mode="lines+markers",
                             name="Silhouette Score", yaxis="y2",
                             line=dict(color="#16a34a")))
    fig.add_vline(x=kopt.optimal_k, line_dash="dot",
                  line_color="#94a3b8",
                  annotation_text=f"optimal k = {kopt.optimal_k}",
                  annotation_position="top")
    fig = style_chart(fig, title="Elbow (inertia) and Silhouette vs k",
                      height=420)
    fig.update_layout(
        xaxis=dict(title="Number of segments (k)",
                   tickmode="array", tickvals=kdf["k"]),
        yaxis=dict(title="Inertia"),
        yaxis2=dict(title="Silhouette", overlaying="y", side="right",
                    showgrid=False),
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        f"Based on the silhouette analysis, the optimal number of segments is "
        f"**k = {kopt.optimal_k}** (peak silhouette score "
        f"{max(kopt.silhouette_scores):.3f}). Combine this with the elbow point "
        f"and the business-readable personas to choose a final configuration. "
        f"The recommended algorithm above ({ALGORITHMS.get(recommendation, 'n/a')}) optimises "
        f"the composite of Silhouette, Calinski-Harabasz, Davies-Bouldin and "
        f"Inertia for this dataset."
    )


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
    section_header(
        "Segment Explorer",
        "Meet each customer persona and read its profile and recommended action.",
    )

    if active.get("error") or not active.get("valid"):
        show_error("No valid segments available for the current configuration.")
        return

    personas = active["personas"]
    profiles_by_id = {p.cluster_id: p for p in active["profiles"]}
    cluster_ids = sorted(personas.keys())
    stats = _segment_stats(base, active["labels"])

    # ------------------------------------------------------------------ #
    # Persona overview cards (strongest visual hierarchy)
    # ------------------------------------------------------------------ #
    st.markdown("##### Customer Personas")
    st.caption("At a glance: who each segment is and how many customers it holds.")

    cards = sorted(cluster_ids, key=lambda c: profiles_by_id.get(c).customer_count,
                   reverse=True)
    cols = st.columns(3)
    for idx, cid in enumerate(cards):
        persona = personas[cid]
        prof = profiles_by_id.get(cid)
        color = PERSONA_COLORS.get(persona.key, "#94a3b8")
        count = prof.customer_count if prof else 0
        pct = prof.percentage if prof else 0.0
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div style="background:#ffffff;border:1px solid {BORDER};
                    border-top:4px solid {color};border-radius:10px;
                    padding:14px 16px;height:100%;margin-bottom:14px;
                    box-shadow:0 1px 2px rgba(15,23,42,0.04);">
                  <div style="font-size:11px;font-weight:700;color:{MUTED};
                    text-transform:uppercase;letter-spacing:0.05em;">
                    Segment {cid}</div>
                  <div style="font-size:16px;font-weight:700;color:{INK};
                    margin:4px 0 8px;">{persona.name}</div>
                  <div style="display:flex;gap:14px;">
                    <div>
                      <div style="font-size:20px;font-weight:700;color:{color};">
                        {count:,}</div>
                      <div style="font-size:11px;color:{MUTED};">customers</div>
                    </div>
                    <div>
                      <div style="font-size:20px;font-weight:700;color:{INK};">
                        {pct}%</div>
                      <div style="font-size:11px;color:{MUTED};">of base</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------ #
    # Detailed drill-down for one segment
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### Segment Detail")
    selection = st.selectbox(
        "Select a segment / persona",
        options=cluster_ids,
        format_func=lambda cid: f"Segment {cid} - {personas[cid].name}",
    )

    persona = personas[selection]
    prof = profiles_by_id.get(selection)
    insight = _PERSONA_INSIGHTS.get(persona.key)
    color = PERSONA_COLORS.get(persona.key, "#94a3b8")

    st.markdown(
        f"""
        <div style="border-left:4px solid {color};padding:4px 0 4px 14px;
            margin-bottom:10px;">
          <div style="font-size:22px;font-weight:700;color:{INK};">
            {persona.name}</div>
          <div style="font-size:13px;color:{MUTED};">Segment {selection}</div>
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

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            kpi_card("Customers", f"{prof.customer_count:,}", f"{prof.percentage}% of base", color)
        with m2:
            kpi_card("Avg Income", fmt_income(avg_income), "per customer", "#ea580c")
        with m3:
            kpi_card("Avg Spending", f"{avg_spending:,.1f}", "score (1-100)", "#7c3aed")
        with m4:
            kpi_card("Avg Age", f"{avg_age:,.0f}", f"top gender: {top_genre}", "#0891b2")

        st.markdown("###### Profile")
        st.write(prof.profile_summary)
        if prof.key_characteristics:
            st.markdown("**Key characteristics:** " + ", ".join(prof.key_characteristics))

        st.markdown("###### Recommended Action")
        st.success(persona.strategy)

    if insight is not None:
        st.markdown("###### Business Recommendations")
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("**Business interpretation**")
            st.write(insight.business_interpretation)
            st.markdown("**Potential opportunity**")
            st.write(insight.potential_opportunity)
        with b2:
            st.markdown("**Recommended marketing strategy**")
            st.write(insight.recommended_marketing_strategy)
            st.markdown("**Retention & engagement**")
            st.write(insight.retention_engagement_recommendation)

    # ------------------------------------------------------------------ #
    # Segment comparison vs overall population
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### Segment vs Overall Population")
    st.caption(
        "How each segment differs from the overall customer base. "
        "Values are standardised differences (Cohen's d): "
        "positive = above average, negative = below average."
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
        styled_metric_table(comp_display)
    else:
        st.info("Segment comparison is not available for this configuration.")

    # ------------------------------------------------------------------ #
    # Feature separation / importance
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### Feature Separation Analysis")
    st.caption(
        "ANOVA F-ratios showing which features most strongly differentiate "
        "the segments. Higher F-ratio = more important for distinguishing "
        "between clusters. This is the appropriate interpretation for "
        "clustering (not supervised feature importance)."
    )
    sep_data = active.get("feature_separation")
    if sep_data:
        sep_rows = [
            {"Feature": r.feature, "F-Ratio": r.f_ratio, "Rank": r.rank}
            for r in sorted(sep_data, key=lambda x: x.rank)
        ]
        sep_df = pd.DataFrame(sep_rows)
        styled_metric_table(sep_df)
    else:
        st.info("Feature separation analysis is not available for this configuration.")


# ===========================================================================
# Section 6 - New Customer Prediction
# ===========================================================================
def section_predict(base: dict[str, Any]) -> None:
    section_header(
        "New Customer Prediction",
        "Enter a customer's attributes to assign them to a segment and receive "
        "a tailored business recommendation.",
    )

    bundle = load_prediction_model()
    if bundle is None:
        show_error("Prediction model could not be loaded (dataset missing).")
        return
    if not bundle.get("personas"):
        show_error(
            "The loaded prediction model has no usable segments.",
            "Retrain it from the Clustering Lab with a valid configuration.",
        )
        return

    cfg_used = bundle.get("config", {})
    st.caption(
        f"Using deployed model: **{cfg_label(cfg_used)}**. "
        "The model is loaded once and reused - submitting the form does not "
        "retrain it."
    )

    # ------------------------------------------------------------------ #
    # Customer Information
    # ------------------------------------------------------------------ #
    st.markdown("##### Customer Information")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=10, max_value=100, value=35, step=1)
            genre = st.selectbox("Gender", options=["Male", "Female"], index=0)
        with c2:
            income = st.number_input(
                "Annual Income (k$)", min_value=1.0, max_value=200.0, value=60.0, step=1.0
            )
            spending = st.number_input(
                "Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1
            )
        submitted = st.form_submit_button("Predict Segment", type="primary")

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
                pred = predict_new(bundle, int(age), genre, float(income), float(spending))
            if pred is None or pred["persona"] is None:
                show_error(
                    "Could not assign this customer to a segment.",
                    "The model has no valid cluster centres. Retrain it from the "
                    "Clustering Lab.",
                )
            else:
                st.session_state.last_prediction = {
                    "pred": pred, "age": age, "genre": genre,
                    "income": income, "spending": spending,
                }

    # ------------------------------------------------------------------ #
    # Prediction Result
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("##### Prediction Result")
    pred_state = st.session_state.get("last_prediction")
    if not pred_state:
        st.info("Enter a customer's details above and choose **Predict Segment** "
                "to see their assigned persona and recommended strategy.")
        return

    pred = pred_state["pred"]
    persona = pred["persona"]
    insight = _PERSONA_INSIGHTS.get(persona.key)
    color = PERSONA_COLORS.get(persona.key, "#94a3b8")

    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid {BORDER};
            border-left:4px solid {color};border-radius:10px;
            padding:16px 18px;box-shadow:0 1px 2px rgba(15,23,42,0.04);">
          <div style="font-size:12px;font-weight:700;color:{MUTED};
            text-transform:uppercase;letter-spacing:0.05em;">Assigned persona</div>
          <div style="font-size:22px;font-weight:700;color:{INK};margin:4px 0;">
            {persona.name}</div>
          <div style="font-size:13px;color:{MUTED};">
            Segment {pred['cluster_id']} &nbsp;·&nbsp; distance to centre:
            {pred['distance']:.2f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("###### Explanation")
    st.write(persona.description)
    if insight is not None:
        st.markdown("###### Recommended Strategy")
        st.success(persona.strategy)
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
    section_header(
        "Export",
        "Download analysis artefacts as CSV for downstream reporting and BI tools.",
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

    # 2. Segment summary
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
            table = run_full_evaluation(base["X"])
    comp_df = pd.DataFrame(table.rows)
    if "Algorithm" in comp_df.columns:
        comp_df["Algorithm"] = comp_df["Algorithm"].map(
            lambda a: ALGORITHMS.get(a, str(a))
        )
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


# ===========================================================================
# App shell
# ===========================================================================
def main() -> None:
    st.set_page_config(
        page_title="Customer Segmentation Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.markdown(
        f"""
        <style>
        html, body, .stApp {{ font-family: {CHART_FONT}; color: {INK}; }}
        .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; }}
        h1, h2, h3 {{ color: {INK}; font-weight: 700; }}
        h2 {{ font-size: 1.45rem; }}
        h3 {{ font-size: 1.05rem; }}
        .stCaption {{ color: {MUTED}; }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px; border-bottom: 1px solid {BORDER};
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 42px; font-weight: 600; color: {MUTED};
        }}
        .stTabs [aria-selected="true"] {{ color: {PRIMARY}; }}
        section [data-testid="stExpander"] {{ border: 1px solid {BORDER}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Intelligent Customer Segmentation")
    st.caption(
        "A modular segmentation dashboard (K-Means, Agglomerative, DBSCAN and "
        "Gaussian Mixture) powered by the project's reusable ML pipeline."
    )

    # Initialise session state.
    if "cfg" not in st.session_state:
        st.session_state.cfg = dict(DEFAULT_CONFIG)
    if "active" not in st.session_state:
        st.session_state.active = None
    if "active_key" not in st.session_state:
        st.session_state.active_key = None
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    base = get_base()
    if base.get("error"):
        logger.error("Dataset loading failed: %s", base["error"])
        st.error("Could not load the dataset.")
        st.caption(base["error"])
        st.stop()
        return

    logger.info("App initialised with algorithm=%s", st.session_state.cfg["algorithm"])

    # Sidebar.
    with st.sidebar:
        st.markdown("### Configuration")
        cfg = st.session_state.cfg
        st.write(f"**Algorithm:** {ALGORITHMS.get(cfg['algorithm'], cfg['algorithm'])}")
        st.write(f"**Setup:** {cfg_label(cfg)}")
        st.divider()
        st.write(
            "Adjust the algorithm and parameters in the **Clustering Lab** tab, "
            "then apply them to update every section."
        )
        st.divider()
        if st.button("Reload data / clear cache"):
            st.cache_resource.clear()
            st.session_state.active = None
            st.session_state.active_key = None
            st.session_state.last_prediction = None
            st.rerun()

    active = get_active(base)

    tabs = st.tabs(
        [
            "Executive Overview",
            "Customer Analytics",
            "Clustering Lab",
            "Model Comparison",
            "Segment Explorer",
            "New Customer Prediction",
            "Export",
        ]
    )

    with tabs[0]:
        section_overview(base, active)
    with tabs[1]:
        section_analytics(base)
    with tabs[2]:
        section_lab(base)
    with tabs[3]:
        section_comparison(base)
    with tabs[4]:
        section_explorer(base, active)
    with tabs[5]:
        section_predict(base)
    with tabs[6]:
        section_export(base, active)


if __name__ == "__main__":
    main()
