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
    DEFAULT_RANDOM_STATE,
    FEATURE_COLUMNS,
    GENRE,
    MAX_K,
    MIN_K,
    MODELS_DIR,
    SPENDING_SCORE,
    ensure_directory,
)

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
    "linkage": "ward",
    "eps": 0.5,
    "min_samples": 4,
}

MODEL_PATH = MODELS_DIR / "segmentation_model.joblib"

# Business-friendly column display names.
DISPLAY = {
    AGE: "Age",
    ANNUAL_INCOME: "Annual Income (k$)",
    SPENDING_SCORE: "Spending Score (1-100)",
    GENRE: "Gender",
}

PALETTE = [
    "#2563eb", "#16a34a", "#dc2626", "#9333ea",
    "#ea580c", "#0891b2", "#ca8a04", "#db2777",
]


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
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        return {"error": str(exc)}

    pre = CustomerDataPreprocessor(
        numeric_features=FEATURE_COLUMNS,
        categorical_features=[],
        drop_duplicates=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, cleaned = pre.fit_transform(raw)
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
    """Run the active clustering configuration and derive personas/profiles."""
    X = base["X"]
    cleaned = base["cleaned"]
    pre = base["pre"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = run_clustering(X, algorithm=cfg["algorithm"], **_build_kwargs(cfg))
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "algorithm": cfg["algorithm"]}

    result.feature_names = pre.output_feature_names

    centers_orig: Optional[pd.DataFrame] = None
    if result.cluster_centers is not None:
        try:
            centers_orig = pre.inverse_transform_centers(result.cluster_centers)
        except Exception:  # noqa: BLE001
            centers_orig = None

    personas: dict[int, Any] = {}
    if centers_orig is not None and len(centers_orig):
        try:
            personas = assign_personas_from_data(centers_orig, cleaned)
        except Exception:  # noqa: BLE001
            personas = {}

    # Metrics (silhouette is the one we surface everywhere).
    from src.evaluation import evaluate_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eval_res = evaluate_model(X, result)
    sil = eval_res.metrics["Silhouette Score"].value

    profiles: list[Any] = []
    if personas:
        profiles = build_persona_profiles(
            cleaned, result.labels, cluster_centers=centers_orig, personas=personas
        )

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
    with warnings.catch_warnings():
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
    return {
        "pre": pre,
        "scaled_centers": scaled_centers,
        "centers_orig": centers_orig,
        "personas": personas,
        "algorithm": cfg["algorithm"],
        "config": dict(cfg),
    }


@st.cache_resource(show_spinner="Loading prediction model...")
def load_prediction_model() -> Optional[dict[str, Any]]:
    """Load the persisted prediction model, training + saving it if absent."""
    base = get_base()
    if base.get("error"):
        return None
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:  # noqa: BLE001
            pass
    bundle = train_bundle(base, DEFAULT_CONFIG)
    ensure_directory(MODELS_DIR)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def predict_new(
    bundle: dict[str, Any], age: int, genre: str, income: float, spending: float
) -> Optional[dict[str, Any]]:
    """Map a new customer to a cluster + persona without retraining."""
    if not bundle or not bundle.get("personas"):
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
    Xs = bundle["pre"].transform(row)
    centers = bundle["scaled_centers"]
    if centers is None or len(centers) == 0:
        return None
    valid = ~np.isnan(centers).any(axis=1)
    if not np.any(valid):
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
def kpi_card(label: str, value: str, sub: str = "", color: str = "#2563eb") -> None:
    """Render a single KPI card via HTML for a polished, business look."""
    html = f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-left:
        4px solid {color};border-radius:8px;padding:16px 18px;height:100%;">
      <div style="font-size:13px;color:#6b7280;font-weight:600;
        text-transform:uppercase;letter-spacing:0.04em;">{label}</div>
      <div style="font-size:26px;color:#111827;font-weight:700;
        margin-top:6px;line-height:1.1;">{value}</div>
      <div style="font-size:12px;color:#9ca3af;margin-top:4px;">{sub}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def styled_metric_table(df: pd.DataFrame) -> None:
    """Render a dataframe with light styling (no emoji clutter)."""
    st.dataframe(df, use_container_width=True, hide_index=True)


def show_error(msg: str, hint: str = "") -> None:
    st.error(msg)
    if hint:
        st.caption(hint)


# ===========================================================================
# Section 1 - Executive Overview
# ===========================================================================
def section_overview(base: dict[str, Any], active: dict[str, Any]) -> None:
    st.header("Executive Overview")
    st.caption(
        "High-level segmentation metrics for the currently selected "
        "clustering configuration."
    )

    raw = base["raw"]
    cfg = st.session_state.cfg

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Customers", f"{len(raw):,}", "After deduplication")
    with col2:
        n_clusters = active.get("n_clusters", 0) if not active.get("error") else 0
        kpi_card(
            "Clusters",
            str(n_clusters),
            f"{active.get('noise', 0)} noise points" if cfg["algorithm"] == "dbscan" else "Segments found",
            color="#16a34a",
        )
    with col3:
        avg_income = raw[ANNUAL_INCOME].mean()
        kpi_card("Avg Income", f"${avg_income:,.1f}k", "Across all customers", color="#ea580c")
    with col4:
        avg_spend = raw[SPENDING_SCORE].mean()
        kpi_card("Avg Spending", f"{avg_spend:,.1f}", "Spending score (1-100)", color="#9333ea")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        kpi_card(
            "Selected Algorithm",
            ALGORITHMS.get(cfg["algorithm"], cfg["algorithm"]),
            "Configurable in the Clustering Lab",
            color="#0891b2",
        )
    with c2:
        kpi_card(
            "Clustering Configuration",
            cfg_label(cfg),
            f"Silhouette: {active.get('silhouette'):.3f}"
            if active.get("silhouette") is not None
            else "Silhouette: n/a",
            color="#ca8a04",
        )

    if active.get("error"):
        st.divider()
        show_error(
            "The selected configuration failed.",
            active["error"],
        )


# ===========================================================================
# Section 2 - Customer Analytics
# ===========================================================================
def section_analytics(base: dict[str, Any]) -> None:
    st.header("Customer Analytics")
    st.caption("Exploratory, interactive views of the underlying customer base.")

    raw = base["raw"]

    st.subheader("Distributions")
    d1, d2, d3 = st.columns(3)
    with d1:
        fig = px.histogram(
            raw, x=AGE, nbins=20, color_discrete_sequence=["#2563eb"],
            title="Age Distribution",
        )
        fig.update_layout(template="plotly_white", margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        fig = px.histogram(
            raw, x=ANNUAL_INCOME, nbins=20, color_discrete_sequence=["#ea580c"],
            title="Annual Income Distribution",
        )
        fig.update_layout(template="plotly_white", margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
    with d3:
        fig = px.histogram(
            raw, x=SPENDING_SCORE, nbins=20, color_discrete_sequence=["#9333ea"],
            title="Spending Score Distribution",
        )
        fig.update_layout(template="plotly_white", margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Composition & Relationships")
    r1, r2 = st.columns(2)
    with r1:
        counts = raw[GENRE].value_counts().reset_index()
        counts.columns = [GENRE, "Count"]
        fig = px.pie(
            counts, names=GENRE, values="Count", title="Gender Distribution",
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(template="plotly_white", margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
    with r2:
        corr = raw[[AGE, ANNUAL_INCOME, SPENDING_SCORE]].corr().round(2)
        fig = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r",
            title="Feature Correlations",
        )
        fig.update_layout(template="plotly_white", margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Relationships")
    rel_mode = st.radio(
        "Colour points by",
        ["Spending vs Income (by Gender)", "Spending vs Income (by Cluster)"],
        horizontal=True,
    )
    if rel_mode.startswith("Spending vs Income (by Gender)"):
        fig = px.scatter(
            raw, x=ANNUAL_INCOME, y=SPENDING_SCORE, color=GENRE,
            hover_data=[AGE, GENRE], title="Income vs Spending Score",
            color_discrete_sequence=PALETTE,
        )
    else:
        active = st.session_state.active
        cleaned = base["cleaned"].copy()
        if active.get("labels") is not None:
            cleaned["Cluster"] = active["labels"]
            fig = px.scatter(
                cleaned, x=ANNUAL_INCOME, y=SPENDING_SCORE, color="Cluster",
                hover_data=[AGE, GENRE], title="Income vs Spending (coloured by cluster)",
                color_continuous_scale="Turbo",
            )
        else:
            fig = px.scatter(
                raw, x=ANNUAL_INCOME, y=SPENDING_SCORE, hover_data=[AGE, GENRE],
                title="Income vs Spending Score",
            )
    fig.update_layout(template="plotly_white", margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# Section 3 - Clustering Lab
# ===========================================================================
def section_lab(base: dict[str, Any]) -> None:
    st.header("Clustering Lab")
    st.caption("Experiment with algorithms and parameters, then apply the result.")

    cfg = st.session_state.cfg
    algorithm = st.selectbox(
        "Algorithm",
        options=list(ALGORITHMS.keys()),
        format_func=lambda a: ALGORITHMS[a],
        index=list(ALGORITHMS.keys()).index(cfg["algorithm"]),
    )

    params: dict[str, Any] = {"algorithm": algorithm}
    if algorithm in ("kmeans", "agglomerative", "gmm"):
        params["n_clusters"] = st.slider(
            "Number of clusters (k)",
            min_value=MIN_K, max_value=MAX_K, value=cfg["n_clusters"],
        )
    if algorithm == "agglomerative":
        params["linkage"] = st.selectbox(
            "Linkage", ["ward", "complete", "average", "single"],
            index=["ward", "complete", "average", "single"].index(cfg["linkage"]),
        )
    if algorithm == "dbscan":
        c1, c2 = st.columns(2)
        with c1:
            params["eps"] = st.number_input(
                "eps (neighbourhood radius)", min_value=0.05, max_value=5.0,
                value=float(cfg["eps"]), step=0.05,
            )
        with c2:
            params["min_samples"] = st.number_input(
                "min_samples", min_value=1, max_value=50,
                value=int(cfg["min_samples"]), step=1,
            )

    # Carry over unchanged fields.
    params.setdefault("n_clusters", cfg["n_clusters"])
    params.setdefault("linkage", cfg["linkage"])
    params.setdefault("eps", cfg["eps"])
    params.setdefault("min_samples", cfg["min_samples"])

    applied = st.button("Apply Configuration", type="primary")
    if applied:
        st.session_state.cfg = params
        st.session_state.active = None
        st.session_state.active_key = None
        st.rerun()

    active = st.session_state.active
    if active.get("error"):
        st.divider()
        show_error("Clustering failed for this configuration.", active["error"])
        return

    labels = active["labels"]
    n_clusters = active["n_clusters"]
    centers_orig = active["centers_orig"]

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
    st.subheader("Cluster Visualization")
    cleaned = base["cleaned"].copy()
    cleaned["Cluster"] = labels

    fig = px.scatter(
        cleaned, x=ANNUAL_INCOME, y=SPENDING_SCORE, color="Cluster",
        hover_data=[AGE, GENRE], title=f"Customer Clusters ({cfg_label(cfg)})",
        color_continuous_scale="Turbo",
    )
    if centers_orig is not None and len(centers_orig):
        fig.add_trace(
            go.Scatter(
                x=centers_orig[ANNUAL_INCOME],
                y=centers_orig[SPENDING_SCORE],
                mode="markers",
                marker=dict(symbol="diamond", size=16, color="black",
                            line=dict(width=2, color="white")),
                name="Centroids",
            )
        )
    fig.update_layout(template="plotly_white", margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cluster Sizes")
        counts = pd.Series(labels).value_counts().sort_index()
        size_df = pd.DataFrame(
            {"Cluster": counts.index, "Customers": counts.values}
        )
        fig = px.bar(
            size_df, x="Cluster", y="Customers",
            color="Cluster", color_continuous_scale="Turbo",
            title="Customers per Cluster",
        )
        fig.update_layout(template="plotly_white", showlegend=False,
                          margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Model Metrics")
        metrics: dict[str, Any] = {}
        if active.get("silhouette") is not None:
            metrics["Silhouette Score"] = round(active["silhouette"], 4)
        if active.get("inertia") is not None:
            metrics["Inertia (WCSS)"] = round(active["inertia"], 2)
        metrics["Clusters (excl. noise)"] = int(n_clusters)
        if st.session_state.cfg["algorithm"] == "dbscan":
            metrics["Noise points"] = int(active.get("noise", 0))
        mdf = pd.DataFrame(
            {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
        )
        styled_metric_table(mdf)

    if centers_orig is not None and len(centers_orig):
        st.subheader("Cluster Centers (original scale)")
        center_df = centers_orig.copy()
        center_df.index = [f"Cluster {i}" for i in center_df.index]
        center_df = center_df.round(1)
        styled_metric_table(center_df.reset_index().rename(columns={"index": "Cluster"}))

    st.divider()
    st.subheader("Deploy this configuration")
    st.caption(
        "Save the active configuration as the deployed prediction model. "
        "The prediction form then reuses it without retraining."
    )
    if st.button("Retrain & Save Prediction Model"):
        bundle = train_bundle(base, st.session_state.cfg)
        ensure_directory(MODELS_DIR)
        joblib.dump(bundle, MODEL_PATH)
        st.cache_resource.clear()
        st.success(f"Prediction model saved to {MODEL_PATH.name}.")


# ===========================================================================
# Section 4 - Model Comparison
# ===========================================================================
def section_comparison(base: dict[str, Any]) -> None:
    st.header("Model Comparison")
    st.caption("Compare candidate algorithms and select an optimal k.")

    X = base["X"]
    with st.spinner("Evaluating candidate algorithms..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = run_full_evaluation(X)

    st.subheader("Algorithm Comparison")
    comp_df = pd.DataFrame(table.rows)
    styled_metric_table(comp_df)

    recommendation = table.recommendation
    st.success(
        f"Recommended configuration: **{recommendation}** "
        f"(highest composite score across available metrics)."
        if recommendation
        else "No candidate produced valid metrics."
    )

    st.divider()
    st.subheader("K-Means Elbow & Silhouette Analysis")
    with st.spinner("Computing k-range diagnostics..."):
        with warnings.catch_warnings():
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
    fig.update_layout(
        template="plotly_white",
        title="Elbow (inertia) and Silhouette vs k",
        xaxis=dict(title="Number of clusters (k)"),
        yaxis=dict(title="Inertia"),
        yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
        margin=dict(t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Based on the silhouette analysis, the optimal number of clusters is "
        f"**k = {kopt.optimal_k}** (peak silhouette score "
        f"{max(kopt.silhouette_scores):.3f}). Combine this with the elbow point "
        f"and the business-readable personas to choose a final configuration. "
        f"The recommended algorithm above ({recommendation or 'n/a'}) optimises "
        f"the composite of Silhouette, Calinski-Harabasz, Davies-Bouldin and "
        f"Inertia for this dataset."
    )


# ===========================================================================
# Section 5 - Segment Explorer
# ===========================================================================
def section_explorer(base: dict[str, Any], active: dict[str, Any]) -> None:
    st.header("Segment Explorer")
    st.caption("Drill into a single persona / cluster for detailed interpretation.")

    if active.get("error") or not active.get("valid"):
        show_error("No valid segments available for the current configuration.")
        return

    personas = active["personas"]
    profiles_by_id = {p.cluster_id: p for p in active["profiles"]}
    cluster_ids = sorted(personas.keys())

    selection = st.selectbox(
        "Select a cluster / persona",
        options=cluster_ids,
        format_func=lambda cid: f"Cluster {cid} - {personas[cid].name}",
    )

    persona = personas[selection]
    prof = profiles_by_id.get(selection)
    insight = _PERSONA_INSIGHTS.get(persona.key)

    # Per-cluster aggregates from the cleaned data (row-aligned with labels).
    labels = active["labels"]
    enriched = base["cleaned"].copy()
    enriched["Cluster"] = labels
    grp = enriched.groupby("Cluster")
    avg_income = float(grp[ANNUAL_INCOME].mean().get(selection, float("nan")))
    avg_spending = float(grp[SPENDING_SCORE].mean().get(selection, float("nan")))

    st.subheader(f"{persona.name}  (Cluster {selection})")
    st.write(persona.description)

    if prof is not None:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            kpi_card("Segment Size", f"{prof.customer_count:,}", "customers", "#2563eb")
        with m2:
            kpi_card("Percentage", f"{prof.percentage}%", "of base", "#16a34a")
        with m3:
            kpi_card("Avg Income", f"${avg_income:,.1f}k", "per customer", "#ea580c")
        with m4:
            kpi_card("Avg Spending", f"{avg_spending:,.1f}", "score (1-100)", "#9333ea")

        st.subheader("Profile")
        st.write(prof.profile_summary)
        if prof.key_characteristics:
            st.markdown("**Key characteristics:** " + ", ".join(prof.key_characteristics))

    if insight is not None:
        st.subheader("Business Recommendations")
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


# ===========================================================================
# Section 6 - New Customer Prediction
# ===========================================================================
def section_predict(base: dict[str, Any]) -> None:
    st.header("New Customer Prediction")
    st.caption(
        "Enter a customer's attributes to assign them to a segment and "
        "receive a tailored business recommendation."
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
    st.info(
        f"Using deployed model: **{cfg_label(cfg_used)}**. "
        "The model is loaded once and reused - submitting the form does not "
        "retrain it."
    )

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
        # Input validation.
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
            return

        with st.spinner("Assigning to segment..."):
            pred = predict_new(bundle, int(age), genre, float(income), float(spending))

        if pred is None or pred["persona"] is None:
            show_error(
                "Could not assign this customer to a segment.",
                "The model has no valid cluster centres. Retrain it from the "
                "Clustering Lab.",
            )
            return

        persona = pred["persona"]
        insight = _PERSONA_INSIGHTS.get(persona.key)
        st.success(
            f"Customer assigned to **Cluster {pred['cluster_id']} - "
            f"{persona.name}** (distance to centroid: {pred['distance']:.2f})."
        )

        st.subheader(persona.name)
        st.write(persona.description)

        if insight is not None:
            st.subheader("Business Recommendation")
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**Marketing strategy**")
                st.write(insight.recommended_marketing_strategy)
                st.markdown("**Opportunity**")
                st.write(insight.potential_opportunity)
            with r2:
                st.markdown("**Retention & engagement**")
                st.write(insight.retention_engagement_recommendation)
                st.markdown("**Interpretation**")
                st.write(insight.business_interpretation)


# ===========================================================================
# Section 7 - Export
# ===========================================================================
def section_export(base: dict[str, Any], active: dict[str, Any]) -> None:
    st.header("Export")
    st.caption("Download analysis artefacts as CSV for downstream use.")

    # 1. Segmented customers
    cleaned = base["cleaned"].copy()
    if active.get("labels") is not None:
        cleaned["Cluster"] = active["labels"]
        seg_csv = cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download segmented customers CSV",
            data=seg_csv,
            file_name="customers_clustered.csv",
            mime="text/csv",
        )

    # 2. Segment summary
    if active.get("profiles"):
        rows = []
        for p in active["profiles"]:
            persona = active["personas"].get(p.cluster_id)
            rows.append(
                {
                    "Cluster": p.cluster_id,
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
            "Download segment summary CSV",
            data=summary_csv,
            file_name="segment_summary.csv",
            mime="text/csv",
        )

    # 3. Model comparison
    with st.spinner("Building model comparison..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = run_full_evaluation(base["X"])
    comp_csv = pd.DataFrame(table.rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download model comparison CSV",
        data=comp_csv,
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Cluster Insights Report")
    if active.get("profiles") and active.get("personas"):
        profiles_df = build_cluster_profiles(base["cleaned"], active["labels"])
        insights = generate_cluster_insights(profiles_df, active["personas"])
        report = "\n\n".join(insights)
        st.download_button(
            "Download cluster insights TXT",
            data=report.encode("utf-8"),
            file_name="cluster_insights.txt",
            mime="text/plain",
        )
        with st.expander("Preview insights"):
            st.text(report)


# ===========================================================================
# App shell
# ===========================================================================
def main() -> None:
    st.set_page_config(
        page_title="Customer Segmentation Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        h1, h2, h3 { color: #111827; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Intelligent Customer Segmentation")
    st.caption(
        "A modular K-Means / Agglomerative / DBSCAN / GMM segmentation "
        "dashboard powered by the project's reusable ML pipeline."
    )

    # Initialise session state.
    if "cfg" not in st.session_state:
        st.session_state.cfg = dict(DEFAULT_CONFIG)
    if "active" not in st.session_state:
        st.session_state.active = None
    if "active_key" not in st.session_state:
        st.session_state.active_key = None

    base = get_base()
    if base.get("error"):
        st.error("Could not load the dataset.")
        st.caption(base["error"])
        st.stop()
        return

    # Sidebar.
    with st.sidebar:
        st.header("Configuration")
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
