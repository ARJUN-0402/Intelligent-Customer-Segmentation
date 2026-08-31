# Project Audit Report

**Date:** 2026-08-31
**Repository:** Intelligent-Customer-Segmentation
**Branch:** `main`
**Latest commit:** `2bd8a85`
**Phase:** 1 — Post-Implementation Audit & Cleanup

---

## 1. Current Architecture

```
Intelligent-Customer-Segmentation/
├── app.py                        # Streamlit dashboard (thin presentation layer)
├── requirements.txt              # 9 Python dependencies
├── README.md                     # Project documentation
├── LICENSE                       # MIT
├── .gitignore                    # Ignores outputs/, models/, __pycache__/
│
├── data/
│   └── Mall_Customers.csv        # Source dataset (200 rows x 5 columns)
│
├── src/
│   ├── __init__.py               # Package init (v0.2.0)
│   ├── utils.py                  # Constants, portable paths, logging
│   ├── data_loader.py            # CSV loading + schema validation
│   ├── preprocessing.py          # ColumnTransformer pipeline (impute/scale/encode)
│   ├── clustering.py             # K-Means, Agglomerative, DBSCAN, GMM + elbow
│   ├── evaluation.py             # Multi-metric comparison + model selection
│   ├── personas.py               # Cluster-to-persona mapping (5 personas)
│   ├── business_insights.py      # Business insight generation + reports
│   ├── analytics.py              # Segment analytics (Cohen's d, ANOVA, stability)
│   └── ui/
│       ├── __init__.py           # UI presentation components (cards, charts)
│       ├── cards.py
│       ├── charts.py
│       ├── components.py
│       └── styles.py
│
├── tests/
│   ├── conftest.py               # Shared pytest fixtures
│   ├── test_data_loader.py       # 12 tests
│   ├── test_preprocessing.py     # 7 tests
│   ├── test_clustering.py        # 43 tests
│   ├── test_evaluation.py        # 25 tests
│   ├── test_personas.py          # 26 tests
│   ├── test_business_insights.py # 20 tests
│   ├── test_prediction.py        # 10 tests
│   ├── test_analytics.py         # 56 tests
│   └── test_model_lifecycle.py   # 19 tests
│
├── models/
│   └── segmentation_model.joblib # Serialized prediction bundle (k=5, kmeans)
│
├── outputs/
│   ├── customers_clustered.csv   # 200 rows x 6 cols (with Cluster label)
│   ├── cluster_profiles.csv      # 5 clusters summary
│   ├── evaluation_results.csv    # 7 algorithm configurations compared
│   ├── segmentation_report.txt   # Human-readable report
│   └── figures/
│       ├── customer_clusters.png
│       ├── elbow_method.png
│       └── silhouette_scores.png
│
├── assets/                       # Historical images (pre-refactor) + tar.gz archive
├── docs/
│   ├── PROJECT_AUDIT.md          # This file
│   ├── README_AUDIT.md           # README verification against codebase
│   └── CURRENT_STATE.md          # Current-state assessment
└── notebooks/                    # Empty (.gitkeep only)
```

**Design principle:** `src/` contains all reusable ML logic. `app.py` is a thin presentation layer that imports from `src/` and never duplicates ML logic.

### Technology Stack
- **Language:** Python 3.x
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, streamlit, plotly, joblib, pytest, pytest-cov
- **Algorithms:** K-Means, Agglomerative Clustering, DBSCAN, Gaussian Mixture Model
- **Preprocessing:** StandardScaler via ColumnTransformer
- **Evaluation:** Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index, Inertia

---

## 2. Current ML Workflow

### Data Loading (`src/data_loader.py`)
- Loads CSV via `pd.read_csv`
- Validates: file exists, non-empty, required columns present, schema (numeric vs categorical) correct
- Custom exceptions: `FileNotFoundError_`, `EmptyDatasetError`, `MissingColumnError`, `SchemaValidationError`
- Logging via `logging.getLogger(__name__)`

### Preprocessing (`src/preprocessing.py`)
- `CustomerDataPreprocessor` wraps a `ColumnTransformer`
- Numeric path: `SimpleImputer(median)` → `StandardScaler`
- Categorical path: `SimpleImputer(most_frequent)` → `OneHotEncoder`
- Supports `fit`, `transform`, `fit_transform`, `inverse_transform_centers`
- Duplicate removal optional (default on)
- Default clustering features: `Annual Income (k$)` + `Spending Score (1-100)` (2D)

### Clustering (`src/clustering.py`)
- Four algorithms via unified `run_clustering` dispatcher:
  - **K-Means**: `sklearn.cluster.KMeans`, `n_init=10`, `k-means++`
  - **Agglomerative**: `sklearn.cluster.AgglomerativeClustering`, all 4 linkages
  - **DBSCAN**: `sklearn.cluster.DBSCAN`, configurable `eps`/`min_samples`
  - **GMM**: `sklearn.mixture.GaussianMixture`, all covariance types
- All return a `ClusterResult` dataclass (labels, centers, inertia, noise, bic/aic, valid flag)
- `evaluate_k_range` computes elbow + silhouette across k=2..10

### Evaluation (`src/evaluation.py`)
- Four metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin, Inertia (K-Means only)
- Metrics only computed when meaningful (>= 2 clusters)
- `compare_models` builds a `ComparisonTable`
- `select_best_clustering_model` ranks by normalized composite score (min-max normalized, lower-is-better reversed, averaged over available metrics, alphabetical tiebreak)
- `run_full_evaluation` runs a predefined suite of 7 configurations

### Personas (`src/personas.py`)
- 5 personas: High-Value Customers, Premium Savers, Budget-Conscious Spenders, Low-Engagement Customers, Growth Opportunity Customers
- Classification via income/spending quartile breakpoints derived from full dataset
- `assign_personas_from_data` uses global quartiles (not cluster-center quartiles)
- `build_persona_profiles` produces per-cluster stats + characteristics

### Business Insights (`src/business_insights.py`)
- `_PERSONA_INSIGHTS` maps each persona key to interpretation, opportunity, strategy, retention
- `build_cluster_profiles` aggregates per-cluster stats
- `generate_report` composes a full text report

---

## 3. Current Dashboard (`app.py`)

Seven Streamlit tabs, all backed by cached data/functions:

| Tab | Function | Key Features |
|-----|----------|-------------|
| Executive Overview | `section_overview` | KPI cards (customers, clusters, avg income/spending, algorithm, silhouette) |
| Customer Analytics | `section_analytics` | Plotly histograms, gender pie, correlation heatmap, scatter plots |
| Clustering Lab | `section_lab` | Algorithm selector, parameter controls, apply button, cluster viz with centroids, sizes, metrics, retrain & save |
| Model Comparison | `section_comparison` | `run_full_evaluation` table, recommendation, dual-axis elbow+silhouette chart |
| Segment Explorer | `section_explorer` | Persona drill-down, profile, business recommendations |
| New Customer Prediction | `section_predict` | Form (age, gender, income, spending) → nearest centroid → persona + recommendation |
| Export | `section_export` | Download segmented CSV, summary CSV, comparison CSV, insights TXT |

**Caching:** `@st.cache_resource` on `get_base` and `load_prediction_model`; `session_state` key caching on `get_active`.

**Prediction model:** Trained once, persisted via `joblib` to `models/segmentation_model.joblib`. Form submission does NOT retrain.

---

## 4. Dataset Summary

| Property | Value |
|----------|-------|
| **File** | `data/Mall_Customers.csv` |
| **Rows** | 200 |
| **Columns** | 5 |
| **Column names** | `CustomerID`, `Genre`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)` |
| **Data types** | int64 (4), str (1) |
| **Missing values** | 0 |
| **Duplicates** | 0 |
| **Numeric columns** | CustomerID, Age, Annual Income (k$), Spending Score (1-100) |
| **Categorical columns** | Genre (Male/Female) |

### Basic Statistics

| Statistic | Age | Annual Income (k$) | Spending Score (1-100) |
|-----------|-----|-------------------|------------------------|
| Mean | 38.85 | 60.56 | 50.20 |
| Std | 13.97 | 26.26 | 25.82 |
| Min | 18.00 | 15.00 | 1.00 |
| Max | 70.00 | 137.00 | 99.00 |

---

## 5. Implemented Features

| Feature | Status | Evidence |
|---------|--------|---------|
| Data loading + validation | IMPLEMENTED | `src/data_loader.py` |
| StandardScaler normalization | IMPLEMENTED | `src/preprocessing.py` |
| K-Means clustering | IMPLEMENTED | `src/clustering.py:135` |
| Agglomerative Clustering | IMPLEMENTED | `src/clustering.py:179` |
| DBSCAN | IMPLEMENTED | `src/clustering.py:232` |
| Gaussian Mixture Models | IMPLEMENTED | `src/clustering.py:286` |
| Elbow Method | IMPLEMENTED | `src/evaluation.py:552`, `src/clustering.py:381` |
| Silhouette Score | IMPLEMENTED | `src/evaluation.py:127` |
| Calinski-Harabasz Index | IMPLEMENTED | `src/evaluation.py:152` |
| Davies-Bouldin Index | IMPLEMENTED | `src/evaluation.py:175` |
| Automated model selection | IMPLEMENTED | `src/evaluation.py:393` |
| Customer personas (5) | IMPLEMENTED | `src/personas.py:45` |
| Business insights | IMPLEMENTED | `src/business_insights.py:32` |
| Streamlit dashboard (7 tabs) | IMPLEMENTED | `app.py` |
| New-customer prediction | IMPLEMENTED | `app.py` |
| CSV export | IMPLEMENTED | `app.py` |
| Model persistence (joblib) | IMPLEMENTED | `app.py` |
| Automated tests (218) | IMPLEMENTED | `tests/` — 9 test files, 218 cases |

---

## 6. Persona Catalog

The canonical persona names (from `src/personas.py` `PERSONA_CATALOG`):

| Key | Name | Description |
|-----|------|-------------|
| `vip` | High-Value Customers | High income and high spending - the ideal customer group. |
| `saver` | Premium Savers | High income but low spending - wealthy but cautious. |
| `impulsive` | Budget-Conscious Spenders | Low income but high spending - enthusiastic but risky. |
| `budget` | Low-Engagement Customers | Low income and low spending - cautious with money. |
| `mainstream` | Growth Opportunity Customers | Average income and average spending - the broad middle. |

### Classification Logic

Personas are assigned using income/spending quartile breakpoints derived from the full dataset. The `_PERSONA_MATRIX` maps `(income_band, spending_band)` tuples to persona keys:

| Income | Spending | Persona |
|--------|----------|---------|
| high | high | High-Value Customers |
| high | low | Premium Savers |
| low | high | Budget-Conscious Spenders |
| low | low | Low-Engagement Customers |
| mid | high | Budget-Conscious Spenders |
| high | mid | Premium Savers |
| low | mid | Low-Engagement Customers |
| mid | mid | Growth Opportunity Customers |

---

## 7. Test Coverage

| Test File | Tests |
|-----------|-------|
| `test_data_loader.py` | 12 |
| `test_preprocessing.py` | 7 |
| `test_clustering.py` | 43 |
| `test_evaluation.py` | 25 |
| `test_personas.py` | 26 |
| `test_business_insights.py` | 20 |
| `test_prediction.py` | 10 |
| `test_analytics.py` | 56 |
| `test_model_lifecycle.py` | 19 |
| **Total** | **218** |

All 218 tests pass. `python -m compileall .` produces no errors.

---

## 8. Runtime Verification

| Test | Result |
|------|--------|
| `python -m compileall .` | PASS — no syntax errors |
| `pytest` | PASS — 218 passed |
| Dataset loads | PASS — 200 rows x 5 cols |
| Model artifact loads | PASS — valid joblib bundle with 5 personas |
| `app.py` imports resolve | PASS |

---

## 9. Audit Findings (Resolved)

The following issues were identified in earlier audits and have been resolved:

1. **Persona name drift** — README and report used "Mainstream Shoppers" while code used "Growth Opportunity Customers". Aligned all artifacts to use the canonical code names.
2. **Stale `outputs/segmentation_report.txt`** — Regenerated from the current pipeline with correct persona names.
3. **Stale `docs/PROJECT_AUDIT.md`** — Replaced with current-state audit.

---

*End of Audit Report*
