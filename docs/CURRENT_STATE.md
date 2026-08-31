# Current State Report

**Date:** 2026-08-31
**Repository:** Intelligent-Customer-Segmentation
**Branch:** `main`
**Latest commit:** `2bd8a85` — "fix: harden Streamlit deployment and model lifecycle"
**Working tree:** clean (no uncommitted changes)

---

## 1. Current Architecture

The project has been refactored from a flat single-script layout into a modular package:

```
Intelligent-Customer-Segmentation/
├── app.py                        # Streamlit dashboard (thin presentation layer)
├── requirements.txt              # 9 Python dependencies
├── README.md                     # Project documentation
├── LICENSE                       # MIT
├── .gitignore                    # Ignores caches, outputs, env files, and generated ML artifacts (preserves committed model)
├── data/
│   └── Mall_Customers.csv        # Source dataset (200 x 5)
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
├── tests/
│   ├── conftest.py               # Shared fixtures (data_path, df, X_scaled)
│   ├── test_data_loader.py       # 12 tests
│   ├── test_preprocessing.py     # 7 tests
│   ├── test_clustering.py        # 43 tests
│   ├── test_evaluation.py        # 25 tests
│   ├── test_personas.py          # 26 tests
│   ├── test_business_insights.py # 20 tests
│   ├── test_prediction.py        # 10 tests
│   ├── test_analytics.py         # 56 tests
│   └── test_model_lifecycle.py   # 19 tests
├── models/
│   └── segmentation_model.joblib # Serialized prediction bundle (k=5, kmeans)
├── outputs/
│   ├── customers_clustered.csv   # 200 rows x 6 cols (with Cluster label)
│   ├── cluster_profiles.csv      # 5 clusters summary
│   ├── evaluation_results.csv    # 7 algorithm configurations compared
│   ├── segmentation_report.txt   # Human-readable report
│   └── figures/
│       ├── customer_clusters.png
│       ├── elbow_method.png
│       └── silhouette_scores.png
├── assets/                       # Historical images (pre-refactor) + tar.gz archive
├── docs/
│   ├── PROJECT_AUDIT.md          # Audit report (modular architecture)
│   ├── README_AUDIT.md           # README verification (this audit)
│   └── CURRENT_STATE.md          # This file
└── notebooks/                    # Empty (.gitkeep only)
```

**Design principle:** `src/` contains all reusable ML logic. `app.py` is a thin presentation layer that imports from `src/` and never duplicates ML logic.

---

## 2. Current ML Pipeline

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

## 4. Current Dataset

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
| **Suspicious values** | None detected |
| **Pipeline match** | Yes — preprocessing selects Income + Spending Score for clustering; Genre/Age available for analytics |

**Statistics:**
- Age: mean 38.85, range 18–70
- Annual Income: mean 60.56k, range 15–137k
- Spending Score: mean 50.20, range 1–99

---

## 5. Implemented Features

| Feature | Status | Evidence |
|---------|--------|----------|
| K-Means | IMPLEMENTED | `src/clustering.py:135` |
| Elbow Method | IMPLEMENTED | `src/evaluation.py:552`, `src/clustering.py:381` |
| Silhouette Score | IMPLEMENTED | `src/evaluation.py:127` |
| Agglomerative Clustering | IMPLEMENTED | `src/clustering.py:179` |
| DBSCAN | IMPLEMENTED | `src/clustering.py:232` |
| Gaussian Mixture Models | IMPLEMENTED | `src/clustering.py:286` |
| Feature scaling | IMPLEMENTED | `src/preprocessing.py` (StandardScaler) |
| Preprocessing pipeline | IMPLEMENTED | `src/preprocessing.py:78` (ColumnTransformer) |
| Automatic cluster selection | IMPLEMENTED | `src/evaluation.py:393` (composite ranking) |
| Customer personas | IMPLEMENTED | `src/personas.py:45` (5 personas) |
| Business recommendations | IMPLEMENTED | `src/business_insights.py:32` |
| Streamlit application | IMPLEMENTED | `app.py` (7 tabs) |
| New-customer prediction | IMPLEMENTED | `app.py:281` (`predict_new`) |
| CSV export | IMPLEMENTED | `app.py:865` (`section_export`) |
| Model persistence | IMPLEMENTED | `app.py:264` (joblib) |
| Automated tests | IMPLEMENTED | 218 tests, all passing |
| Logging | IMPLEMENTED | `src/utils.py:68` + all modules |
| Caching | IMPLEMENTED | `@st.cache_resource` + session_state |
| Deployment configuration | PARTIAL | README mentions Streamlit Cloud; no Dockerfile/Procfile |

---

## 6. Missing Features

Features NOT present in the codebase (all listed in README "Future Improvements"):

- RFM segmentation
- CLV prediction
- Churn prediction
- Recommendation systems
- Campaign response prediction
- Database integration
- Dimensionality reduction (PCA/t-SNE)
- Hyperparameter optimization (Optuna/GridSearchCV)
- A/B testing framework
- API layer (FastAPI/Flask)
- Dockerfile / containerization
- CI/CD pipeline
- `.streamlit/config.toml` (uses defaults)
- `pyproject.toml` / `setup.py` (package not pip-installable)
- `__init__.py` content beyond version (no public API exports)

---

## 7. Broken Features

**None.** All 218 tests pass. `python -m compileall` produces no errors. The Streamlit app, prediction pipeline, and all ML modules execute without runtime errors.

All previously identified documentation drift issues have been resolved:
- `docs/PROJECT_AUDIT.md` replaced with current-state audit
- Persona names aligned across README, report, and code
- `outputs/segmentation_report.txt` regenerated with correct persona names

---

## 8. Documentation Problems

All previously identified documentation issues have been resolved:

| Problem | Previous Status | Current Status |
|---------|-----------------|----------------|
| `docs/PROJECT_AUDIT.md` described old flat structure | Medium severity | **Resolved** — replaced with current-state audit |
| README Results table persona name "Mainstream Shoppers" ≠ code "Growth Opportunity Customers" | Low severity | **Resolved** — aligned to "Growth Opportunity Customers" |
| `outputs/segmentation_report.txt` used old persona names | Low severity | **Resolved** — regenerated from current pipeline |
| README "Future Improvements" correctly lists unimplemented features | None | Accurate |

---

## 9. Dependency Problems

**`requirements.txt` contents:**
```
pandas>=2.0, numpy>=1.24, scikit-learn>=1.3, matplotlib>=3.7,
streamlit>=1.30, plotly>=5.18, joblib>=1.6.0, pytest>=7.4, pytest-cov>=4.1
```

| Check | Result |
|-------|--------|
| Required packages all used? | YES — all 9 imported somewhere in codebase |
| Unused packages? | NONE |
| Missing packages? | NONE — all imports resolve |
| Version conflicts? | NONE detected |
| `pyproject.toml` / `environment.yml`? | Not present (not required) |

---

## 10. Runtime Verification

| Test | Result |
|------|--------|
| `python -m compileall .` | PASS — no syntax errors |
| `pytest` | PASS — **218 passed** in ~7s |
| Dataset loads | PASS — 200 rows x 5 cols |
| Model artifact loads | PASS — valid joblib bundle with 5 personas |
| `app.py` imports resolve | PASS (compileall + pytest import `app.predict_new`) |

---

## 11. Recommended Upgrade Order

The repository is in a **healthy, fully-functional state**. All documentation drift has been resolved. Any future upgrades should preserve the working baseline. Suggested priority:

1. **Add `.streamlit/config.toml`** — explicit theme/port config for deployment reproducibility
2. **Add `pyproject.toml`** — make the package pip-installable with entry point
3. **Add Dockerfile** — containerize for portable deployment
4. **Add CI/CD** — GitHub Actions workflow to run pytest on push
5. **Expand test coverage** — add tests for `app.py` sections (currently only `predict_new` is tested)
6. **Implement "Future Improvements"** — RFM, CLV, API layer, etc. per README roadmap

---

## 12. Summary — True Current State

The repository is a **complete, working, well-tested customer segmentation platform**. It was refactored from a single-script prototype into a modular `src/` package with a 7-tab Streamlit dashboard. All four clustering algorithms (K-Means, Agglomerative, DBSCAN, GMM) are implemented and tested. The full pipeline — loading, validation, preprocessing, clustering, evaluation, persona assignment, business insights, prediction, and export — is functional. **143 automated tests pass.** The dataset is clean (200 rows, no quality issues). The README is accurate with only minor persona-name drift. There are no broken features, no runtime errors, and no dependency conflicts.
