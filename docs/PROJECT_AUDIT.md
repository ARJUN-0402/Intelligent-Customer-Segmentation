# Project Audit Report

**Date:** 2026-08-30
**Repository:** Intelligent-Customer-Segmentation
**Phase:** 1 — Repository Audit, Cleanup & Baseline

---

## 1. Current Architecture

```
Intelligent-Customer-Segmentation/
├── .gitignore                          # Standard Python gitignore
├── README.md                           # Project documentation (lowercase filename)
├── customer_segmentation.py            # Main Python script
├── Mall_Customers.csv                  # Dataset
├── age_distribution.png                # Generated visualization
├── annual_income_k_distribution.png    # Generated visualization
├── correlation_heatmap.png             # Generated visualization
├── customer_clusters.png               # Generated visualization (only one script produces)
├── elbow_method.png                    # Generated visualization
├── genre_distribution.png              # Generated visualization
├── pair_plot.png                       # Generated visualization
├── spending_score_1-100_distribution.png # Generated visualization
├── Customer_Segmentation_Project.tar.gz # Archive of previous version
└── docs/
    └── PROJECT_AUDIT.md                # This file
```

### Technology Stack
- **Language:** Python 3.x
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Algorithm:** K-Means Clustering
- **Preprocessing:** StandardScaler

---

## 2. Current ML Workflow

The current `customer_segmentation.py` implements a minimal workflow:

1. **Data Loading:** Read `Mall_Customers.csv` with pandas
2. **Feature Selection:** Extract `Annual Income (k$)` and `Spending Score (1-100)`
3. **Scaling:** Apply `StandardScaler` to normalize features
4. **Clustering:** Apply K-Means with k=5, init='k-means++', random_state=42
5. **Analysis:** Print cluster centers (inverse transformed) and group means
6. **Visualization:** Generate a single scatter plot (`customer_clusters.png`)
7. **Business Insights:** Provided only as code comments (not printed or documented)

### What the Script Does NOT Do
- No Exploratory Data Analysis (EDA) plots
- No Elbow Method computation or plotting
- No Silhouette Score calculation or display
- No pair plot generation
- No correlation heatmap generation
- No distribution plots

---

## 3. Dataset Summary

| Property | Value |
|----------|-------|
| **Shape** | 200 rows x 5 columns |
| **Source** | Mall_Customers.csv |

### Columns

| Column | Type | Unique Values | Description |
|--------|------|---------------|-------------|
| CustomerID | int64 | 200 | Unique customer identifier |
| Genre | object (str) | 2 | Gender (Male/Female) |
| Age | int64 | 51 | Customer age (18-70) |
| Annual Income (k$) | int64 | 64 | Annual income in thousands (15-137) |
| Spending Score (1-100) | int64 | 84 | Spending score (1-99) |

### Data Quality

| Check | Result |
|-------|--------|
| Missing Values | 0 (none) |
| Duplicate Rows | 0 (none) |
| Data Types | All correct (int64 for numeric, object for categorical) |

### Basic Statistics

| Statistic | CustomerID | Age | Annual Income (k$) | Spending Score (1-100) |
|-----------|------------|-----|-------------------|------------------------|
| Mean | 100.50 | 38.85 | 60.56 | 50.20 |
| Std | 57.88 | 13.97 | 26.26 | 25.82 |
| Min | 1.00 | 18.00 | 15.00 | 1.00 |
| Max | 200.00 | 70.00 | 137.00 | 99.00 |

---

## 4. Existing Features

### Implemented
- [x] Data loading from CSV
- [x] Feature selection (Income + Spending Score)
- [x] StandardScaler normalization
- [x] K-Means clustering (k=5)
- [x] Cluster label assignment
- [x] Cluster center calculation
- [x] Basic group statistics
- [x] Single scatter plot visualization
- [x] Business insights as code comments

### Claimed but Not Implemented
- [ ] EDA visualizations (distributions, pair plot, heatmap)
- [ ] Elbow Method analysis
- [ ] Silhouette Score calculation
- [ ] Automated optimal k selection

---

## 5. Identified Problems

### Critical Issues

| # | Issue | Description | Impact |
|---|-------|-------------|--------|
| 1 | **README vs Script Mismatch** | README claims 8 visualizations but script only generates 1 | Documentation is misleading |
| 2 | **Missing EDA Code** | No code for distributions, pair plots, heatmaps | Cannot reproduce full analysis |
| 3 | **Missing Elbow Method** | No elbow method computation or visualization | Cannot verify k=5 is optimal |
| 4 | **Missing Silhouette Analysis** | `silhouette_score` imported but never called | No cluster quality validation |

### Code Quality Issues

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| 5 | **Unused Import** | Line 4 | `seaborn` imported as `sns` but never used |
| 6 | **Unused Import** | Line 7 | `silhouette_score` imported but never called |
| 7 | **Missing n_init** | Line 22 | KMeans `n_init` not set (causes warning in sklearn >= 1.4) |
| 8 | **Hardcoded Path** | Line 10 | CSV path hardcoded as `'Mall_Customers.csv'` |
| 9 | **Insights Not Printed** | Lines 54-67 | Business insights only in comments, not output |

### Structural Issues

| # | Issue | Description |
|---|-------|-------------|
| 10 | **No requirements.txt** | Dependencies not documented |
| 11 | **No Project Structure** | Flat directory, no src/tests/config folders |
| 12 | **No Configuration** | No config file for paths, hyperparameters |
| 13 | **Archive Redundancy** | `Customer_Segmentation_Project.tar.gz` contains identical files |

### Archive Analysis

The `Customer_Segmentation_Project.tar.gz` contains:
- An older version of `customer_segmentation.py` (identical to current)
- An older version of `readme.md` (different from current README.md)
- Identical copies of all 8 PNG visualizations
- Identical copy of `Mall_Customers.csv`

**Conclusion:** The archive is a backup snapshot. All PNG files exist in both the archive and the working directory with identical sizes, confirming they were generated by a different (more complete) version of the script that is NOT in the repository.

---

## 6. Silhouette Score Analysis

Computed silhouette scores for k=2 to k=10:

| k | Silhouette Score |
|---|-----------------|
| 2 | 0.3213 |
| 3 | 0.4666 |
| 4 | 0.4939 |
| **5** | **0.5547** (highest) |
| 6 | 0.5399 |
| 7 | 0.5281 |
| 8 | 0.4552 |
| 9 | 0.4571 |
| 10 | 0.4432 |

**Finding:** k=5 is indeed optimal based on silhouette score, confirming the current implementation's choice.

### Cluster Distribution (k=5)

| Cluster | Count | Avg Income | Avg Spending | Interpretation |
|---------|-------|------------|--------------|----------------|
| 0 | 81 | 55.3k | 49.5 | Average income, average spending |
| 1 | 39 | 86.5k | 82.1 | High income, high spending |
| 2 | 22 | 25.7k | 79.4 | Low income, high spending |
| 3 | 35 | 88.2k | 17.1 | High income, low spending |
| 4 | 23 | 26.3k | 20.9 | Low income, low spending |

---

## 7. Recommended Improvements

### High Priority
1. **Implement complete EDA pipeline** - distributions, pair plot, heatmap
2. **Implement Elbow Method** - compute and plot inertia for k=1-10
3. **Implement Silhouette Analysis** - compute and display scores
4. **Add `n_init` parameter** - set `n_init=10` to avoid sklearn warnings
5. **Remove unused imports** - clean up seaborn and silhouette_score if not used

### Medium Priority
6. **Create `requirements.txt`** - document all dependencies
7. **Remove or archive tar.gz** - decide if backup is needed
8. **Fix README** - ensure it matches actual script capabilities
9. **Print business insights** - output insights instead of just comments

### Low Priority
10. **Add project structure** - src/, tests/, config/ directories
11. **Add configuration file** - centralize paths and hyperparameters
12. **Add logging** - replace print statements with proper logging
13. **Add docstrings** - document functions

---

## 8. Proposed Final Architecture

```
Intelligent-Customer-Segmentation/
├── README.md                           # Updated documentation
├── requirements.txt                    # Dependencies
├── config/
│   └── config.yaml                     # Paths, hyperparameters
├── data/
│   └── Mall_Customers.csv              # Dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Data loading and validation
│   ├── eda.py                          # Exploratory data analysis
│   ├── preprocessing.py                # Feature scaling, selection
│   ├── clustering.py                   # K-Means implementation
│   ├── evaluation.py                   # Elbow, Silhouette
│   └── visualization.py                # Plot generation
├── notebooks/
│   └── 01_eda.ipynb                    # Jupyter notebook for exploration
├── output/
│   ├── figures/                        # Generated visualizations
│   └── results/                        # Cluster assignments, metrics
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_clustering.py
│   └── test_preprocessing.py
└── docs/
    ├── PROJECT_AUDIT.md                # This file
    └── ARCHITECTURE.md                 # Architecture documentation
```

---

## 9. Files Inspected

| File | Size | Status | Notes |
|------|------|--------|-------|
| `.gitignore` | 4905 bytes | Retained | Standard Python gitignore |
| `README.md` | 2329 bytes | Update needed | Claims features not in script |
| `customer_segmentation.py` | 2955 bytes | Refactor needed | Incomplete implementation |
| `Mall_Customers.csv` | 3980 bytes | Retained | Clean dataset, no issues |
| `age_distribution.png` | 23837 bytes | Generated artifact | Cannot regenerate with current script |
| `annual_income_k_distribution.png` | 30489 bytes | Generated artifact | Cannot regenerate with current script |
| `correlation_heatmap.png` | 32370 bytes | Generated artifact | Cannot regenerate with current script |
| `customer_clusters.png` | 62081 bytes | Generated artifact | Only one script can produce |
| `elbow_method.png` | 29260 bytes | Generated artifact | Cannot regenerate with current script |
| `genre_distribution.png` | 12746 bytes | Generated artifact | Cannot regenerate with current script |
| `pair_plot.png` | 216216 bytes | Generated artifact | Cannot regenerate with current script |
| `spending_score_1-100_distribution.png` | 29421 bytes | Generated artifact | Cannot regenerate with current script |
| `Customer_Segmentation_Project.tar.gz` | 411920 bytes | Review needed | Backup archive with identical files |

---

## 10. Files Retained/Removed Decision

| File | Decision | Reason |
|------|----------|--------|
| `Customer_Segmentation_Project.tar.gz` | **Retain for now** | May contain historical reference; can be removed in cleanup phase |
| All PNG files | **Retain for now** | Cannot be regenerated with current script; will be regenerated after script completion |
| `customer_segmentation.py` | **Refactor** | Core file, needs significant expansion |
| `Mall_Customers.csv` | **Retain** | Essential dataset |
| `README.md` | **Update** | Needs to match actual implementation |
| `.gitignore` | **Retain** | Appropriate configuration |

---

## 11. Summary

### Current State
- Minimal viable K-Means implementation
- Only 1 of 8 claimed visualizations is reproducible
- No EDA, no model evaluation beyond basic statistics
- Code has unused imports and missing parameters

### Key Findings
1. The repository has **8 PNG visualizations** that were generated by a script version NOT in the repository
2. The current script only generates `customer_clusters.png`
3. k=5 is validated as optimal via silhouette score analysis
4. Dataset is clean (no missing values, no duplicates)
5. Business insights exist only as code comments

### Recommended Next Phase
**Phase 2 — Implement Complete ML Pipeline:**
1. Implement full EDA (distributions, pair plot, heatmap)
2. Implement Elbow Method visualization
3. Implement Silhouette Score analysis
4. Generate all 8 visualizations programmatically
5. Print business insights as output
6. Fix code quality issues (unused imports, missing parameters)

---

*End of Audit Report*
