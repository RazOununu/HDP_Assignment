# HDP Assignment — Implementation Plan

## Overview

**Dataset:** UCI Heart Disease (296 samples × 14 columns after cleaning)  
**Goal:** High-dimensional data analysis — not classification. Extract and document insights about the dataset's geometric and statistical properties.  
**Deliverables:** Public GitHub repo + PDF report in academic English + documented AI tool usage.

---

## Current State

| File | Description |
|------|-------------|
| `heart.csv` | Raw dataset |
| `heart_clean.csv` | After dropping faulty rows (ca=4, thal=0) |
| `heart_standardized.csv` | Continuous columns z-score standardized |
| `clean_data.py` | Cleaning script |
| `standardize.py` | Standardization script |
| `data_desctription/describe_data.py` | Basic EDA — histograms, distributions |

---

## Implementation Steps

### Step 0 — Feature Matrix Preparation
**File:** `analysis/00_prepare_features.py`

One-hot encode all categorical columns so every feature is numerical. This expands the dataset from 13 features to ~20 fully numerical dimensions, which is required for PCA, Mahalanobis, k-NN, and all other downstream analyses.

**Output:** `heart_features.csv`

---

### Step 1 — PCA Analysis
**File:** `analysis/01_pca_analysis.py`

Understand the linear structure and effective dimensionality of the data.

- Scree plot — explained variance per principal component
- Cumulative explained variance — how many PCs reach 95%?
- 2D and 3D scatter plots coloured by label (healthy vs disease)
- Loadings heatmap — which features dominate each PC

**Figures:** `figures/pca_scree.png`, `figures/pca_scatter_2d.png`, `figures/pca_scatter_3d.png`, `figures/pca_loadings.png`

---

### Step 2 — Intrinsic Dimensionality Estimation
**File:** `analysis/02_intrinsic_dim.py`

Estimate the true (nonlinear) dimensionality beyond what PCA reveals.

- **TwoNN estimator** (Facco et al. 2017) — ratio of first two k-NN distances
- **MLE estimator** (Levina & Bickel) — maximum-likelihood over k-NN distances
- **Correlation Dimension** — log-log plot of pair-distance count vs radius
- Comparison table of all three estimates

**Figures:** `figures/intrinsic_dim_twonn.png`, `figures/corr_dim.png`

---

### Step 3 — Random Projections / Johnson-Lindenstrauss
**File:** `analysis/03_random_projections.py`

Demonstrate the JL lemma empirically — random projections preserve pairwise distances.

- Project data to k = 2, 3, 5, 10 dimensions using random Gaussian matrices
- Compare pairwise distance distributions before and after projection
- Plot distortion ε as a function of projection dimension k

**Figures:** `figures/jl_distance_preservation.png`, `figures/jl_distortion.png`

---

### Step 4 — Random Matrix Theory / Marchenko-Pastur
**File:** `analysis/04_rmt_analysis.py`

Determine how much of the covariance spectrum is signal versus noise.

- Compute eigenvalue distribution of the sample covariance matrix
- Overlay the theoretical Marchenko-Pastur density (γ = D/N, σ²=1)
- Identify eigenvalues exceeding the MP upper edge — these represent true signal components

**Figures:** `figures/rmt_spectrum.png`

---

### Step 5 — Anomaly Detection
**File:** `analysis/05_anomaly_detection.py`

Identify outliers using distance-based and density-based methods motivated by HDP theory.

- **Mahalanobis distance** — flag top 5%, compare to χ² theoretical distribution
- **Local Outlier Factor (LOF)** — k=20
- **Isolation Forest** — contamination=0.05
- Cross-method comparison: which samples are flagged by multiple methods?

**Figures:** `figures/anomaly_mahal.png`, `figures/anomaly_lof.png`, `figures/anomaly_comparison.png`

---

### Step 6 — Nonlinear Visualisations
**File:** `analysis/06_visualisations.py`

Reveal nonlinear structure in the data using manifold learning methods.

- **t-SNE** — perplexity = 10, 30, 50 (robustness check)
- **UMAP** — n_neighbors=15, min_dist=0.1
- Both coloured by label and by age quartile

**Figures:** `figures/tsne_umap.png`

---

### Step 7 — Fisher Separability & Hubness
**File:** `analysis/07_fisher_separability.py`

Measure class separability and hubness as dimensionality increases.

- **Fisher Discriminant Ratio** in PCA subspaces of increasing dimension (1 → D)
- **Hubness analysis** — distribution of how many times each point appears as a k-NN of others (a high-dimensional geometry phenomenon)

**Figures:** `figures/fisher_vs_dim.png`, `figures/hubness.png`

---

### Step 8 — Report
**File:** `report/report.ipynb` → exported to `report/report.pdf`

A Jupyter notebook written in academic English that imports all analysis modules, embeds all figures with captions, and presents findings in a structured report format.

**Sections:**
1. Introduction
2. Dataset Description
3. Methods
4. Results (one subsection per analysis step)
5. Discussion
6. Conclusion
7. AI Tool Documentation

Export command: `jupyter nbconvert --to pdf report/report.ipynb`

---

## Final Directory Structure

```
HDP_Assignment/
├── heart.csv
├── heart_clean.csv
├── heart_standardized.csv
├── heart_features.csv        ← one-hot encoded feature matrix
├── clean_data.py
├── standardize.py
├── data_desctription/
│   ├── describe_data.py
│   └── *.png
├── analysis/
│   ├── 00_prepare_features.py
│   ├── 01_pca_analysis.py
│   ├── 02_intrinsic_dim.py
│   ├── 03_random_projections.py
│   ├── 04_rmt_analysis.py
│   ├── 05_anomaly_detection.py
│   ├── 06_visualisations.py
│   └── 07_fisher_separability.py
├── figures/
│   └── *.png
└── report/
    ├── report.ipynb
    └── report.pdf
```

---

## Dependencies

```bash
pip install scikit-learn umap-learn matplotlib seaborn scipy numpy pandas jupyter nbconvert
```
