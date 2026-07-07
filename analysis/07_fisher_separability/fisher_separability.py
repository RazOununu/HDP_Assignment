"""
Step 7 — Fisher Separability & Hubness
========================================
Purpose : Measure class separability as a function of dimensionality, and
          analyse the hubness phenomenon — the tendency of certain points to
          become universal nearest neighbours in high dimensions.

What it does:
  - Fisher Discriminant Ratio (FDR) vs number of PCA dimensions:
      FDR = (μ₁ - μ₀)ᵀ Σ⁻¹ (μ₁ - μ₀) / (D variance)
      Computed in PCA subspaces of increasing size (1 → 20 PCs).
  - Hubness analysis:
      For each point, counts how many times it appears in the k-NN list of
      other points (its "hub count"). In high dimensions, the distribution
      becomes skewed — a few "hub" points appear as neighbours of many others.
      Computed for k=5 across the full 20-dimensional space and projected
      2D/5D subspaces to compare the effect.

Input  : heart_features.csv
Output : fisher_vs_dim.png, hubness.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_FILE = os.path.join(BASE_DIR, "heart_features.csv")
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(FEATURES_FILE)
X = df.drop(columns=["target"]).values
y = df["target"].values
N, D = X.shape

# ─────────────────────────────────────────
# 1. Fisher Discriminant Ratio vs dimension
# ─────────────────────────────────────────
pca_full = PCA(n_components=D)
X_pca = pca_full.fit_transform(X)

dims = list(range(1, D + 1))
fdr_values = []

for d in dims:
    X_d = X_pca[:, :d]
    mu0 = X_d[y == 0].mean(axis=0)
    mu1 = X_d[y == 1].mean(axis=0)
    diff = mu1 - mu0

    # within-class scatter (pooled covariance)
    S_W = (np.cov(X_d[y == 0].T, ddof=1) * (y == 0).sum() +
           np.cov(X_d[y == 1].T, ddof=1) * (y == 1).sum()) / N

    if d == 1:
        # scalar case
        fdr = float(diff ** 2 / (S_W + 1e-12))
    else:
        try:
            S_W_inv = np.linalg.inv(S_W + np.eye(d) * 1e-6)
            fdr = float(diff @ S_W_inv @ diff)
        except np.linalg.LinAlgError:
            fdr = np.nan

    fdr_values.append(fdr)
    print(f"d={d:2d}  FDR={fdr:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(dims, fdr_values, marker="o", color="steelblue", linewidth=2, markersize=5)
ax.set_xlabel("Number of PCA dimensions")
ax.set_ylabel("Fisher Discriminant Ratio")
ax.set_title("Class Separability vs PCA Dimensionality")
ax.set_xticks(dims)
ax.axvline(7, color="red", linestyle="--", alpha=0.6, label="~80% variance (7 PCs)")
ax.axvline(13, color="orange", linestyle="--", alpha=0.6, label="~95% variance (13 PCs)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fisher_vs_dim.png"), dpi=150)
plt.close()
print("Saved: fisher_vs_dim.png")

# ─────────────────────────────────────────
# 2. Hubness Analysis
# ─────────────────────────────────────────
k = 5

def hub_counts(X_space):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_space)
    indices = nbrs.kneighbors(X_space, return_distance=False)[:, 1:]  # exclude self
    counts = np.zeros(len(X_space), dtype=int)
    for row in indices:
        counts[row] += 1
    return counts

spaces = {
    f"Original (D={D})": X,
    "PCA 5D": X_pca[:, :5],
    "PCA 2D": X_pca[:, :2],
}

hub_results = {name: hub_counts(Xs) for name, Xs in spaces.items()}

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, counts) in zip(axes, hub_results.items()):
    ax.hist(counts, bins=range(0, counts.max() + 2), color="steelblue",
            edgecolor="black", alpha=0.8, align="left")
    ax.set_xlabel(f"Hub count (k={k})")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Hubness — {name}")
    skewness = float(pd.Series(counts).skew())
    ax.text(0.97, 0.95, f"skewness={skewness:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    print(f"{name}: max hub count={counts.max()}, skewness={skewness:.2f}, "
          f"fraction with 0 neighbours={( counts==0).mean():.2%}")

plt.suptitle("Hubness: k-NN Hub Count Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "hubness.png"), dpi=150)
plt.close()
print("Saved: hubness.png")
