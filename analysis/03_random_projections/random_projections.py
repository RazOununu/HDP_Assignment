"""
Step 3 — Random Projections / Johnson-Lindenstrauss
=====================================================
Purpose : Demonstrate the Johnson-Lindenstrauss (JL) lemma empirically —
          random projections to a lower-dimensional space approximately
          preserve pairwise distances between points.

What it does:
  - Projects the 20-dimensional data to k = 2, 4, 6, 8, 10, 15 dimensions
    using random Gaussian matrices (each column normalised).
  - Computes all pairwise distances before and after projection.
  - Measures distortion: ε = |d_projected - d_original| / d_original
  - Plots the distribution of distortion for each k.
  - Plots mean distortion vs k to show how accuracy improves with dimension.

Key insight: even at k=10 (half the original 20 dimensions), most pairwise
distances are preserved within a small margin — demonstrating that the data
does not need all 20 dimensions to faithfully represent its geometry.

Input  : heart_features.csv
Output : figures/jl_distortion_distributions.png, jl_mean_distortion.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_FILE = os.path.join(BASE_DIR, "heart_features.csv")
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

df = pd.read_csv(FEATURES_FILE)
X = df.drop(columns=["target"]).values
N, D = X.shape

# pairwise distances in original space (upper triangle only)
dist_original = pairwise_distances(X)
upper_idx = np.triu_indices(N, k=1)
d_orig = dist_original[upper_idx]

k_values = [2, 4, 6, 8, 10, 15]

mean_distortions = []
all_distortions = {}

for k in k_values:
    # random Gaussian projection matrix, columns normalised
    R = np.random.randn(D, k) / np.sqrt(k)
    X_proj = X @ R

    dist_proj = pairwise_distances(X_proj)
    d_proj = dist_proj[upper_idx]

    # relative distortion per pair
    distortion = np.abs(d_proj - d_orig) / (d_orig + 1e-10)
    all_distortions[k] = distortion
    mean_distortions.append(distortion.mean())

    print(f"k={k:2d}  mean distortion={distortion.mean():.4f}  "
          f"median={np.median(distortion):.4f}  "
          f"90th pct={np.percentile(distortion, 90):.4f}")

# --- Plot 1: distortion distributions per k ---
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, k in zip(axes.flat, k_values):
    ax.hist(all_distortions[k], bins=40, color="steelblue",
            edgecolor="black", alpha=0.8)
    ax.axvline(all_distortions[k].mean(), color="red", linestyle="--",
               label=f"mean={all_distortions[k].mean():.3f}")
    ax.set_title(f"k = {k} dimensions")
    ax.set_xlabel("Relative distortion  ε")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
plt.suptitle("Pairwise Distance Distortion after Random Projection", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "jl_distortion_distributions.png"), dpi=150)
plt.close()
print("Saved: jl_distortion_distributions.png")

# --- Plot 2: mean distortion vs k ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_values, mean_distortions, marker="o", color="steelblue", linewidth=2)
ax.set_xlabel("Projection dimension  k")
ax.set_ylabel("Mean relative distortion  ε")
ax.set_title("Johnson-Lindenstrauss: Distortion vs Projection Dimension")
ax.set_xticks(k_values)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "jl_mean_distortion.png"), dpi=150)
plt.close()
print("Saved: jl_mean_distortion.png")
