"""
Step 5 — Anomaly Detection
===========================
Purpose : Identify outlier patients using two complementary methods:
          Mahalanobis distance (from the assignment) and Isolation Forest
          (tree-based, model-agnostic).

What it does:
  - Mahalanobis distance: measures how far each point is from the data
    centroid, accounting for feature correlations and scale. Points with
    distance exceeding the 95th percentile are flagged as anomalies.
    Compared against the theoretical chi-squared distribution (df=D).
  - Isolation Forest: randomly partitions the feature space; points that
    are isolated in fewer splits are anomalies. contamination=0.05.
  - Compares which patients are flagged by both methods.

Input  : heart_features.csv
Output : mahalanobis.png, isolation_forest.png, comparison.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest
from sklearn.covariance import LedoitWolf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_FILE = os.path.join(BASE_DIR, "heart_features.csv")
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(FEATURES_FILE)
X = df.drop(columns=["target"]).values
y = df["target"].values
N, D = X.shape

# ─────────────────────────────────────────
# 1. Mahalanobis Distance
# ─────────────────────────────────────────
# Use Ledoit-Wolf shrinkage for a more stable covariance estimate
lw = LedoitWolf().fit(X)
cov_inv = np.linalg.inv(lw.covariance_)
mean = X.mean(axis=0)

mahal_dist = np.array([mahalanobis(x, mean, cov_inv) for x in X])
mahal_sq = mahal_dist ** 2

threshold_95 = chi2.ppf(0.95, df=D)
anomalies_mahal = mahal_sq > threshold_95

print(f"Mahalanobis — anomalies (95th pct, χ² threshold={threshold_95:.1f}): "
      f"{anomalies_mahal.sum()} / {N}")

# plot: Mahalanobis² vs chi-squared theoretical
x_chi = np.linspace(0, mahal_sq.max() * 1.1, 300)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(mahal_sq, bins=30, density=True, color="steelblue",
             edgecolor="black", alpha=0.7, label="Empirical D²")
axes[0].plot(x_chi, chi2.pdf(x_chi, df=D), "r-", linewidth=2,
             label=f"χ²(df={D}) theoretical")
axes[0].axvline(threshold_95, color="red", linestyle="--",
                label=f"95th pct = {threshold_95:.1f}")
axes[0].set_xlabel("Mahalanobis² distance")
axes[0].set_ylabel("Density")
axes[0].set_title("Mahalanobis Distance Distribution")
axes[0].legend(fontsize=8)

colors = np.where(anomalies_mahal, "indianred", "steelblue")
axes[1].scatter(range(N), mahal_sq, c=colors, s=15, alpha=0.7)
axes[1].axhline(threshold_95, color="red", linestyle="--",
                label=f"Threshold ({anomalies_mahal.sum()} anomalies)")
axes[1].set_xlabel("Patient index")
axes[1].set_ylabel("Mahalanobis² distance")
axes[1].set_title("Anomalies per Patient")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "mahalanobis.png"), dpi=150)
plt.close()
print("Saved: mahalanobis.png")

# ─────────────────────────────────────────
# 2. Isolation Forest
# ─────────────────────────────────────────
iso = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso.fit_predict(X)           # -1 = anomaly, 1 = normal
iso_scores = -iso.score_samples(X)        # higher = more anomalous
anomalies_iso = iso_labels == -1

print(f"Isolation Forest — anomalies (contamination=5%): "
      f"{anomalies_iso.sum()} / {N}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(iso_scores, bins=30, color="steelblue",
             edgecolor="black", alpha=0.7)
axes[0].axvline(iso_scores[anomalies_iso].min(), color="red",
                linestyle="--", label=f"Anomaly threshold")
axes[0].set_xlabel("Anomaly score (higher = more anomalous)")
axes[0].set_ylabel("Count")
axes[0].set_title("Isolation Forest Score Distribution")
axes[0].legend()

colors_iso = np.where(anomalies_iso, "indianred", "steelblue")
axes[1].scatter(range(N), iso_scores, c=colors_iso, s=15, alpha=0.7)
axes[1].set_xlabel("Patient index")
axes[1].set_ylabel("Anomaly score")
axes[1].set_title(f"Anomalies per Patient ({anomalies_iso.sum()} flagged)")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "isolation_forest.png"), dpi=150)
plt.close()
print("Saved: isolation_forest.png")

# ─────────────────────────────────────────
# 3. Comparison
# ─────────────────────────────────────────
both   = anomalies_mahal & anomalies_iso
only_m = anomalies_mahal & ~anomalies_iso
only_i = ~anomalies_mahal & anomalies_iso

print(f"\nComparison:")
print(f"  Flagged by both           : {both.sum()}")
print(f"  Only Mahalanobis          : {only_m.sum()}")
print(f"  Only Isolation Forest     : {only_i.sum()}")

# scatter on PC1 vs PC2
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_2d[~anomalies_mahal & ~anomalies_iso, 0],
           X_2d[~anomalies_mahal & ~anomalies_iso, 1],
           c="steelblue", s=20, alpha=0.5, label="Normal")
ax.scatter(X_2d[only_m, 0], X_2d[only_m, 1],
           c="orange", s=60, marker="^", label=f"Mahalanobis only ({only_m.sum()})")
ax.scatter(X_2d[only_i, 0], X_2d[only_i, 1],
           c="purple", s=60, marker="s", label=f"Isolation Forest only ({only_i.sum()})")
ax.scatter(X_2d[both, 0], X_2d[both, 1],
           c="indianred", s=80, marker="*", label=f"Both methods ({both.sum()})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("Anomaly Detection — Method Comparison (PCA projection)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "comparison.png"), dpi=150)
plt.close()
print("Saved: comparison.png")
