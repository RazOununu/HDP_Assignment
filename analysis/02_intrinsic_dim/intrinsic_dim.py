"""
Step 2 — Intrinsic Dimensionality Estimation
=============================================
Purpose : Estimate the true (nonlinear) dimensionality of the dataset —
          i.e., the dimension of the manifold on which the data lies,
          as opposed to the ambient dimension (20) or the linear PCA dimension (13).

What it does:
  - TwoNN estimator  : uses the ratio of distances to the 1st and 2nd nearest
                       neighbours to infer local dimensionality.
  - MLE estimator    : maximum-likelihood estimate based on k-NN distance
                       distributions (Levina & Bickel, 2004).
  - Correlation Dimension : counts how many point-pairs fall within radius r,
                            fits a log-log curve whose slope is the dimension.

Input  : heart_features.csv
Output : 02_intrinsic_dim/twonn.png, corr_dim.png, mle.png
         prints a summary table of all three estimates
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_FILE = os.path.join(BASE_DIR, "heart_features.csv")
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(FEATURES_FILE)
X = df.drop(columns=["target"]).values
N, D = X.shape

# ─────────────────────────────────────────
# 1. TwoNN estimator
# ─────────────────────────────────────────
# For each point, compute distances to its 2 nearest neighbours.
# Under the assumption of a locally flat d-dimensional manifold,
# the ratio mu = r2/r1 follows a Pareto distribution whose shape
# parameter gives 1/d.

nbrs = NearestNeighbors(n_neighbors=3).fit(X)
distances, _ = nbrs.kneighbors(X)
r1 = distances[:, 1]   # distance to 1st neighbour
r2 = distances[:, 2]   # distance to 2nd neighbour

mu = r2 / r1
mu = mu[mu > 1]        # discard degenerate cases

# empirical CDF of mu
mu_sorted = np.sort(mu)
empirical_cdf = np.arange(1, len(mu_sorted) + 1) / len(mu_sorted)

# MLE of Pareto exponent → d = 1 / mean(log(mu))
twonn_dim = 1.0 / np.mean(np.log(mu_sorted))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.log(mu_sorted), np.log(1 - empirical_cdf + 1e-9), "o",
        markersize=3, alpha=0.6, label="Empirical")
slope_x = np.log(mu_sorted)
ax.plot(slope_x, -twonn_dim * slope_x, "r--", label=f"Fitted slope (d={twonn_dim:.2f})")
ax.set_xlabel("log(μ)  where  μ = r₂/r₁")
ax.set_ylabel("log(1 − F(μ))")
ax.set_title("TwoNN Estimator")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "twonn.png"), dpi=150)
plt.close()
print(f"TwoNN estimate       : {twonn_dim:.2f}")

# ─────────────────────────────────────────
# 2. MLE estimator (Levina & Bickel)
# ─────────────────────────────────────────
# For each point, use its k nearest neighbours.
# The MLE of local dimensionality is the reciprocal of the mean
# log-ratio of the k-th distance to each closer distance.

def mle_dim(X, k=10):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # distances[:, 0] is always 0 (self), skip it
    dists = distances[:, 1:]                      # shape (N, k)
    r_k = dists[:, -1][:, None]                   # k-th distance
    log_ratios = np.log(r_k / dists[:, :-1])      # log(r_k / r_i) for i < k
    local_dims = (k - 1) / log_ratios.sum(axis=1)
    return local_dims

k_values = range(5, 25)
mle_estimates = [mle_dim(X, k).mean() for k in k_values]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(k_values), mle_estimates, marker="o", color="steelblue")
ax.axhline(np.mean(mle_estimates), color="red", linestyle="--",
           label=f"Mean = {np.mean(mle_estimates):.2f}")
ax.set_xlabel("k (number of neighbours)")
ax.set_ylabel("Estimated intrinsic dimension")
ax.set_title("MLE Estimator (Levina & Bickel)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "mle.png"), dpi=150)
plt.close()
mle_estimate = np.mean(mle_estimates)
print(f"MLE estimate (mean)  : {mle_estimate:.2f}")

# ─────────────────────────────────────────
# 3. Correlation Dimension
# ─────────────────────────────────────────
# Count how many point-pairs have distance < r for a range of r.
# In a d-dimensional space: C(r) ~ r^d
# So: log C(r) = d * log(r) + const  → slope of log-log plot = d

from sklearn.metrics import pairwise_distances

pairwise = pairwise_distances(X)
upper = pairwise[np.triu_indices(N, k=1)]   # upper triangle only

r_values = np.logspace(np.log10(upper.min() * 1.1),
                       np.log10(upper.max() * 0.9), 40)
C_r = np.array([(upper < r).mean() for r in r_values])

# fit slope in the linear region (middle 50%)
mask = (C_r > 0.05) & (C_r < 0.95)
log_r = np.log(r_values[mask])
log_C = np.log(C_r[mask])
slope, intercept = np.polyfit(log_r, log_C, 1)
corr_dim = slope

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(np.log(r_values), np.log(np.clip(C_r, 1e-9, None)),
           s=20, color="steelblue", label="log C(r)")
ax.plot(log_r, slope * log_r + intercept, "r--",
        label=f"Slope = {corr_dim:.2f}  (correlation dim)")
ax.set_xlabel("log(r)")
ax.set_ylabel("log C(r)")
ax.set_title("Correlation Dimension")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "corr_dim.png"), dpi=150)
plt.close()
print(f"Correlation dim      : {corr_dim:.2f}")

# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────
print("\n── Intrinsic Dimensionality Summary ──")
print(f"  Ambient dimension  : {D}")
print(f"  PCA (95% var)      : 13")
print(f"  TwoNN              : {twonn_dim:.2f}")
print(f"  MLE                : {mle_estimate:.2f}")
print(f"  Correlation dim    : {corr_dim:.2f}")
