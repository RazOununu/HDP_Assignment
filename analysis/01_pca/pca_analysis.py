"""
Step 1 — PCA Analysis
======================
Purpose : Analyse the linear dimensionality structure of the dataset using
          Principal Component Analysis (PCA).

What it does:
  - Fits PCA on the full 20-dimensional feature matrix.
  - Produces a scree plot and cumulative explained variance curve to determine
    how many components are needed to capture 95% of the variance.
  - Projects the data to 2D and 3D for visual inspection, coloured by label.
  - Generates a loadings heatmap showing which original features drive each PC.

Input  : heart_features.csv
Output : figures/pca_scree.png, pca_scatter_2d.png, pca_scatter_3d.png, pca_loadings.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_FILE = os.path.join(BASE_DIR, "heart_features.csv")
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))  # figures saved next to this script

df = pd.read_csv(FEATURES_FILE)
X = df.drop(columns=["target"]).values
y = df["target"].values
feature_names = df.drop(columns=["target"]).columns.tolist()

pca_full = PCA()
pca_full.fit(X)
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)
n_components_95 = np.searchsorted(cumulative, 0.95) + 1

print(f"Dimensions          : {X.shape[1]}")
print(f"Components for 95%  : {n_components_95}")
print(f"Top-5 explained var : {explained[:5].round(3)}")

# --- Scree plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(1, len(explained) + 1), explained, color="steelblue", edgecolor="black")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Scree Plot")

axes[1].plot(range(1, len(cumulative) + 1), cumulative, marker="o", color="steelblue")
axes[1].axhline(0.95, color="red", linestyle="--", label="95% threshold")
axes[1].axvline(n_components_95, color="orange", linestyle="--", label=f"{n_components_95} components")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].set_title("Cumulative Explained Variance")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "pca_scree.png"), dpi=150)
plt.close()
print("Saved: pca_scree.png")

# --- 2D scatter ---
pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X)

colors = {0: "seagreen", 1: "indianred"}
labels_text = {0: "Healthy", 1: "Disease"}

fig, ax = plt.subplots(figsize=(7, 5))
for cls in [0, 1]:
    mask = y == cls
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=colors[cls], label=labels_text[cls], alpha=0.7, edgecolors="k", linewidths=0.3)
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
ax.set_title("PCA — 2D Projection")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "pca_scatter_2d.png"), dpi=150)
plt.close()
print("Saved: pca_scatter_2d.png")

# --- 3D scatter ---
pca3 = PCA(n_components=3)
X_3d = pca3.fit_transform(X)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
for cls in [0, 1]:
    mask = y == cls
    ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
               c=colors[cls], label=labels_text[cls], alpha=0.7, edgecolors="k", linewidths=0.2)
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({explained[2]*100:.1f}%)")
ax.set_title("PCA — 3D Projection")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "pca_scatter_3d.png"), dpi=150)
plt.close()
print("Saved: pca_scatter_3d.png")

# --- Loadings heatmap ---
# 7 components chosen because they collectively explain ~80% of total variance,
# a standard threshold that balances coverage with readability.
N_LOAD = 7
pca_load = PCA(n_components=N_LOAD)
pca_load.fit(X)
loadings = pd.DataFrame(
    pca_load.components_.T,
    index=feature_names,
    columns=[f"PC{i+1}" for i in range(N_LOAD)]
)

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(loadings.values, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
ax.set_xticks(range(N_LOAD))
ax.set_xticklabels(loadings.columns)
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels(feature_names, fontsize=8)
ax.set_title("PCA Loadings (first 7 components, covering ~80% of variance)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "pca_loadings.png"), dpi=150)
plt.close()
print("Saved: pca_loadings.png")

# --- Loadings heatmap (full — all 20 components) ---
loadings_full = pd.DataFrame(
    pca_full.components_.T,
    index=feature_names,
    columns=[f"PC{i+1}" for i in range(X.shape[1])]
)

fig, ax = plt.subplots(figsize=(18, 7))
im = ax.imshow(loadings_full.values, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels(loadings_full.columns, fontsize=8)
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels(feature_names, fontsize=8)
ax.set_title("PCA Loadings — All 20 Components")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "pca_loadings_full.png"), dpi=150)
plt.close()
print("Saved: pca_loadings_full.png")
